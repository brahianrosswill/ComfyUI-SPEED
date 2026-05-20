import math
import torch
from tqdm.auto import trange

import comfy.samplers
import comfy.utils
from comfy_api.latest import io


# ── DCT helpers (2D separable, type-II, pure PyTorch) ────────────────────────

def _dct_matrix(n, device=None, dtype=None):
    n_range = torch.arange(n, device=device, dtype=dtype)
    k = n_range.unsqueeze(1)
    dct_mat = torch.cos(torch.pi * k * (2 * n_range + 1) / (2 * n))
    dct_mat[0] *= 1.0 / math.sqrt(n)
    dct_mat[1:] *= math.sqrt(2.0 / n)
    return dct_mat


def dct2(x):
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    D_h = _dct_matrix(H, device=device, dtype=dtype)
    D_w = _dct_matrix(W, device=device, dtype=dtype)
    x = x.reshape(B * C, H, W)
    x = D_h @ x
    x = x @ D_w.T
    return x.reshape(B, C, H, W)


def idct2(x):
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    D_h = _dct_matrix(H, device=device, dtype=dtype)
    D_w = _dct_matrix(W, device=device, dtype=dtype)
    x = x.reshape(B * C, H, W)
    x = D_h.T @ x
    x = x @ D_w
    return x.reshape(B, C, H, W)


# ── Timestep shift helpers ────────────────────────────────────────────────────

def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


# ── Spectral noise expansion ──────────────────────────────────────────────────

def spectral_expand(model_sampling, x_lo, sigma_i, scale_lo, scale_hi, target_h, target_w):
    h_hi = round(target_h * scale_hi)
    w_hi = round(target_w * scale_hi)

    if x_lo.ndim == 5:
        B, C, T, h_lo, w_lo = x_lo.shape
        x_lo_4d = x_lo.permute(0, 2, 1, 3, 4).reshape(B * T, C, h_lo, w_lo)
    else:
        B, C, h_lo, w_lo = x_lo.shape
        T = None
        x_lo_4d = x_lo

    xi = dct2(x_lo_4d.float())

    # ── EXACT MATH APPLIED DIRECTLY TO SIGMA ──
    # No shifting/unshifting required. ComfyUI's sigma is the exact noise ratio.
    sigma_val = float(sigma_i)
    r = scale_hi / scale_lo 
    
    # Paper Eq 5 & 6 substituting sigma
    sigma_aligned = (r * sigma_val) / (1.0 + (r - 1.0) * sigma_val) 
    kappa = r / (1.0 + (r - 1.0) * sigma_val) 

    xi_new = torch.zeros(x_lo_4d.shape[0], C, h_hi, w_hi,
                         device=x_lo.device, dtype=torch.float32)
    xi_new[:, :, :h_lo, :w_lo] = xi
    
    noise = torch.randn_like(xi_new)
    high_mask = torch.zeros_like(xi_new)
    high_mask[:, :, h_lo:, :] = 1.0
    high_mask[:, :, :h_lo, w_lo:] = 1.0
    
    # Inject noise perfectly matching the solver's current sigma!
    xi_new = xi_new + high_mask * sigma_val * noise

    x_new_4d = idct2(xi_new).to(x_lo.dtype)
    x_new_4d = kappa * x_new_4d

    if T is not None:
        x_new = x_new_4d.reshape(B, T, C, h_hi, w_hi).permute(0, 2, 1, 3, 4)
    else:
        x_new = x_new_4d

    # sigma_aligned is ready to be handed directly back to the solver
    sigma_out = torch.tensor(sigma_aligned, device=x_new.device, dtype=x_new.dtype)

    return x_new, sigma_out

# ── Main sampler function ─────────────────────────────────────────────────────

@torch.no_grad()
def sample_speed_euler(model, x, sigmas, extra_args=None, callback=None,
                       disable=None, scales=None):
    extra_args = {} if extra_args is None else extra_args

    if scales is None:
        n = len(sigmas) - 1
        t60 = float(sigmas[max(0, int(n * 0.4))])
        t30 = float(sigmas[max(0, int(n * 0.7))])
        scales = [(t60, 0.5), (t30, 0.75)]

    transitions = sorted(scales, key=lambda p: -p[0])
    transition_idx = 0
    H_full, W_full = x.shape[-2], x.shape[-1]

    if transitions:
        current_scale = transitions[0][1]
    else:
        current_scale = 1.0

    if current_scale < 1.0:
        h0 = round(H_full * current_scale)
        w0 = round(W_full * current_scale)
        if x.ndim == 5:
            B, C, T, _, _ = x.shape
            x_4d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H_full, W_full)
            x_dct = dct2(x_4d.float())[:, :, :h0, :w0]
            x_low = idct2(x_dct).to(x.dtype)
            x = x_low.reshape(B, T, C, h0, w0).permute(0, 2, 1, 3, 4)
        else:
            x_dct = dct2(x.float())[:, :, :h0, :w0]
            x = idct2(x_dct).to(x.dtype)

    model_sampling = model.inner_model.model_patcher.get_model_object('model_sampling')

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        was_expanded = False

        while (transition_idx < len(transitions) and
               float(sigma) <= transitions[transition_idx][0]):
            if transition_idx + 1 < len(transitions):
                next_scale = transitions[transition_idx + 1][1]
            else:
                next_scale = 1.0

            if next_scale > current_scale:
                # # print(f"[SPEED loop] EXPANDING at step {i}, sigma={float(sigma):.6f}, "
                #       f"curr_scale={current_scale} -> next_scale={next_scale}")
                # # print(f"[SPEED loop] Before expand: x mean={x.mean():.6f}, "
                #       f"std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}")
                x, sigma = spectral_expand(
                    model_sampling, x, sigma, current_scale, next_scale,
                    H_full, W_full
                )
                current_scale = next_scale
                was_expanded = True
                # # print(f"[SPEED loop] After expand: x mean={x.mean():.6f}, "
                #       f"std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}, "
                #       f"sigma={float(sigma):.6f}")

            transition_idx += 1

        # Re-space remaining sigmas after expansion so the ODE solver steps
        # smoothly from sigma_aligned to 0 (paper Section 4.3)
        if was_expanded:
            orig_val = float(sigmas[i])  # original sigma before expansion
            new_val = float(sigma)        # sigma_aligned
            if orig_val > 0 and new_val != orig_val:
                remaining = sigmas[i + 1:].clone()
                remaining = new_val * (remaining / orig_val)
                sigmas = sigmas.clone()
                sigmas[i + 1:] = remaining

        s_in_curr = x.new_ones([x.shape[0]])
        denoised = model(x, sigma * s_in_curr, **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i],
                      'sigma_hat': sigma, 'denoised': denoised})

        # if was_expanded:
            # # print(f"[SPEED loop] Expansion Euler: sigma={float(sigma):.6f}, "
            #       f"sigmas[i+1]={float(sigmas[i+1]):.6f}, dt={float(sigmas[i+1]-sigma):.6f}")
            # # print(f"[SPEED loop] denoised: mean={denoised.mean():.6f}, "
            #       f"std={denoised.std():.6f}, min={denoised.min():.6f}, max={denoised.max():.6f}")

        d = (x - denoised) / sigma
        dt = sigmas[i + 1] - sigma
        x = x + d * dt

        # if was_expanded:
        #     # print(f"[SPEED loop] After Euler: x mean={x.mean():.6f}, "
        #           f"std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}")

    # print(f"[SPEED final] x: mean={x.mean():.6f}, std={x.std():.6f}, "
        #   f"min={x.min():.6f}, max={x.max():.6f}, has_nan={torch.isnan(x).any().item()}")
    # print(f"[SPEED final] current_scale={current_scale}")

    if current_scale < 1.0:
        x = comfy.utils.common_upscale(x, W_full, H_full, "bicubic", "disabled")
        # print(f"[SPEED final] after upscale: mean={x.mean():.6f}, std={x.std():.6f}")

    return x


# ── Node class ────────────────────────────────────────────────────────────────

class SamplerSPEED(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerSPEED",
            display_name="Sampler SPEED (Spectral Progressive)",
            category="sampling/custom_sampling/samplers",
            inputs=[
                io.Float.Input("start_scale", default=0.5, min=0.1, max=1.0, step=0.05,
                               tooltip="Initial resolution fraction (0.5 = half res)"),
                io.Float.Input("mid_scale", default=0.75, min=0.1, max=1.0, step=0.05,
                               tooltip="Intermediate resolution fraction (0=skip)"),
                io.Float.Input("transition_1", default=0.6, min=0.0, max=1.0, step=0.01,
                               tooltip="Sigma fraction to expand from start to mid scale"),
                io.Float.Input("transition_2", default=0.3, min=0.0, max=1.0, step=0.01,
                               tooltip="Sigma fraction to expand from mid to full scale"),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, start_scale, mid_scale, transition_1, transition_2):
        scales = [(transition_1, start_scale), (transition_2, mid_scale)]
        sampler = comfy.samplers.KSAMPLER(
            sample_speed_euler,
            extra_options={"scales": scales}
        )
        return io.NodeOutput(sampler)
