import math
import torch

import comfy.samplers
import comfy.utils
from comfy_api.latest import io


# ── DCT helpers (2D separable, type-II, pure PyTorch) ────────────────────────
# Type-II DCT matrices are cached on (n, device, dtype). Cache is tiny in
# practice (a handful of distinct resolutions per session).

_DCT_CACHE: dict = {}


def _dct_matrix(n, device=None, dtype=None):
    key = (n, str(device), str(dtype))
    cached = _DCT_CACHE.get(key)
    if cached is not None:
        return cached
    n_range = torch.arange(n, device=device, dtype=dtype)
    k = n_range.unsqueeze(1)
    dct_mat = torch.cos(torch.pi * k * (2 * n_range + 1) / (2 * n))
    dct_mat[0] *= 1.0 / math.sqrt(n)
    dct_mat[1:] *= math.sqrt(2.0 / n)
    _DCT_CACHE[key] = dct_mat
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


def dct2_partial(x, h_keep, w_keep):
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    D_h = _dct_matrix(H, device=device, dtype=dtype)[:h_keep]
    D_w = _dct_matrix(W, device=device, dtype=dtype)[:w_keep]
    x = x.reshape(B * C, H, W)
    x = D_h @ x
    x = x @ D_w.T
    return x.reshape(B, C, h_keep, w_keep)


def idct2(x):
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    D_h = _dct_matrix(H, device=device, dtype=dtype)
    D_w = _dct_matrix(W, device=device, dtype=dtype)
    x = x.reshape(B * C, H, W)
    x = D_h.T @ x
    x = x @ D_w
    return x.reshape(B, C, H, W)


# ── Cosine taper window (1D) for smooth DCT-band crossfade ───────────────────

def _cos_taper_1d(n, kept, taper, device=None, dtype=None):
    w = torch.ones(n, device=device, dtype=dtype)
    taper = max(0, min(taper, kept))
    if taper > 0:
        ramp_start = kept - taper
        idx = torch.arange(taper, device=device, dtype=dtype)
        w[ramp_start:kept] = 0.5 * (1.0 + torch.cos(torch.pi * (idx + 1.0) / taper))
    if kept < n:
        w[kept:] = 0.0
    return w


def _preserve_mask_2d(h_hi, w_hi, h_lo, w_lo, taper, device, dtype):
    w_h = _cos_taper_1d(h_hi, h_lo, taper, device, dtype)
    w_w = _cos_taper_1d(w_hi, w_lo, taper, device, dtype)
    return w_h[:, None] * w_w[None, :]


# ── Spectral noise expansion ──────────────────────────────────────────────────

def spectral_expand(x_lo, sigma_i, scale_lo, scale_hi, target_h, target_w,
                    taper=8, model_sampling=None):
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

    sigma_val = float(sigma_i)
    r = scale_hi / scale_lo

    # Convert sigma to flow-matching interpolation factor t ∈ [0, 1].
    #   Anima/Flux/SD3: sigma = t directly
    #   Cosmos:         sigma = t/(1-t), so t = sigma/(1+sigma)
    if model_sampling is not None and float(model_sampling.sigma_max) > 1.0:
        t = sigma_val / (1.0 + sigma_val)
    else:
        t = sigma_val

    # Paper Eq 5 & 6 — kappa corrects for amplitude reduction from
    # zero-padded DCT upsampling, t_aligned is the effective noise level
    # at the higher resolution.
    t_aligned = (r * t) / (1.0 + (r - 1.0) * t)
    kappa = r / (1.0 + (r - 1.0) * t)

    # Convert t_aligned back to sigma for the ODE solver
    if model_sampling is not None and float(model_sampling.sigma_max) > 1.0:
        sigma_aligned = t_aligned / (1.0 - t_aligned)
    else:
        sigma_aligned = t_aligned

    # Build the preserved-coefficient mask (1 inside core, smooth fade at edge).
    preserved_w = _preserve_mask_2d(h_hi, w_hi, h_lo, w_lo, taper,
                                    device=x_lo.device, dtype=torch.float32)

    xi_padded = torch.zeros(x_lo_4d.shape[0], C, h_hi, w_hi,
                            device=x_lo.device, dtype=torch.float32)
    xi_padded[:, :, :h_lo, :w_lo] = xi

    # Crossfade: preserved coefficients × w + new noise × (1 - w).
    # √(1 - w²) keeps total variance exactly preserved at each bin (assuming the
    # preserved coefficient ≈ Gaussian); plain (1 - w) is a touch over-noised
    # in the ring but is what the paper authors' reference code uses. Stick
    # with (1 - w) for fidelity to the published math.
    noise = torch.randn_like(xi_padded)
    noise_w = 1.0 - preserved_w
    xi_new = xi_padded * preserved_w + t * noise * noise_w

    x_new_4d = idct2(xi_new).to(x_lo.dtype)
    x_new_4d = kappa * x_new_4d

    if T is not None:
        x_new = x_new_4d.reshape(B, T, C, h_hi, w_hi).permute(0, 2, 1, 3, 4)
    else:
        x_new = x_new_4d

    sigma_out = torch.tensor(sigma_aligned, device=x_new.device, dtype=x_new.dtype)
    return x_new, sigma_out


# ── Initial DCT downscale with taper ──────────────────────────────────────────

def initial_downscale(x, scale, taper=8):
    H_full, W_full = x.shape[-2], x.shape[-1]
    h0 = round(H_full * scale)
    w0 = round(W_full * scale)

    if x.ndim == 5:
        B, C, T, _, _ = x.shape
        x_4d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H_full, W_full)
    else:
        x_4d = x

    xi_kept = dct2_partial(x_4d.float(), h0, w0)
    if taper > 0:
        w_h = _cos_taper_1d(h0, h0, taper, device=x.device, dtype=torch.float32)
        w_w = _cos_taper_1d(w0, w0, taper, device=x.device, dtype=torch.float32)
        win2d = w_h[:, None] * w_w[None, :]
        xi_kept = xi_kept * win2d
    x_low_4d = idct2(xi_kept).to(x.dtype)

    if x.ndim == 5:
        B, C, T, _, _ = x.shape
        x_low = x_low_4d.reshape(B, T, C, h0, w0).permute(0, 2, 1, 3, 4)
    else:
        x_low = x_low_4d
    return x_low


# ── Comfy sampler dispatch ────────────────────────────────────────────────────

def _get_comfy_sampler_fn(name):
    import comfy.k_diffusion.sampling as kds
    fn = getattr(kds, f"sample_{name}", None)
    if fn is None:
        raise ValueError(f"Unknown sampler '{name}'. Not found in "
                         f"comfy.k_diffusion.sampling.")
    return fn


def _list_available_samplers():
    try:
        import comfy.k_diffusion.sampling as kds
    except (ImportError, AttributeError):
        return ["euler", "euler_ancestral", "heun", "dpmpp_2m", "er_sde",
                "res_multistep"]
    excluded = {"dpm_fast", "dpm_adaptive", "lcm"}
    names = []
    for attr in dir(kds):
        if not attr.startswith("sample_"):
            continue
        short = attr[len("sample_"):]
        if short in excluded:
            continue
        names.append(short)
    return sorted(names)


# ── Main sampler function ─────────────────────────────────────────────────────


def _segment_callback(outer_cb, segment_start_idx):
    if outer_cb is None:
        return None

    def inner(d):
        d = dict(d)
        d["i"] = d.get("i", 0) + segment_start_idx
        outer_cb(d)

    return inner


@torch.no_grad()
def sample_speed(model, x, sigmas, extra_args=None, callback=None, disable=None,
                 scales=None, base_sampler="euler", taper=8):
    extra_args = {} if extra_args is None else extra_args
    sampler_fn = _get_comfy_sampler_fn(base_sampler)

    if scales is None:
        n = len(sigmas) - 1
        t60 = float(sigmas[max(0, int(n * 0.4))])
        t30 = float(sigmas[max(0, int(n * 0.7))])
        scales = [(t60, 0.5), (t30, 0.75)]

    transitions = sorted(scales, key=lambda p: -p[0])
    H_full, W_full = x.shape[-2], x.shape[-1]
    current_scale = transitions[0][1] if transitions else 1.0

    # Initial DCT downscale to first scale (if < 1) with cosine-taper edge.
    if current_scale < 1.0:
        x = initial_downscale(x, current_scale, taper=taper)

    # Walk the sigma schedule and find boundary indices where each SPEED
    # transition fires. We split sampling into segments at those boundaries.
    boundary_idxs = []
    boundary_scales = []
    transition_idx = 0
    walking_scale = current_scale
    for i in range(len(sigmas) - 1):
        while (transition_idx < len(transitions) and
               float(sigmas[i]) <= transitions[transition_idx][0]):
            next_scale = (transitions[transition_idx + 1][1]
                          if transition_idx + 1 < len(transitions) else 1.0)
            if next_scale > walking_scale:
                boundary_idxs.append(i)
                boundary_scales.append((walking_scale, next_scale))
                walking_scale = next_scale
            transition_idx += 1

    # Thread through model_sampling for Cosmos sigma→t conversion.
    model_sampling = model.inner_model.model_patcher.get_model_object('model_sampling')

    # Run sampler on each segment, applying SPEED expansion in between.
    segment_starts = [0] + boundary_idxs
    sigmas = sigmas.clone()

    for seg_i, seg_start in enumerate(segment_starts):
        seg_end = (boundary_idxs[seg_i] if seg_i < len(boundary_idxs)
                   else len(sigmas) - 1)
        seg_sigmas = sigmas[seg_start:seg_end + 1]
        if len(seg_sigmas) >= 2:
            wrapped_cb = _segment_callback(callback, seg_start)
            x = sampler_fn(model, x, seg_sigmas,
                           extra_args=extra_args,
                           callback=wrapped_cb,
                           disable=disable)

        # Apply SPEED transition (skip after final segment).
        if seg_i < len(boundary_idxs):
            old_scale, new_scale = boundary_scales[seg_i]
            x, sigma_aligned = spectral_expand(
                x, sigmas[seg_end], old_scale, new_scale,
                H_full, W_full, taper=taper, model_sampling=model_sampling,
            )
            current_scale = new_scale
            orig_val = float(sigmas[seg_end])
            new_val = float(sigma_aligned)
            if orig_val > 0 and new_val != orig_val:
                remaining = sigmas[seg_end + 1:]
                sigmas[seg_end + 1:] = new_val * (remaining / orig_val)
            sigmas[seg_end] = sigma_aligned

    # Tail-end fallback if user set transitions to never reach 1.0.
    if current_scale < 1.0:
        x = comfy.utils.common_upscale(x, W_full, H_full, "bicubic", "disabled")

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
                io.Combo.Input("base_sampler", options=_list_available_samplers(),
                               default="euler",
                               tooltip="Underlying ODE solver. Any comfy k_diffusion sampler is supported (euler, heun, dpmpp_2m, er_sde, res_multistep, dpmpp_3m_sde, uni_pc, …). euler = original SPEED behaviour. Multistep solvers (dpmpp_2m, res_multistep, er_sde, lms) reset to first-order at each SPEED transition boundary."),
                io.Float.Input("start_scale", default=0.5, min=0.1, max=1.0, step=0.05,
                               tooltip="Initial resolution fraction (0.5 = half res)"),
                io.Float.Input("mid_scale", default=0.75, min=0.1, max=1.0, step=0.05,
                               tooltip="Intermediate resolution fraction (0=skip)"),
                io.Float.Input("transition_1", default=0.8, min=0.0, max=1.0, step=0.01,
                               tooltip="Sigma fraction to expand from start to mid scale"),
                io.Float.Input("transition_2", default=0.6, min=0.0, max=1.0, step=0.01,
                               tooltip="Sigma fraction to expand from mid to full scale"),
                io.Int.Input("taper", default=8, min=0, max=32, step=1,
                             tooltip="DCT-bin width of the cosine crossfade at the preserved/noise boundary. 0 = original hard truncation (more ringing). Higher = smoother seam, slightly more low-pass."),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, base_sampler, start_scale, mid_scale, transition_1, transition_2, taper):
        scales = [(transition_1, start_scale), (transition_2, mid_scale)]
        sampler = comfy.samplers.KSAMPLER(
            sample_speed,
            extra_options={"scales": scales, "base_sampler": base_sampler, "taper": int(taper)},
        )
        return io.NodeOutput(sampler)