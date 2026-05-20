"""Math sanity tests for SPEED's DCT helpers + spectral_expand.

Run from the repo root with:
  python -m unittest tests.test_math -v

Doesn't require ComfyUI or a model — pure tensor math.
"""
import sys
import pathlib
import unittest

# Allow running without the comfy_api/comfy modules (the import at the top of
# speed_sampler.py would otherwise fail). Stub the missing modules so we can
# import the pure-math helpers.
class _Stub:
    def __getattr__(self, _name):
        raise AttributeError("comfy.* not available in unit test environment")

for mod_name in ("comfy", "comfy.samplers", "comfy.utils", "comfy_api",
                 "comfy_api.latest"):
    sys.modules.setdefault(mod_name, _Stub())
# comfy_api.latest.io is imported by name; needs an object with .ComfyNode etc.
class _IO:
    class ComfyNode: pass
    class Schema: pass
    class Float:
        class Input:
            def __init__(self, *a, **k): pass
    class Int:
        class Input:
            def __init__(self, *a, **k): pass
    class Combo:
        class Input:
            def __init__(self, *a, **k): pass
    class Sampler:
        class Output:
            def __init__(self, *a, **k): pass
    class NodeOutput:
        def __init__(self, *a, **k): pass
sys.modules["comfy_api.latest"].io = _IO()

# Now we can import the sampler
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from speed_sampler import (  # noqa: E402
    dct2, idct2, _cos_taper_1d, _preserve_mask_2d, initial_downscale,
    spectral_expand,
)


class TestDCTRoundTrip(unittest.TestCase):
    def test_dct_idct_identity(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, 16, 16)
        x_rt = idct2(dct2(x))
        self.assertLess((x_rt - x).abs().max().item(), 1e-4,
                        msg="DCT followed by IDCT should reconstruct input")

    def test_dct_idct_identity_5d_via_reshape(self):
        torch.manual_seed(0)
        x = torch.randn(1, 4, 3, 16, 16)
        B, C, T, H, W = x.shape
        x_4d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x_rt_4d = idct2(dct2(x_4d))
        x_rt = x_rt_4d.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        self.assertLess((x_rt - x).abs().max().item(), 1e-4)


class TestCosTaper(unittest.TestCase):
    def test_taper_zero_is_hard_truncation(self):
        w = _cos_taper_1d(8, kept=5, taper=0)
        self.assertEqual(w.tolist(), [1, 1, 1, 1, 1, 0, 0, 0])

    def test_taper_smooth(self):
        w = _cos_taper_1d(16, kept=8, taper=4)
        # First 4 are ones (kept-taper=4 entries before ramp).
        self.assertTrue(torch.allclose(w[:4], torch.ones(4)))
        # Ramp from kept-taper to kept is monotone decreasing.
        ramp = w[4:8]
        self.assertTrue(((ramp[1:] - ramp[:-1]) <= 0).all().item())
        # Outside kept = 0.
        self.assertTrue(torch.allclose(w[8:], torch.zeros(8)))


class TestSpectralExpandVariance(unittest.TestCase):
    """If we hand in a pure-noise latent at sigma=σ₀, the post-expand latent
    should still be (statistically) σ₀ * unit-noise: variance preserved up to
    the κ rescaling already in the code."""

    def _check_variance_preserved(self, scale_lo, scale_hi, taper, sigma):
        torch.manual_seed(42)
        H_full = W_full = 64
        h_lo = round(H_full * scale_lo)
        w_lo = round(W_full * scale_lo)
        # Pure noise at sigma σ:
        x_lo = sigma * torch.randn(1, 4, h_lo, w_lo)
        x_hi, sigma_aligned = spectral_expand(
            x_lo, sigma, scale_lo, scale_hi, H_full, W_full, taper=taper,
        )
        # Expected std at the new resolution should be sigma_aligned (close to σ
        # for moderate r). With the κ rescaling in spectral_expand, we should
        # observe roughly that.
        observed = x_hi.std().item()
        # Loose bound — this is a stochastic test on a small latent.
        rel_err = abs(observed - float(sigma_aligned)) / max(float(sigma_aligned), 1e-3)
        self.assertLess(rel_err, 0.4,
                        f"std={observed:.4f} vs σ_aligned={float(sigma_aligned):.4f}"
                        f" (scale {scale_lo}→{scale_hi}, taper={taper})")

    def test_variance_taper_0(self):
        self._check_variance_preserved(0.5, 0.75, taper=0, sigma=1.0)

    def test_variance_taper_8(self):
        self._check_variance_preserved(0.5, 0.75, taper=8, sigma=1.0)

    def test_variance_full_expand_taper_0(self):
        self._check_variance_preserved(0.75, 1.0, taper=0, sigma=1.0)

    def test_variance_full_expand_taper_8(self):
        self._check_variance_preserved(0.75, 1.0, taper=8, sigma=1.0)


class TestInitialDownscale(unittest.TestCase):
    def test_output_shape(self):
        x = torch.randn(1, 4, 64, 64)
        x_low = initial_downscale(x, scale=0.5, taper=4)
        self.assertEqual(tuple(x_low.shape), (1, 4, 32, 32))

    def test_taper_zero_matches_hard_truncation(self):
        torch.manual_seed(0)
        x = torch.randn(1, 4, 32, 32)
        # Hard truncation reference.
        xi = dct2(x.float())[:, :, :16, :16]
        ref = idct2(xi).to(x.dtype)
        # initial_downscale with taper=0 must match exactly.
        out = initial_downscale(x, scale=0.5, taper=0)
        self.assertLess((out - ref).abs().max().item(), 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
