# ComfyUI-SPEED

> Warning: This repository is "vibecoded" — it may contain experimental code, unconventional styles, or project-specific shortcuts.

This repository provides a ComfyUI custom node that integrates the SPEED (Spectral Progressive Diffusion) idea for faster sampling. SPEED is a method for progressively growing image resolution during diffusion denoising to reduce computation while preserving visual quality. (only tested for anima model)

Key references
- Original project page: https://howardxiao.ca/speed/
- Paper (PDF): https://howardxiao.ca/speed/paper/paper.pdf
- arXiv: https://arxiv.org/abs/2605.18736

Summary

Speed comparison (Anima, using the current default input config)

| SPEED sampler (this node) | Baseline (standard sampler) |
|---|---|
| ![SPEED](images/anima_speed.png)<br><br>**14.38s**<br>**1.84× faster** | ![Original](images/anima_original.png)<br><br>**26.43s**<br>**1.00×** |

Notes
- **Artifacts:** This implementation can produce visible artifacts on some outputs; results may vary by model and prompt. Inspect the example images above for a representative comparison.
- **Torch compile:** Compiling with `torch.compile` did not improve performance for this implementation and in our tests made sampling slower than running without it. It may be possible for others to make the node work with `torch.compile`, but this remains a known / open issue.

- Spectral Progressive Diffusion (SPEED) progressively increases resolution and injects higher-frequency components along the denoising trajectory, enabling training-free acceleration and a light fine-tuning recipe.
- The authors report substantial speedups (multiple×) while maintaining quality; see the paper and project page for quantitative and qualitative results.

Usage
- Connect the SPEED node output to `KSamplerAdvanced` like any other custom node in ComfyUI.
- Place this folder under your ComfyUI `custom_nodes` directory, then restart ComfyUI.
- The node is meant to be easy to wire up: pick the `Sampler SPEED (Spectral Progressive)`, sampler into `SamplerCustomAdvanced`, and use your normal prompt/model flow.

Inputs
- `start_scale`: the first resolution fraction. `0.5` means the sampler starts at half resolution, so it does the earliest denoising work on a smaller image.
- `mid_scale`: the second resolution fraction. `0.75` means it expands to 75% resolution before going to full size.
- `transition_1`: when the first expansion happens. Higher values delay the first jump; lower values make it happen earlier.
- `transition_2`: when the second expansion happens. Higher values delay the final jump to full resolution; lower values move it earlier.

Credits & license
- This implementation links to and builds on the ideas from "Spectral Progressive Diffusion for Efficient Image and Video Generation" by Howard Xiao, Brian Chao, Lior Yariv, and Gordon Wetzstein. Please see the original project page and paper for full details, authorship, and license information.

BibTeX
```
@article{xiao2026spectral,
  author    = {Xiao, Howard and Chao, Brian and Yariv, Lior and Wetzstein, Gordon},
  title     = {Spectral Progressive Diffusion for Efficient Image and Video Generation},
  year      = {2026},
}
```
