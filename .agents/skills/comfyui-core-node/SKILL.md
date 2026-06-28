---
name: comfyui-core-node
description: >
  Expert skill for developing ComfyUI custom nodes that touch ComfyUI's internal
  architecture — model management, attention patching, sampling pipeline, latent ops,
  text encoders, model architecture integration, and activation/dataflow hooks.
  Use this skill whenever the task involves: implementing a research paper into ComfyUI,
  wrapping a new model architecture, writing custom samplers or attention processors,
  patching ComfyUI's forward pass, or any node development that goes beyond simple
  tensor math and requires understanding of ComfyUI source code. Triggers on phrases
  like "implement this paper in ComfyUI", "write a custom node for X model", "patch
  ComfyUI attention", "custom sampler node", "custom VAE node", "model architecture
  in ComfyUI", "comfy node that does [non-trivial ML operation]", "ModelPatcher",
  "transformer_options", "model_options", "set_model_patch", "CFGGuider",
  "load_checkpoint", "ComfyUI GGUF", "IP-Adapter node", "LoRA patching node",
  "attention injection", "noise schedule", "sigma", "latent processing node".
  Always invoke this skill for any ComfyUI node work that touches model internals —
  do not attempt ComfyUI core integration from general knowledge alone, because the
  internal API evolves frequently and stale patterns will produce broken nodes.
---

# ComfyUI Core Node Development

This skill guides you through developing custom ComfyUI nodes that interact with ComfyUI's internal architecture. It covers everything from simple utility nodes to complex model patches, custom samplers, and research paper implementations.

## Bootstrap: Environment Setup

Before writing any code or referencing any source, run this setup:

```bash
ln -sfn ~/ComfyUI/comfy/ /comfy
```

This creates a stable symlink so all source lookups use `/comfy/` as root. Every file reference in this skill uses this path.

**Why this matters**: ComfyUI's internal API is not stable across versions. The source at `/comfy/` is the only authoritative truth. Your training data about ComfyUI internals may be outdated. When in doubt, `grep /comfy/` and read what you find.

## Best Practices

These rules exist because real-world ComfyUI node development has specific failure modes that are hard to debug if you don't know about them upfront.

**BP-01: Read before writing.** Before implementing any node that touches a ComfyUI subsystem, grep `/comfy/` for the relevant module and read it. ComfyUI's internals evolve frequently — patterns that worked three months ago may be wrong today. This is the single most important rule.

**BP-02: Clone before patching.** When using `ModelPatcher`, always call `.clone()` and patch the clone. The original `ModelPatcher` is shared across the graph — mutating it in place causes unpredictable cross-node interference. The `clone()` method is lightweight because it shares the underlying model weights.

**BP-03: Respect dtype and device.** Tensors in ComfyUI can be fp32, fp16, bf16, or quantized (GGUF, NF4, etc). Custom ops must be dtype-agnostic or explicitly cast with `.to(dtype)`. Never assume tensors are on CPU — they may be on any CUDA device, and ComfyUI manages device placement through `model_management`.

**BP-04: Preserve the LATENT contract.** The `LATENT` type is always a dict with at least `"samples"` (a tensor). It may also contain `"noise_mask"`, `"batch_index"`, and other keys. Return it as a dict. Never return a raw tensor where a LATENT is expected, and preserve any keys you don't modify.

**BP-05: Use `model_options` for sampler-time hooks.** Don't monkey-patch model attributes at node execution time. Use `model_options["transformer_options"]` and the `set_model_patch` / `set_model_patch_replace` methods so patches are scoped to a single sampling call and don't leak state.

**BP-06: Implement only the delta.** When implementing a paper, identify the closest existing ComfyUI module, read its forward pass, identify exactly where the paper's contribution diverges, and implement only that divergence. Full reimplementations are fragile and miss ComfyUI-specific handling (dtype casting, device management, nested tensors, etc).

**BP-07: Node outputs are lazy.** Never perform heavy computation in `INPUT_TYPES` or `__init__`. All work happens in the execute method (the function named by `FUNCTION`). `INPUT_TYPES` is called frequently for UI rendering.

**BP-08: Surface errors with meaning.** Use `raise ValueError("NodeName: <specific problem>")` so users see which node failed and why. Raw tracebacks through ComfyUI's execution engine are hard to trace back to the source.

**BP-09: Test with a minimal graph.** For sampling/attention patches, test with a minimal `KSampler` graph before integrating into a complex workflow. This isolates whether your patch works from whether the workflow wiring is correct.

**BP-10: Avoid name collisions.** Check your `NODE_CLASS_MAPPINGS` keys against existing installed nodes. Name collisions silently override the previous node with no warning.

## Domain Reference Files

This skill is organized by domain. Read the relevant reference file before implementing in that domain:

| Domain | File | Read when... |
|--------|------|-------------|
| Node registration & type system | `references/node_registration.md` | Writing any new node — INPUT_TYPES, RETURN_TYPES, IS_CHANGED, CATEGORY |
| Model management & loading | `references/model_management.md` | Loading checkpoints, wrapping foreign models, ModelPatcher usage |
| Attention mechanisms & patching | `references/attention_patching.md` | Injecting attention processors, cross/self-attention hooks, IP-Adapter style patches |
| Sampling pipeline | `references/sampling_pipeline.md` | Custom samplers, CFGGuider, noise schedules, post-CFG callbacks |
| Latent space operations | `references/latent_ops.md` | LATENT type handling, VAE encode/decode, mask-aware operations |
| Conditioning & text encoders | `references/conditioning.md` | CLIP/T5 conditioning structure, ControlNet injection, conditioning manipulation |
| Model architecture integration | `references/model_architecture.md` | Registering new architectures, supported_models pattern, wrapping diffusers models |
| Dataflow & activation patching | `references/activation_patching.md` | set_model_patch, forward-pass hooks, patcher_extension wrappers |

## Paper-to-Code Workflow

When implementing a research paper into ComfyUI, follow this decision tree:

### Step 1: Parse the contribution

Read the paper (or description) and classify what it changes:
- **Attention mechanism** → read `references/attention_patching.md`
- **Sampling/guidance** → read `references/sampling_pipeline.md`
- **Model architecture** → read `references/model_architecture.md`
- **Conditioning/encoding** → read `references/conditioning.md`
- **Latent/VAE processing** → read `references/latent_ops.md`

Is there a reference implementation? If yes, read it. If no, proceed to Step 2.

### Step 2: Identify the insertion point

```bash
# Find the relevant operation in ComfyUI source
grep -rn "relevant_function_or_class" /comfy/
```

Read the actual forward pass to understand:
- What tensor shapes flow through
- What dtype/device assumptions exist
- What `model_options` / `transformer_options` keys are already used
- Where hook points exist

### Step 3: Determine integration strategy

| Paper changes... | Integration approach |
|-----------------|---------------------|
| Attention computation | `model_patcher.set_model_attn1_patch()` or `set_model_attn2_patch()` via `model_options["transformer_options"]` |
| Sampling/CFG logic | Custom `SAMPLER` node, or `set_model_options_post_cfg_function` / `set_model_options_pre_cfg_function` |
| Model architecture | New class in `supported_models` pattern, or `ModelPatcher` extension |
| Latent/VAE processing | New node that takes `LATENT` input, processes `samples` tensor, returns `LATENT` dict |
| Conditioning | Manipulate the `CONDITIONING` type (list of `[tensor, dict]` pairs) |
| Forward-pass activation | `patcher_extension` wrappers (`WrappersMP.DIFFUSION_MODEL`, `WrappersMP.APPLY_MODEL`, etc) |

### Step 4: Implement the delta

Write only what the paper changes, not a full reimplementation. Use existing ComfyUI infrastructure for everything else.

### Step 5: Wire into the type system

Define `INPUT_TYPES`, `RETURN_TYPES`, `RETURN_NAMES`, `FUNCTION`, `CATEGORY` following the patterns in `references/node_registration.md`.

### Step 6: Add cache invalidation

Add `IS_CHANGED` if the node has non-tensor state that affects outputs (e.g., random seeds, external file timestamps, configuration that doesn't flow through inputs).

## File Structure

For any non-trivial implementation, organize the custom node package like this:

```
custom_nodes/
└── comfyui_<name>/
    ├── __init__.py           # NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    ├── nodes.py              # Node class definitions
    ├── model_utils.py        # Model loading / patching helpers (if needed)
    ├── attention_patches.py  # Attention modification logic (if needed)
    ├── sampling.py           # Custom sampler logic (if needed)
    └── README.md             # What the nodes do, expected inputs/outputs
```

The `__init__.py` is the entry point — ComfyUI discovers nodes by importing this module and reading `NODE_CLASS_MAPPINGS`.

```python
# __init__.py pattern
from .nodes import MyNode, MyOtherNode

NODE_CLASS_MAPPINGS = {
    "MyNode": MyNode,
    "MyOtherNode": MyOtherNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyNode": "My Node (PackName)",
    "MyOtherNode": "My Other Node (PackName)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

## GitHub Reference Repositories

These repos demonstrate patterns at different complexity levels. Read the actual source when relevant — don't rely on cached knowledge of these repos, as they evolve alongside ComfyUI:

| Repo | What to learn from it |
|------|----------------------|
| `comfyanonymous/ComfyUI` | Ground truth — core source for model_management, samplers, latent ops |
| `blepping/comfyui_jankhidiffusion` | Attention map patching at inference time, UNet forward pass hooks |
| `city96/ComfyUI-GGUF` | Custom model loader, dtype/quantization-aware model wrapping |
| `kijai/ComfyUI-KJNodes` | Wide variety of latent/conditioning/sampling utility nodes |
| `cubiq/ComfyUI_IPAdapter_plus` | Attention processor injection, multi-model conditioning |
| `Extraltodeus/ComfyUI-AutomaticCFG` | CFG/sampler post-processing hooks, custom noise schedulers |
| `pythongosssss/ComfyUI-Custom-Scripts` | Node metadata, widget lifecycle, frontend integration |
| `ltdrdata/ComfyUI-Impact-Pack` | Complex multi-node workflows, mask/latent region routing |

To fetch and study a reference repo:
```bash
# Clone to a temp location and read the relevant module
git clone --depth 1 https://github.com/<owner>/<repo>.git /tmp/<repo>
# Then grep/read the specific patterns you need
```

## Quick Reference: Key Source Files

When you need to understand a subsystem, start with these files:

| What you need | Read this file |
|--------------|---------------|
| Model loading & device management | `/comfy/model_management.py` |
| ModelPatcher (clone, patch, options) | `/comfy/model_patcher.py` |
| Sampling pipeline | `/comfy/samplers.py`, `/comfy/sample.py` |
| Attention abstraction | `/comfy/ldm/modules/attention.py` |
| Supported model architectures | `/comfy/supported_models.py`, `/comfy/supported_models_base.py` |
| Model base classes | `/comfy/model_base.py` |
| Checkpoint loading | `/comfy/sd.py` |
| Hooks and patcher extensions | `/comfy/hooks.py`, `/comfy/patcher_extension.py` |
| LoRA loading and weight calculation | `/comfy/lora.py` |
| Latent formats | `/comfy/latent_formats.py` |
| CLIP/text encoders | `/comfy/sd1_clip.py`, `/comfy/text_encoders/` |
| Conditioning processing | `/comfy/conds.py` |
| Model sampling (sigma/noise) | `/comfy/model_sampling.py` |
| ControlNet | `/comfy/controlnet.py` |
| Quantization ops | `/comfy/quant_ops.py` |
| Built-in nodes (patterns) | `~/ComfyUI/nodes.py` |
