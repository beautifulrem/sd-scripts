<div align="center">

# sd-scripts-remi

**An Anima-only training and adapter research fork of `sd-scripts`.**

[English](README.md) · [简体中文](README.zh-CN.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/Code%20License-Apache--2.0-blue.svg)](LICENSE.md)
![Scope](https://img.shields.io/badge/Scope-Anima--only-7c3aed)
![Status](https://img.shields.io/badge/Status-Experimental-orange)

[Highlights](#highlights) · [Quick start](#quick-start) · [Documentation](#documentation) · [Validation](#validation)

</div>

`sd-scripts-remi` is a focused fork of
[`kohya-ss/sd-scripts`](https://github.com/kohya-ss/sd-scripts) for training
and studying [Anima](https://huggingface.co/circlestone-labs/Anima). It keeps
the complete Anima training surface while removing unrelated model families,
then adds Anima-specific adapter, data, memory, compilation, and distillation
experiments.

> [!IMPORTANT]
> This repository supports **Anima only**. FLUX, SD3, SDXL, and other training
> entry points from upstream are intentionally not included. Some retained
> shared modules are still required by Anima.

## Highlights

| Area | What this fork adds |
| --- | --- |
| Focused runtime | Dedicated Anima flow matching, loss, model I/O, inference scheduler, prompt, metadata, and network trainer modules |
| Complete training paths | Standard LoRA, LoHa, LoKr, full DiT fine-tuning, ControlNet-LLLite, minimal inference, and ComfyUI conversion |
| Adapter controls | SVD-Down initialization, T-LoRA timestep masks, AdaLN LoRA, regex-based rank/alpha/LR, LoRA+, and dropout |
| Data pipeline | Free-fit buckets, configurable resize interpolation, latent/text caches, and corrected masked-loss semantics |
| Memory and speed | Dynamic `torch.compile`, activation budgets, QKV/KV fusion, fused AdamW, block swap, and Unsloth activation offload |
| Research objectives | REPA + DoG, Self-Flow, AnyFlow, DP-DMD Turbo, Soft Tokens, Hydra, Chimera, and EasyControl |
| Reliability | Context/RNG-safe checkpointing, adapter-aware resume and inference, ModelSpec metadata, and explicit incompatibility checks |

### Feature maturity

- **Core paths:** standard LoRA, full DiT fine-tuning, ControlNet-LLLite,
  caching, saving, resuming, and minimal inference.
- **Standard-compatible extensions:** SVD-Down, T-LoRA, AdaLN LoRA, and
  per-module rank/LR controls. These still save standard LoRA weights.
- **Research features:** REPA, Self-Flow, AnyFlow, DP-DMD, Soft Tokens, Hydra,
  Chimera, and EasyControl. Validate these with small A/B runs before a long job.

See the [complete `remi` feature and parameter reference](docs/remi_branch_features.md)
for formats, sidecars, and incompatibility rules.

## Supported workflows

| Workflow | Entry point | Output |
| --- | --- | --- |
| LoRA / LoHa / LoKr / research adapters | `anima_train_network.py` | LoRA or full adapter checkpoint |
| Full Anima DiT fine-tuning | `anima_train.py` | Anima checkpoint |
| ControlNet-LLLite training | `anima_train_control_net_lllite.py` | LLLite weights |
| Minimal inference | `anima_minimal_inference.py` | Generated images |
| ControlNet-LLLite inference | `anima_minimal_inference_control_net_lllite.py` | Conditioned images |
| ComfyUI LoRA conversion | `networks/convert_anima_lora_to_comfy.py` | ComfyUI-compatible LoRA |

## Model choice

Use **`anima-base-v1.0.safetensors`** as the default base for character and
style LoRA training. The newer checkpoint is **Anima Aesthetic v1.1**, not a
new Base v1.1. It is structurally compatible with this repository, but the
Anima author still recommends training LoRAs on Base for maximum flexibility
and style adherence.

- Train on Base v1.0, then validate the LoRA on Aesthetic v1.1.
- Train directly on Aesthetic v1.1 only when deployment is fixed to that
  checkpoint and inheriting its default aesthetic is intentional.
- Keep Qwen3 and the LLM Adapter frozen for ordinary small-dataset LoRAs.

Read the [Aesthetic v1.1 base-model assessment](docs/anima_v11_lora_recommendation.md)
for the evidence and a controlled A/B plan.

## Quick start

### Requirements

- Python 3.10 or later
- PyTorch 2.6 or later with a build matching the target accelerator
- An NVIDIA CUDA environment for real training validation
- Anima DiT, Qwen3-0.6B, Qwen-Image VAE, and a TOML dataset config

The `configs/qwen3_06b/` and `configs/t5_old/` directories contain config and
tokenizer files, not model weights.

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --use-pep517 -r requirements.txt
accelerate config
```

Install the PyTorch build appropriate for your CUDA and GPU generation instead
of copying the example index URL blindly.

### Recommended LoRA baseline

Replace every path before running. Save early checkpoints and select the first
one that learns the target without damaging pose, composition, or base-model
knowledge.

```bash
accelerate launch --num_cpu_threads_per_process 1 anima_train_network.py \
  --pretrained_model_name_or_path=/models/anima-base-v1.0.safetensors \
  --qwen3=/models/qwen_3_06b_base.safetensors \
  --vae=/models/qwen_image_vae.safetensors \
  --dataset_config=/data/anima_dataset.toml \
  --output_dir=/output --output_name=anima_lora \
  --network_module=networks.lora_anima --network_train_unet_only \
  --network_dim=32 --network_alpha=32 --learning_rate=2e-5 \
  --optimizer_type=AdamW --lr_scheduler=constant_with_warmup \
  --lr_warmup_steps=100 --gradient_accumulation_steps=4 \
  --timestep_sampling=sigmoid --weighting_scheme=uniform --loss_type=l2 \
  --max_train_steps=1500 --save_every_n_steps=250 \
  --mixed_precision=bf16 --save_precision=bf16 --gradient_checkpointing \
  --cache_latents --cache_latents_to_disk \
  --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk \
  --qwen_image_vae_2d --seed=42
```

`--network_train_unet_only` is the upstream legacy name. For Anima, it means
that the network is attached only to the DiT, not Qwen3. Start with this plain
LoRA baseline; enable one experimental feature at a time.

## Documentation

| Topic | Guide |
| --- | --- |
| All `remi` additions and parameters | [Feature reference](docs/remi_branch_features.md) |
| Character/style LoRA, LoHa, and LoKr | [Training research](docs/anima_lora_training_latest_research.md) |
| Base v1.0 vs Aesthetic v1.1 | [Model-choice assessment](docs/anima_v11_lora_recommendation.md) |
| Standard network training | [Anima LoRA guide](docs/anima_train_network.md) |
| Experimental objectives and adapters | [Advanced training](docs/anima_advanced_training.md) |
| LoHa and LoKr | [LoHa/LoKr guide](docs/loha_lokr.md) |
| ControlNet-LLLite | [Training guide](docs/anima_train_control_net_lllite.md) |
| Compilation and performance | [`torch.compile` guide](docs/anima_torch_compile.md) |
| Dataset configuration | [Dataset guide](docs/config_README-en.md) |
| Free-fit bucket example | [`dataset_free_fit.toml`](examples/anima/dataset_free_fit.toml) |
| Validation and limitations | [Validation guide](docs/validation.md) |

## Validation

The current CPU/macOS and synthetic-adapter suite passes **138 tests with 5
skipped**, together with `git diff --check`. It covers loading, parameter
validation, adapter behavior, saving, and resume paths.

This is **not** a substitute for a CUDA smoke test with real Anima weights.
Before a long run, train a small dataset for one epoch on the target GPU and
verify generation, checkpoint save, and resume.

## Upstream, license, and model terms

This project is derived from `kohya-ss/sd-scripts`; see the upstream project
for its authors and credits. Repository code is distributed under the
[Apache License 2.0](LICENSE.md). Anima model weights and derivative adapters
have separate terms. Review the
[official Anima model card and license](https://huggingface.co/circlestone-labs/Anima#license)
before training, distributing, or commercially deploying weights.
