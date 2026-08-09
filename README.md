# Anima training scripts

This checkout is a reduced, Anima-only variant of `kohya-ss/sd-scripts`.
It keeps all Anima training modes and the shared modules required by those
trainers. Other model families and unrelated utilities have been removed.

## Training entry points

| Training mode | Entry point | Output |
| --- | --- | --- |
| Full DiT fine-tuning | `anima_train.py` | Anima model checkpoint |
| LoRA / network training | `anima_train_network.py` | LoRA `.safetensors` |
| ControlNet-LLLite | `anima_train_control_net_lllite.py` | LLLite `.safetensors` |

Minimal inference scripts are retained to validate trained weights:

- `anima_minimal_inference.py`
- `anima_minimal_inference_control_net_lllite.py`
- `networks/convert_anima_lora_to_comfy.py`

## Required model files

All training modes require:

- Anima DiT `.safetensors`
- Qwen3-0.6B model directory or `.safetensors`
- Qwen-Image VAE `.safetensors` or `.pth`
- a TOML dataset configuration

The bundled `configs/qwen3_06b/` and `configs/t5_old/` directories contain
configuration/tokenizer files, not the model weights.

## Installation

Use Python 3.10 and install the PyTorch build appropriate for the target CUDA
environment first. Then install the remaining dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --use-pep517 -r requirements.txt
accelerate config
```

## LoRA smoke test

The following is a starting template. Replace every placeholder path and run a
one-epoch smoke test before a long job.

```bash
accelerate launch --num_cpu_threads_per_process 1 anima_train_network.py \
  --pretrained_model_name_or_path=/path/to/anima.safetensors \
  --qwen3=/path/to/qwen3-0.6b \
  --vae=/path/to/qwen_image_vae.safetensors \
  --dataset_config=/path/to/dataset.toml \
  --output_dir=/path/to/output \
  --output_name=anima_lora \
  --save_model_as=safetensors \
  --network_module=networks.lora_anima \
  --network_dim=8 \
  --network_alpha=1 \
  --network_train_unet_only \
  --learning_rate=1e-4 \
  --optimizer_type=AdamW8bit \
  --lr_scheduler=constant \
  --timestep_sampling=sigmoid \
  --mixed_precision=bf16 \
  --gradient_checkpointing \
  --cache_latents \
  --cache_text_encoder_outputs \
  --qwen_image_vae_2d \
  --max_train_epochs=1
```

`--network_train_unet_only` uses the upstream legacy option name; for Anima it
means that LoRA is applied only to the DiT side. It is required when text
encoder outputs are cached.

## Documentation

- [Anima LoRA training](docs/anima_train_network.md)
- [Anima-specific optimization and experimental training](docs/anima_advanced_training.md)
- [Anima full fine-tuning and common options](anima_train.py)
- [Anima ControlNet-LLLite training](docs/anima_train_control_net_lllite.md)
- [Anima `torch.compile`](docs/anima_torch_compile.md)
- [Dataset configuration](docs/config_README-en.md)
- [Fine-tuning metadata format](docs/dataset_metadata.md)
- [Validation](docs/validation.md)
- [Masked loss](docs/masked_loss_README.md)

The runtime has been narrowed to Anima. Flow matching, inference scheduling,
prompt parsing, metadata, loss handling, CLI arguments, and the LoRA trainer
base now live in dedicated `anima_*` modules; the former FLUX, SD3, SD/SDXL,
and generic `train_network.py` implementations are not retained.
