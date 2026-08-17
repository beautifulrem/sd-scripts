<div align="center">

# sd-scripts-remi

**专注于 Anima 的训练与适配器研究版 `sd-scripts`。**

[English](README.md) · [简体中文](README.zh-CN.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/代码许可-Apache--2.0-blue.svg)](LICENSE.md)
![Scope](https://img.shields.io/badge/范围-Anima--only-7c3aed)
![Status](https://img.shields.io/badge/状态-实验分支-orange)

[项目亮点](#项目亮点) · [快速开始](#快速开始) · [文档导航](#文档导航) · [验证状态](#验证状态)

</div>

`sd-scripts-remi` 是 [`kohya-ss/sd-scripts`](https://github.com/kohya-ss/sd-scripts)
面向 [Anima](https://huggingface.co/circlestone-labs/Anima) 的专用分支。它保留
完整的 Anima 训练能力，移除无关模型家族，并加入 Anima 专属的适配器、数据管线、
显存、编译和蒸馏实验。

> [!IMPORTANT]
> 本仓库只支持 **Anima**。上游的 FLUX、SD3、SDXL 等训练入口已被有意移除；
> 仍保留的少量共享模块是 Anima 的必要依赖。

## 项目亮点

| 方向 | `remi` 分支新增能力 |
| --- | --- |
| 专用运行时 | 独立的 Anima flow matching、loss、模型 I/O、推理 scheduler、prompt、metadata 和 network trainer |
| 完整训练路径 | 标准 LoRA、LoHa、LoKr、全量 DiT 微调、ControlNet-LLLite、最小推理和 ComfyUI 转换 |
| 适配器控制 | SVD-Down 初始化、T-LoRA timestep mask、AdaLN LoRA、正则表达式 rank/alpha/LR、LoRA+ 和 dropout |
| 数据管线 | Free-fit Bucket、可选 resize interpolation、latent/text cache 和正确的 masked-loss 语义 |
| 显存与速度 | dynamic `torch.compile`、activation budget、QKV/KV fusion、fused AdamW、block swap 和 Unsloth activation offload |
| 训练研究 | REPA + DoG、Self-Flow、AnyFlow、DP-DMD Turbo、Soft Tokens、Hydra、Chimera 和 EasyControl |
| 工程可靠性 | checkpoint context/RNG 恢复、适配器感知的 resume/推理、ModelSpec metadata 和参数冲突检查 |

### 功能成熟度

- **核心路径：**标准 LoRA、全量 DiT 微调、ControlNet-LLLite、缓存、保存、恢复和最小推理。
- **兼容标准 LoRA 的扩展：**SVD-Down、T-LoRA、AdaLN LoRA 和分模块 rank/LR；结果仍保存为标准 LoRA。
- **研究功能：**REPA、Self-Flow、AnyFlow、DP-DMD、Soft Tokens、Hydra、Chimera 和 EasyControl；长训练前必须做小规模 A/B。

参数格式、sidecar 和互斥规则见
[`remi` 分支完整功能与参数说明](docs/remi_branch_features.md)。

## 支持的工作流

| 工作流 | 入口 | 输出 |
| --- | --- | --- |
| LoRA / LoHa / LoKr / 研究适配器 | `anima_train_network.py` | LoRA 或完整适配器 checkpoint |
| Anima DiT 全量微调 | `anima_train.py` | Anima checkpoint |
| ControlNet-LLLite 训练 | `anima_train_control_net_lllite.py` | LLLite 权重 |
| 最小推理 | `anima_minimal_inference.py` | 生成图像 |
| ControlNet-LLLite 推理 | `anima_minimal_inference_control_net_lllite.py` | 条件生成图像 |
| 转换 ComfyUI LoRA | `networks/convert_anima_lora_to_comfy.py` | ComfyUI 兼容 LoRA |

## 底模选择

角色和画风 LoRA 默认使用 **`anima-base-v1.0.safetensors`**。更新的权重是
**Anima Aesthetic v1.1**，不是新的 Base v1.1。它在结构上与本仓库兼容，
但 Anima 作者仍建议在 Base 上训练 LoRA，以保留最大的灵活性和画风服从能力。

- 在 Base v1.0 上训练，再到 Aesthetic v1.1 上验证。
- 只有部署固定为 Aesthetic v1.1，且确实希望继承其默认审美时，才直接在 v1.1 上训练。
- 普通小数据 LoRA 应冻结 Qwen3 和 LLM Adapter。

证据与对照实验方案见
[Aesthetic v1.1 底模评估](docs/anima_v11_lora_recommendation.md)。

## 快速开始

### 环境要求

- Python 3.10 或更新版本
- PyTorch 2.6 或更新版本，并与目标加速器匹配
- 真实训练验证需要 NVIDIA CUDA 环境
- Anima DiT、Qwen3-0.6B、Qwen-Image VAE 和 TOML 数据集配置

`configs/qwen3_06b/` 和 `configs/t5_old/` 只包含配置或 tokenizer 文件，
不包含模型权重。

### 安装

```bash
python -m venv venv
source venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --use-pep517 -r requirements.txt
accelerate config
```

请按实际 CUDA 版本和显卡代际选择 PyTorch，不要机械复制示例中的 index URL。

### 推荐 LoRA 基线

运行前替换所有路径。尽早保存 checkpoint，选择最早学会目标且没有破坏姿势、构图和
底模知识的版本。

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

`--network_train_unet_only` 是上游沿用的历史名称；在 Anima 中表示只把网络挂到
DiT，不训练 Qwen3。先建立这个普通 LoRA 基线，再一次只启用一个实验功能。

## 文档导航

| 主题 | 文档 |
| --- | --- |
| `remi` 全部新增功能与参数 | [功能总览](docs/remi_branch_features.md) |
| 角色/画风 LoRA、LoHa 和 LoKr | [训练研究](docs/anima_lora_training_latest_research.md) |
| Base v1.0 与 Aesthetic v1.1 | [底模评估](docs/anima_v11_lora_recommendation.md) |
| 标准 network 训练 | [Anima LoRA 文档](docs/anima_train_network.md) |
| 实验训练目标与适配器 | [高级训练](docs/anima_advanced_training.md) |
| LoHa 与 LoKr | [LoHa/LoKr 文档](docs/loha_lokr.md) |
| ControlNet-LLLite | [训练文档](docs/anima_train_control_net_lllite.md) |
| 编译与性能 | [`torch.compile` 文档](docs/anima_torch_compile.md) |
| 数据集配置 | [Dataset 文档](docs/config_README-en.md) |
| Free-fit Bucket | [`dataset_free_fit.toml`](examples/anima/dataset_free_fit.toml) |
| 验证与限制 | [验证文档](docs/validation.md) |

## 验证状态

当前 CPU/macOS 与合成适配器测试为 **138 passed、5 skipped**，并通过
`git diff --check`。测试覆盖加载、参数校验、适配器行为、保存和恢复路径。

这些结果不能代替真实 Anima 权重的 CUDA smoke test。长训练前，应在目标 GPU 上
使用小数据训练一个 epoch，并验证生成、checkpoint 保存和 resume。

## 上游、许可与模型条款

本项目派生自 `kohya-ss/sd-scripts`；作者与致谢信息请参阅上游项目。仓库代码按照
[Apache License 2.0](LICENSE.md) 分发。Anima 权重和衍生适配器受单独条款约束；
训练、分发或商业部署权重前，请核对
[Anima 官方模型卡与许可证](https://huggingface.co/circlestone-labs/Anima#license)。
