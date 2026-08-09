# sd-scripts-remi：Anima-only 训练与研究分支

`remi` 是从 [`kohya-ss/sd-scripts`](https://github.com/kohya-ss/sd-scripts)
收紧而来的 **Anima 专用训练分支**。它保留了完整的 Anima LoRA、全量 DiT
微调和 ControlNet-LLLite 路径，并将原本散落在其他模型模块中的必要逻辑整理为
独立的 `anima_*` 实现。

这个分支不只是“删减版 sd-scripts”：它还加入了 Free-fit Bucket、Anima LoRA
细粒度控制、LoHa/LoKr、REPA、Self-Flow、AnyFlow、DP-DMD，以及针对
`torch.compile`、显存和训练恢复可靠性的实验改进。

> [!IMPORTANT]
> 本仓库只面向 Anima。FLUX、SD3、SDXL 等其他模型家族的训练入口已被移除；
> 仍保留的共享代码都是 Anima 运行所需依赖。

## 分支特色

| 方向 | `remi` 分支提供的能力 |
| --- | --- |
| Anima-only 收紧 | 独立的 flow matching、loss、模型加载、推理 scheduler、prompt、metadata 和 network trainer |
| 完整训练路径 | 普通 LoRA、LoHa、LoKr、全量 DiT 微调、ControlNet-LLLite、专用推理与 ComfyUI 转换 |
| LoRA 精细控制 | SVD-Down 初始化、T-LoRA timestep mask、AdaLN LoRA、正则表达式分模块 rank/alpha/LR、LoRA+、dropout |
| 数据与分辨率 | Free-fit Bucket、可选 resize interpolation、磁盘 latent/text cache、正确的 alpha-mask 优先级 |
| 显存与速度 | dynamic `torch.compile`、activation memory budget、QKV/KV fusion、fused AdamW、block swap、Unsloth activation offload |
| 关系与一致性正则 | REPA + DoG、Self-Flow、masked loss |
| 独立实验适配器 | Soft Tokens、Hydra、Chimera、EasyControl、AnyFlow |
| 少步蒸馏 | DP-DMD Turbo 及其 critic sidecar、保存和恢复流程 |
| 工程可靠性 | checkpoint 上下文/RNG 恢复、LoHa/LoKr 从权重恢复、完整适配器推理、ModelSpec metadata 和参数冲突检查 |

功能分为三个稳定性层级：

- **常规训练路径**：普通 LoRA、全量微调、ControlNet-LLLite、缓存与基础显存优化。
- **兼容标准 LoRA 的增强**：SVD-Down、T-LoRA、AdaLN、分模块 rank/LR 等；保存结果仍可按标准 LoRA 使用。
- **实验功能**：REPA、Self-Flow、AnyFlow、DP-DMD、Soft Tokens、Hydra、Chimera、EasyControl；应先做小规模 A/B 和恢复测试。

完整参数、格式和互斥关系请看
[`remi` 分支新功能与参数总览](docs/remi_branch_features.md)。

## 支持的训练入口

| 任务 | 入口 | 主要输出 |
| --- | --- | --- |
| LoRA / LoHa / LoKr / 实验适配器 | `anima_train_network.py` | `.safetensors` 或适配器完整 checkpoint |
| 全量 Anima DiT 微调 | `anima_train.py` | Anima checkpoint |
| ControlNet-LLLite | `anima_train_control_net_lllite.py` | LLLite `.safetensors` |
| 最小推理验证 | `anima_minimal_inference.py` | 生成图像 |
| ControlNet-LLLite 推理 | `anima_minimal_inference_control_net_lllite.py` | 条件生成图像 |
| 转换为 ComfyUI LoRA | `networks/convert_anima_lora_to_comfy.py` | ComfyUI 格式 LoRA |

## 底模选择：Base v1.0 与 Aesthetic v1.1

截至 2026-08-09，官方新发布的是 **Anima Aesthetic v1.1**，不是新的
Base v1.1。官方仍明确建议使用 `anima-base-v1.0.safetensors` 训练 LoRA。

- 角色 LoRA：默认在 Base v1.0 上训练，再到 Aesthetic v1.1 上验证。
- 画风 LoRA：同样优先 Base；只有最终部署固定为 Aesthetic v1.1 且目标画风依赖其默认审美时，才建议进行 v1.1 直训 A/B。
- LoHa / LoKr：不会改变上述底模选择，先验证普通 LoRA 是否确实容量不足。
- Aesthetic v1.1 与 Base v1.0 参数拓扑兼容，本仓库可以加载和训练它；结构兼容不代表 LoRA 效果完全可迁移。

详细证据和最小对照方案见
[Aesthetic v1.1 LoRA 底模建议](docs/anima_v11_lora_recommendation.md)。

## 必需模型文件

所有训练路径都需要：

- Anima DiT `.safetensors`；普通角色/画风 LoRA 推荐官方 Base v1.0。
- Qwen3-0.6B 模型目录或 `.safetensors`。
- Qwen-Image VAE `.safetensors` 或 `.pth`。
- TOML 数据集配置。

`configs/qwen3_06b/` 和 `configs/t5_old/` 只包含配置或 tokenizer 文件，
不包含模型权重。官方建议普通小数据 LoRA 不要训练 LLM Adapter；它对文本条件影响很大，容易被破坏。

## 安装

推荐使用 Python 3.10，并先安装与目标 CUDA 环境匹配的 PyTorch，再安装其余依赖：

```bash
python -m venv venv
source venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --use-pep517 -r requirements.txt
accelerate config
```

## 推荐 LoRA 起点

下面是普通角色或画风 LoRA 的保守起点，不是固定最优解。请替换所有路径，
先进行短程 smoke test，并每 250 step 保存一次用于固定 prompt/seed 对照。

```bash
accelerate launch --num_cpu_threads_per_process 1 anima_train_network.py \
  --pretrained_model_name_or_path=/models/anima-base-v1.0.safetensors \
  --qwen3=/models/qwen_3_06b_base.safetensors \
  --vae=/models/qwen_image_vae.safetensors \
  --dataset_config=/data/anima_dataset.toml \
  --output_dir=/output --output_name=anima_lora \
  --save_model_as=safetensors --save_precision=bf16 \
  --network_module=networks.lora_anima \
  --network_train_unet_only \
  --network_dim=32 --network_alpha=32 \
  --learning_rate=2e-5 \
  --optimizer_type=AdamW \
  --optimizer_args "betas=0.9,0.99" "weight_decay=0.01" "eps=1e-8" \
  --lr_scheduler=constant_with_warmup --lr_warmup_steps=100 \
  --gradient_accumulation_steps=4 --max_grad_norm=1.0 \
  --timestep_sampling=sigmoid --sigmoid_scale=1.0 \
  --weighting_scheme=uniform --loss_type=l2 \
  --max_train_steps=1500 --save_every_n_steps=250 \
  --mixed_precision=bf16 --gradient_checkpointing \
  --cache_latents --cache_latents_to_disk \
  --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk \
  --qwen_image_vae_2d --seed=42
```

`--network_train_unet_only` 沿用了上游历史命名；在 Anima 中表示只把 LoRA
挂到 DiT，不训练 Qwen3。缓存 text-encoder 输出时必须采用这一训练方式。

建议从普通 LoRA `dim=32 / alpha=32 / lr=2e-5` 开始。不要因为分支提供了大量
实验开关就一次全部启用；先建立基线，再逐项进行 A/B。

## Free-fit Bucket

普通 bucket 从预先生成的候选分辨率中选择；Free-fit Bucket 则根据每张图片的
宽高比，在面积、步长和上下界约束内实时求出更贴合的分辨率。它可以减少裁切和
无效 padding，但会产生更多 shape，建议与本分支的 dynamic compile 支持配合使用。

```toml
[[datasets]]
resolution = [1024, 1024]
enable_bucket = true
bucket_free_fit = true
bucket_reso_steps = 16
min_bucket_reso = 512
max_bucket_reso = 2048
resize_interpolation = "lanczos"
```

示例配置见 [`examples/anima/dataset_free_fit.toml`](examples/anima/dataset_free_fit.toml)。

## 文档导航

- [`remi` 分支全部新功能与参数](docs/remi_branch_features.md)
- [Anima Aesthetic v1.1 是否适合作为 LoRA 底模](docs/anima_v11_lora_recommendation.md)
- [最新角色/画风 LoRA、LoHa、LoKr 训练研究](docs/anima_lora_training_latest_research.md)
- [Anima LoRA / network 训练](docs/anima_train_network.md)
- [高级与实验训练方法](docs/anima_advanced_training.md)
- [LoHa / LoKr](docs/loha_lokr.md)
- [Anima 全量微调参数](anima_train.py)
- [ControlNet-LLLite](docs/anima_train_control_net_lllite.md)
- [`torch.compile` 与性能优化](docs/anima_torch_compile.md)
- [数据集配置](docs/config_README-en.md)
- [Masked loss](docs/masked_loss_README.md)
- [验证说明](docs/validation.md)

## 验证边界

当前分支的 CPU/Mac 与合成适配器测试为 `138 passed, 5 skipped`，同时通过
`git diff --check`。这些测试覆盖加载、保存、恢复、参数校验和适配器行为，但不能
替代真实 Anima 权重上的 CUDA 训练验证。开始长训练前，仍应在目标 GPU 上执行一次
小数据、单 epoch、包含保存与恢复的 smoke test。

Anima 模型权重及衍生 LoRA 受其模型许可证约束；训练、分发或商业部署前请单独核对
[官方模型卡与许可证](https://huggingface.co/circlestone-labs/Anima#license)。
