# `remi` 分支新功能与参数总览

> 更新日期：2026-08-09  
> 比较基线：`upstream/main` (`37a1cbb`) → `remi` (`0ed5338`)  
> 适用范围：Anima LoRA / LoHa / LoKr、全量 DiT 微调、ControlNet-LLLite、专用推理和实验性适配器

## 1. 分支定位

`remi` 是从 `kohya-ss/sd-scripts` 收紧而来的 **Anima-only** 实验分支。它不只是删除其他模型脚本，还把 Anima 训练所需的 flow matching、loss、模型加载、保存、推理 scheduler、prompt 解析和 network trainer 收入独立的 `anima_*` 模块，并新增了一组 Anima 专用实验功能。

保留的主要入口：

| 任务 | 入口 |
| --- | --- |
| Anima 全量 DiT 微调 | `anima_train.py` |
| LoRA / LoHa / LoKr / 自定义适配器 | `anima_train_network.py` |
| ControlNet-LLLite | `anima_train_control_net_lllite.py` |
| Anima 推理与适配器验证 | `anima_minimal_inference.py` |
| ControlNet-LLLite 推理 | `anima_minimal_inference_control_net_lllite.py` |
| LoRA 转 ComfyUI | `networks/convert_anima_lora_to_comfy.py` |

## 2. 功能分级

| 级别 | 含义 | 典型功能 |
| --- | --- | --- |
| 常规增强 | 仍保存标准 LoRA，适合普通角色/画风实验 | SVD-Down、T-LoRA、AdaLN LoRA、分模块 rank/LR/alpha |
| 工程优化 | 主要影响速度、显存或数据处理 | Free-fit bucket、dynamic compile、QKV fusion、activation budget |
| 附加正则 | 在普通 flow-matching loss 上加目标 | REPA + DoG、Self-Flow、masked loss |
| 独立适配器 | 不是普通 LoRA，需要完整 checkpoint 和专用推理协议 | AnyFlow、Soft Tokens、Hydra、Chimera、EasyControl |
| 蒸馏任务 | 目标是少步或任意区间推理，不应混入普通角色/画风 LoRA | DP-DMD Turbo、AnyFlow |

## 3. Anima 专用基础模块

分支将原先分散在 FLUX、SD3 和通用 trainer 中的 Anima 依赖拆分为：

| 模块 | 职责 |
| --- | --- |
| `library/anima_args.py` | Anima-only CLI、dataset、optimizer、保存、Huber/masked-loss 参数 |
| `library/anima_flow_matching.py` | Rectified Flow scheduler、timestep sampling、shift/offset、加噪过程 |
| `library/anima_inference_scheduler.py` | 推理时的 Anima flow scheduler |
| `library/anima_loss.py` | L1/L2/Huber、masked loss、加权 loss 正确归约 |
| `library/anima_model_io.py` | Anima safetensors、hash 和 metadata 保存 |
| `library/anima_model_spec.py` | Anima/LoRA 的 ModelSpec metadata |
| `library/anima_prompt_utils.py` | 推理和采样 prompt 参数解析 |
| `library/anima_network_trainer.py` | 收紧后的 Anima network trainer |
| `library/anima_advanced_training.py` | REPA/DoG 和 Self-Flow 数学辅助函数 |
| `library/anima_repa.py` | REPA sidecar 几何与插值 metadata 校验 |

这些模块的目标是让 Anima 训练不再依赖大量 FLUX/SD3/SDXL 代码，同时保留 Anima 实际需要的共享底层。

## 4. 标准 LoRA 增强

以下参数使用：

```bash
--network_module=networks.lora_anima --network_args "key=value" ...
```

### 4.1 SVD-Down 初始化

```bash
--network_args "down_init=weight_svd"
```

- 对冻结 Linear 权重做低秩随机 SVD，用主右奇异子空间初始化 `lora_down`。
- `lora_up` 仍为零，因此初始输出与底模完全一致。
- 保存为标准 LoRA，不需要专用推理器。
- 当前只支持 Linear 层的 weight-SVD。

### 4.2 T-LoRA timestep rank mask

```bash
--network_args \
  "use_timestep_mask=true" \
  "min_rank=4" \
  "alpha_rank_scale=1.0"
```

- 高噪声阶段只激活较少 rank channel。
- 越接近干净数据，激活 rank 越接近完整 `network_dim`。
- `min_rank` 限制最小活跃 rank；`alpha_rank_scale` 控制 rank 曲线的非线性。
- 采样、验证和推理时使用完整 rank，保存格式仍为标准 LoRA。
- 已为 gradient-checkpoint recomputation 保存/恢复当前 mask context。

### 4.3 AdaLN modulation LoRA

```bash
--network_args \
  "train_adaln=true" \
  "adaln_rank=16" \
  "adaln_alpha=16" \
  "adaln_lr=2e-5"
```

- 默认 LoRA 会排除 Anima AdaLN modulation；该开关可显式纳入。
- AdaLN 拥有独立 rank、alpha 和 LR。
- 仍保存成标准 LoRA 键。
- 对整体画风、色调和全局调制有研究价值，对普通角色 LoRA 应默认关闭。

### 4.4 分模块 rank / alpha / learning rate

```bash
--network_args \
  "network_reg_dims=.*self_attn.*=32,.*cross_attn.*=32,.*mlp.*=16" \
  "network_reg_alphas=.*self_attn.*=32,.*cross_attn.*=32,.*mlp.*=16" \
  "network_reg_lrs=.*self_attn.*=2e-5,.*cross_attn.*=2e-5,.*mlp.*=1e-5"
```

- 正则表达式对原始模块全名做 `re.fullmatch()`。
- 可分别控制 self-attention、cross-attention、MLP 和 AdaLN。
- 分模块设定优先于全局 `network_dim/network_alpha/learning_rate`。
- 对“角色学会了，但训练集画风泄漏”的情况，可先降低 MLP LR，不必直接删掉全部 MLP。

### 4.5 通道缩放 / 梯度重平衡

```bash
--network_args \
  "channel_scaling_alpha=0.5" \
  "channel_scaling_stats=/path/to/stats.safetensors"
```

- 读取每个 LoRA 模块的 `mean(abs(x))` 输入通道校准统计。
- 对通道做 SmoothQuant-style 梯度重平衡。
- 保存时烘焙回标准 LoRA，不引入推理开关。
- 当前分支实现了统计消费和烘焙，但尚没有完整的用户级统计采集 CLI；因此仍属开发者功能。

### 4.6 Dropout、LoRA+ 和模块选择

保留并加强：

- `rank_dropout`
- `module_dropout`
- `include_patterns`
- `exclude_patterns`
- `loraplus_lr_ratio`
- `loraplus_unet_lr_ratio`
- `loraplus_text_encoder_lr_ratio`

恢复 checkpoint 时会重建 LoRA+ 学习率比例和各模块属性，避免“首轮有效，resume 后静默失效”。

## 5. LoHa / LoKr 原生增强

`networks.loha` 和 `networks.lokr` 现在可以直接作用于 Anima DiT。

共同支持：

- Linear 和 Conv2d 1x1。
- Conv2d 3x3+ 的 flat 或 Tucker 分解：`conv_dim`、`conv_alpha`、`use_tucker=True`。
- `rank_dropout`、`module_dropout`。
- `include_patterns`、`exclude_patterns`。
- `network_reg_dims`、`network_reg_lrs`。
- `train_llm_adapter=True`（官方建议普通小数据训练不要开）。
- LoRA+ 学习率分组。
- 从权重恢复 dropout、正则 rank/LR 和 LoRA+ 设定。

LoKr 额外支持：

```bash
--network_args "factor=-1"
```

- `factor=-1` 自动寻找平衡因子。
- 正整数可强制分解，例如 `factor=4` 或 `factor=8`。
- 恢复旧 checkpoint 时，如果 metadata 没有 factor，会从 `lokr_w1` 形状与底模维度推断兼容 factor，防止 shape mismatch。
- 当 rank 足够大时，第二个 Kronecker 因子会转为 full matrix，并显式记录警告。

参见 [LoHa / LoKr 文档](loha_lokr.md)。

## 6. Free-fit bucket 与数据路径

### 6.1 Free-fit bucket

```toml
[[datasets]]
resolution = [1024, 1024]
enable_bucket = true
bucket_free_fit = true
bucket_reso_steps = 16
min_bucket_reso = 512
max_bucket_reso = 2048
```

- 以配置的目标像素面积为基准，为每张图生成近似原生宽高比的 bucket。
- 尽量减少 crop，允许 upscale。
- 与 `bucket_no_upscale` 互斥。
- 不同宽高比可产生不同 token 数，建议与 dynamic-sequence compile 配合。

示例：[dataset_free_fit.toml](../examples/anima/dataset_free_fit.toml)。

### 6.2 resize interpolation

dataset 可显式指定：

```toml
resize_interpolation = "lanczos"
```

可选：`lanczos`、`nearest`、`bilinear/linear`、`bicubic/cubic`、`area`。REPA sidecar 会保存并校验该字段，避免用 bicubic 缓存却用 lanczos 训练的静默不一致。

### 6.3 Masked loss 收紧

```bash
--masked_loss
```

- 支持 alpha mask。
- 当 batch 同时包含 `alpha_masks` 和 ControlNet/EasyControl 条件图时，优先使用显式 alpha mask。
- `--normalize_alpha_mask_loss` 是显式可选开关：按每个样本的 alpha mask 平均覆盖率归一化，使不同 mask 面积的样本具有可比的 loss 尺度；默认关闭以保持旧训练语义。
- ControlNet-LLLite 数据集支持在 TOML subset 中设置 `alpha_mask = true`，并把目标图 alpha 沿配置、dataset、batch 链路传入 loss；`conditioning_data_dir` 始终是控制图，不会被误当作 mask。
- 只在没有 alpha mask 且调用者声明条件图就是 mask 时，才将 `conditioning_images[:, 0]` 当作 mask。
- mask 会用 area interpolation 缩放到 latent loss 尺寸。

### 6.4 Caption 与 text-encoder cache 约束

- 固定 caption 可使用 `--cache_text_encoder_outputs`。
- Anima 支持整句 `caption_dropout_rate` 与 text cache 共存。
- `shuffle_caption`、`caption_tag_dropout_rate` 或 token warmup 会改变实际 token，不能与单份固定 text cache 共存。

## 7. Compile、显存与速度优化

### 7.1 Per-block compile 的动态序列支持

```bash
--compile \
--compile_dynamic_sequence \
--compile_dynamic_sequence_min_tokens=256 \
--compile_dynamic_sequence_max_tokens=8192 \
--compile_cache_size_limit=32
```

- 在每个编译的 Anima block 前标记 T/H/W token 轴为 dynamic。
- 让多个 free-fit 宽高比共享编译图，减少因 shape 增加导致的重复编译。
- min/max token 范围可用于提前拒绝异常 shape。
- `--compile` 与 Accelerate 的 `--torch_compile` 互斥。

### 7.2 Activation memory budget

```bash
--compile --activation_memory_budget=0.7
```

- 取值 `[0,1]`。
- 越低表示保存更少 activation，用更多 recompute 换显存。
- `1` 接近最大速度规划。
- 需要当前 PyTorch 暴露对应 functorch 配置；不支持时会显式报错。

### 7.3 QKV / KV projection fusion

```bash
--fuse_qkv_projections
```

- 将冻结底模 self-attention Q/K/V 合并为宽 Linear。
- 将 cross-attention K/V 合并。
- 减少 HBM 读取和 kernel launch。
- 保留原 q/k/v 参数与标准 LoRA key，adapter residual 仍单独计算。

### 7.4 Fused AdamW

```bash
--optimizer_type=AdamW --fused_adamw
```

- 要求 PyTorch AdamW fused implementation。
- 与非 AdamW optimizer 或 `optimizer_args fused=False` 冲突时立即报错，不会静默 fallback。

### 7.5 其他保留优化

- `--qwen_image_vae_2d`：单图 Qwen-Image 2D VAE 路径，适合 latent cache。
- Qwen VAE spatial chunking 使用标准 Conv2d 输出尺寸公式，修复 Free-fit Bucket 长图/高分辨率下 stride-2、无 padding 的确定性尺寸崩溃；chunked 与普通 Conv2d 有数值对照回归测试。
- `--cuda_allow_tf32`。
- `--cuda_cudnn_benchmark`。
- gradient checkpointing、block swap、CPU/Unsloth activation offload。

## 8. REPA + DoG 关系正则

先缓存每张图的 vision-token sidecar：

```bash
python tools/cache_anima_repa_features.py \
  --image_dir=/path/to/images \
  --vision_model=/path/to/vision-encoder \
  --target_pixels=1048576 \
  --resolution_step=16 \
  --max_bucket_reso=2048 \
  --resize_interpolation=lanczos \
  --overwrite
```

训练：

```bash
--repa_weight=0.05 \
--repa_layer=8 \
--repa_feature_suffix=_anima_pe_spatial.safetensors \
--repa_feature_key=image_features \
--repa_dog_sigma_divisor=16 \
--repa_max_tokens=512 \
--repa_anneal_steps=0.3
```

行为：

- 捕获主 DiT forward 指定 block 的 token，不追加第二次 DiT forward。
- 将 DiT token pool 到 vision grid，对齐 token 之间的 Gram/关系矩阵，不需要额外 dimension-matching head。
- DoG 对 target 做高频带通与标准化。
- `repa_no_cls_token` 表示不从 target 移除 CLS token。
- `repa_disable_dog` 可关闭 DoG。
- `repa_anneal_steps` 在 `(0,1]` 时表示总步数比例，大于 1 表示绝对 optimizer step。

约束：

- dataset 必须关闭 flip、color aug 和 random crop。
- `target_pixels`、`resolution_step`、`max_bucket_reso`、`resize_interpolation` 必须与训练一致。
- 新 sidecar 保存 bucket size 和 interpolation enum；训练时不匹配会提前失败。

## 9. Self-Flow 一致性正则

```bash
--self_flow_weight=0.1 --self_flow_delta=0.05
```

- 使用 stop-gradient 的主 prediction 将 `x_sigma` 运输到更低 sigma。
- 第二次 DiT forward 在该位置重建 velocity。
- 大约增加一次 DiT forward，不是“免训练少步蒸馏”。
- 更适合独立研究一致性/平滑性，不是普通角色 LoRA 默认项。

## 10. AnyFlow-style 任意区间蒸馏

```bash
--network_module=networks.flow_map_lora_anima \
--network_dim=32 --network_alpha=32 \
--network_args "interval_hidden_dim=128" \
--anyflow_weight=1.0 \
--anyflow_teacher_steps=4 \
--anyflow_min_interval=0.05
```

- adapter 为每个 block 添加 `(source_t, target_r)` interval embedding。
- 冻结 Anima 底模用 Euler 生成 teacher transition。
- student 学习任意区间的平均 velocity。
- 每 batch 约需 `1 + teacher_steps + 1` 次 DiT forward。
- interval embedder 无法用普通 LoRA key 表达，必须保存/加载完整 adapter。
- 与 DP-DMD 格式互斥，不能同轮开启。

## 11. DP-DMD Turbo

```bash
--network_module=networks.turbo_dmd_anima \
--network_dim=32 --network_alpha=32 \
--dp_dmd_weight=0.1 \
--dp_dmd_critic_weight=1.0 \
--dp_dmd_steps=4
```

- 在同一冻结 Anima 底模上使用 generator LoRA 和 fake-distribution critic LoRA。
- 真实数据 velocity loss 负责锚定多样性。
- DMD 梯度在 rollout 第一步 stop-gradient，后续步可训练。
- 训练成本约为 `dp_dmd_steps + 3` 次 DiT forward / batch。

保存：

- `<name>.safetensors`：generator，可作为普通 LoRA 推理。
- `<name>_dmd_critic.safetensors`：critic sidecar，只在 resume 蒸馏时需要。
- checkpoint retention 和 Hugging Face upload 会跟踪 critic sidecar。

## 12. 新增适配器模块

### 12.1 Soft Tokens

```bash
--network_module=networks.soft_tokens_anima \
--network_args "num_tokens=4" "num_timestep_bins=8" "init_std=0.02"
```

- 为每层、每个 sigma bin 学习 cross-attention conditioning tokens。
- 保存为完整 Soft Token adapter，不是普通 LoRA。

### 12.2 Hydra LoRA

```bash
--network_module=networks.hydra_lora_anima \
--network_args \
  "num_experts=4" \
  "balance_weight=0.01" \
  "orthogonal_weight=0.01" \
  "export_mode=full"
```

- 共享 `lora_down`，使用多个 expert `lora_up`。
- 使用样本路由、load-balance loss 和 expert orthogonality loss。
- `export_mode=full` 保留路由能力。
- `export_mode=standard_mean` 将 expert 均值导出为标准 LoRA，但显式丢失路由。
- 支持 SVD-Down、T-LoRA、AdaLN、dropout 等基础能力。

### 12.3 Chimera LoRA

```bash
--network_module=networks.chimera_lora_anima \
--network_args \
  "num_experts=4" \
  "num_frequency_experts=2" \
  "balance_weight=0.01" \
  "orthogonal_weight=0.01" \
  "export_mode=full"
```

- content router 与 sigma/频率能量 router 同时工作。
- 使用两组低秩 up-expert pool。
- `full` 保留双路由 checkpoint。
- `standard_mean` 将两组 pool 按 rank 拼接后导出，但丢失路由行为，metadata 会显式标记 lossy。

### 12.4 EasyControl

```bash
--network_module=networks.easycontrol_anima \
--network_dim=16 \
--network_args "condition_channels=16"
```

- 用普通 self-attention 和 condition-K/V 扩展 self-attention 的 zero-gated blend。
- 零 gate 保证初始输出与底模一致。
- 要求 ControlNet-style dataset 配置 `conditioning_data_dir`。
- 条件图用同一 Anima VAE 编码。
- 每 block 计算两次 self-attention，attention 成本约翻倍。
- 保存为完整 EasyControl adapter。

## 13. 完整 adapter 推理协议

`anima_minimal_inference.py` 新增：

```bash
--adapter_module=networks.flow_map_lora_anima \
--adapter_weight=/path/to/adapter.safetensors \
--adapter_multiplier=1.0
```

可加载 AnyFlow、Soft Tokens、Hydra、Chimera 和 EasyControl 等完整 adapter。

EasyControl：

```bash
--control_image=/path/to/control.png
```

- prompt 文件和交互模式可以用 `--cn /path/to/control.png` 逐条覆盖。
- 条件图会 resize 到目标输出尺寸并用 Anima VAE 编码。

普通 LoRA 仍使用：

```bash
--lora_weight a.safetensors b.safetensors \
--lora_multiplier 0.8 1.0
```

如果只提供 LoRA 权重而省略 multiplier，会自动为每个权重生成 `1.0`；单个 multiplier 可广播到多个 LoRA，数量不匹配会报错。

自定义 adapter 在每步推理前可获得：

- 当前/next sigma。
- flow interval。
- frequency context。
- EasyControl condition。

步结束后会清理 context，避免泄漏到下一步或下一个 prompt。

## 14. 模型加载、保存与 resume 收紧

### 14.1 独立 LLM Adapter 权重

```bash
--llm_adapter_path=/path/to/llm_adapter.safetensors
```

- 支持 LLM Adapter 嵌入 DiT 文件的官方单文件形式。
- 也支持 DiT 与 LLM Adapter 拆分保存的形式。
- 独立文件 key 会归一化到 `llm_adapter.*`。
- 加载器会根据合并后 state 推断是否启用 LLM Adapter、层数、head 数和维度。

### 14.2 Anima 模型配置自动推断

- 根据 state dict 形状推断 hidden size、block 数、head 和 adapter 配置。
- 支持连续任意 block 数以及编号 safetensors 分片；已覆盖 40-block Anima 2.9B Preview 和最多 38 blocks swap，不引入 28/40 特判。
- 支持 ComfyUI `net.` 前缀。
- 加强 Anima Aesthetic/alternate weight key 归一化。
- 对不完整或形状不一致的权重提前报错，避免后续 AttributeError。

### 14.3 ModelSpec metadata

新增 Anima-only metadata 生成器，记录：

- architecture / implementation。
- title、author、description、license、tags。
- resolution、timestep range。
- implementation commit。
- 可选 thumbnail data URL。
- merge 来源和其他扩展字段。

### 14.4 Checkpoint sidecar 和 resume

- DP-DMD critic sidecar 参与保存、retention、resume 和 Hugging Face upload。
- Hydra/Chimera/Soft Tokens/EasyControl/AnyFlow 可从完整权重重建 network。
- LoKr 可从形状推断 factor。
- LoRA/LoHa/LoKr 恢复 dropout、分模块设定和 LoRA+ 比率。
- validation 与 save 路径统一 unwrap network，避免 DDP/Accelerate wrapper 引发的 key 或类型错误。

## 15. Gradient checkpointing 与 compile 上下文安全

T-LoRA、Chimera、Turbo-DMD 和 EasyControl 的 forward 依赖当前 batch/step 外部 context。分支为每个 adapter 增加 context snapshot/restore，确保 gradient-checkpoint recomputation 不会误用后续 batch 的 sigma、route、mode 或 condition。

允许：

```bash
--compile --gradient_checkpointing
```

对上述 context-dependent adapter，禁止：

```bash
--compile --compile_fullgraph --gradient_checkpointing
```

原因是 restore context 需要 graph break。训练器会在开始时拒绝该组合，不会让它静默产生错误梯度。

## 16. Flow matching 与 loss 收紧

### 16.1 Timestep sampling

支持：

- `sigma`
- `uniform`
- `sigmoid`
- `shift`
- `flux_shift`

关键行为：

- `sigmoid_scale` 作用于 sigmoid 系列分布。
- `discrete_flow_shift` 只在对应 shift-aware 路径生效，日志会明确标记“生效/忽略”。
- `min_timestep/max_timestep` 会把所有 sampling 分布线性映射到指定区间。
- endpoint 会 clamp 到有效 scheduler index，避免罕见越界。
- `--show_timesteps` 使用与真实 Anima 训练相同的 sampler。
- `--show_timesteps_offset` 只对 `sigmoid/shift/flux_shift` 生效，其他分布会显式提示 ignored。

### 16.2 Loss

- L2、L1、Huber、Smooth L1。
- Huber `constant` 与 `exponential` schedule。
- timestep weighting 和 dataset sample weight 在正确的维度上缩减，避免意外生成 `B x B` 外积。
- masked loss 应用在空间缩减之前。
- 全量微调、LoRA 和 ControlNet-LLLite 共用同一 Anima loss 实现。

## 17. 全量微调与 ControlNet-LLLite

### 17.1 全量 DiT 微调

- 继续支持 self-attention、cross-attention、MLP、modulation 和 LLM Adapter 的分组学习率。
- 使用 Anima-only flow scheduler、loss、ModelSpec 和 checkpoint I/O。
- 支持 alpha masked loss。
- 训练、采样和 validation 的 prompt/模型加载路径与 LoRA trainer 对齐。

### 17.2 ControlNet-LLLite

- 保留 Anima ControlNet-LLLite 训练和独立推理。
- 支持 Qwen-Image 2D VAE、独立 LLM Adapter 权重和多 LoRA。
- LLLite 权重 metadata 可推断 `cond_emb_dim`、`mlp_dim`、`target_layers`、`cond_dim`、`cond_resblocks`，也可 CLI 覆盖。
- `--lllite_cond_input=pixel|latent` 隔离 v2 pixel 与 v2.1 Qwen-VAE latent stem；pixel 默认不变，metadata 保存输入空间，错载权重会拒绝。semantic trunk 未引入。
- `tools/dev/run_anima_lllite_cond_ab.py` 用相同 config、数据、seed 和 steps 生成 CUDA A/B；真实速度和效果需在目标 GPU 上验证。
- prompt 文件支持每条 `--cn` 条件图与 `--am` multiplier 覆盖。
- masked loss 优先显式 alpha mask，不再将普通条件图静默当作 mask。

## 18. 互斥和限制总表

| 组合 | 结果 |
| --- | --- |
| `bucket_free_fit` + `bucket_no_upscale` | 禁止 |
| `--compile` + `--torch_compile` | 禁止 |
| `--compile_fullgraph` + `--split_attn` | 禁止 |
| T-LoRA/Chimera/Turbo-DMD/EasyControl + gradient checkpointing + fullgraph | 禁止；普通 `--compile` 可用 |
| `activation_memory_budget` 不配 `--compile` | 禁止 |
| dynamic sequence 不配 `--compile` | 禁止 |
| AnyFlow + DP-DMD | 禁止 |
| REPA + flip/color/random-crop aug | 禁止 |
| text-encoder cache + shuffle/tag-dropout/token-warmup | 禁止 |
| xformers attention 不配 `--split_attn` | 禁止 |
| SageAttention 训练 | 不支持，仅推理 |
| `--fp8_base` Anima 训练 | 不支持 |

## 19. 面向角色/画风 LoRA 的采用顺序

### 角色 LoRA

1. 普通 `networks.lora_anima` 基线。
2. 只加 `down_init=weight_svd`。
3. 只加 T-LoRA mask。
4. 使用 `network_reg_lrs` 将 MLP LR 降低一半。
5. 训练构图粘死时单独尝试 REPA。
6. 复杂细节仍不足时尝试 LoKr。

### 画风 LoRA

1. 普通 `networks.lora_anima` 基线。
2. SVD-Down A/B。
3. AdaLN rank/LR A/B。
4. 内容或构图被画风覆盖时，单独尝试 REPA。
5. 多画风/多概念才研究 Hydra 或 Chimera。

### 不应混入首轮基线

- Self-Flow。
- AnyFlow。
- DP-DMD。
- Soft Tokens。
- Hydra/Chimera/EasyControl。
- LLM Adapter LoRA。
- Channel scaling（在补齐校准 CLI 之前）。

## 20. 验证状态与证据边界

当前分支提供了以下自动化覆盖：

- Anima flow matching 与 loss。
- Free-fit bucket 和 dynamic compile。
- SVD-Down、T-LoRA、AdaLN、channel scaling。
- LoHa/LoKr 恢复、factor 推断与 LoRA+。
- REPA metadata 和插值校验。
- AnyFlow、DP-DMD、Soft Tokens、Hydra、Chimera、EasyControl 的 save/load 和 context。
- masked loss 优先级。
- LoRA multiplier 广播和数量校验。
- 独立 LLM Adapter 权重加载。

但要注意：

- CPU/Mac 上的合成 adapter 单测不等于真实 CUDA Anima 训练质量验证。
- SVD-Down、T-LoRA、AdaLN、REPA 等对角色/画风的收敛和效果仍需要在目标 GPU 上进行同数据、同 seed、同 prompt 的单变量 A/B。
- AnyFlow、DP-DMD、Hydra、Chimera 等应拥有独立的质量、速度、显存和 wall-clock 报告，不应与普通 LoRA 训练结果混为一谈。

## 21. 相关文档

- [Anima LoRA 训练](anima_train_network.md)
- [Anima 高级训练](anima_advanced_training.md)
- [Anima 角色/画风 LoRA 研究](anima_lora_training_latest_research.md)
- [LoHa / LoKr](loha_lokr.md)
- [Free-fit dataset 示例](../examples/anima/dataset_free_fit.toml)
- [`torch.compile`](anima_torch_compile.md)
- [Validation](validation.md)
- [Masked loss](masked_loss_README.md)
- [ControlNet-LLLite](anima_train_control_net_lllite.md)
