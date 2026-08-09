# Anima 角色与画风 LoRA / LoHa / LoKr 训练研究（2026-08-09）

## 结论先行

目前最有证据支撑的起点不是照搬 SDXL，而是：

- 底模固定为官方 `anima-base-v1.0.safetensors`，不要用 Aesthetic、Turbo 或旧 Preview 训练。官方明确写明 LoRA 应在 Base 上训练；Base 具有最大的灵活性、多样性和风格服从能力。[Anima 官方模型卡](https://huggingface.co/circlestone-labs/Anima)
- 第一轮一律从普通 LoRA 开始：`dim=32`、`alpha=32`、AdamW、`lr=2e-5`、BF16、DiT-only、LLM Adapter 与 Qwen3 均冻结。rank 32、`2e-5` 和不要训练 LLM Adapter 是模型作者给出的起点；`alpha=32` 是本仓库中使 `alpha/dim=1` 的工程选择，不是作者单独发布的 alpha 最优结论。[官方模型卡](https://huggingface.co/circlestone-labs/Anima)、[作者公开画风配置](https://gist.github.com/tdrussell/3f79596efb8e27672da0881afd9c1d51)
- 角色与画风都先用 `timestep_sampling=sigmoid`、`weighting_scheme=uniform`、L2、dropout=0。不要把网上常见的 `discrete_flow_shift=3` 与 `sigmoid` 同时使用并期待它生效；本仓库明确说明该 shift 在 sigmoid 下会被忽略。[sd-scripts Anima 文档](https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md)
- LoHa/LoKr 已能在本仓库训练，但上游仍称其为 experimental；没有 Anima 官方 LoRA/LoHa/LoKr 质量 A/B。它们应是第二轮实验，不是默认答案。[sd-scripts LoHa/LoKr 文档](https://github.com/kohya-ss/sd-scripts/blob/main/docs/loha_lokr.md)
- 不存在诚实的“固定最佳 epoch/step”。应每 200–250 step 或每 2–5 epoch 保存，并用固定 seed/prompt 网格及验证集选择“刚学会目标且仍保留姿势、服装、构图和底模知识”的最早 checkpoint。

## 证据等级

- **A（已证实）**：Anima 官方模型卡、模型作者公开配置、上游源码/文档、论文原文。
- **B（强工程推断）**：由官方配置映射到本仓库 CLI，或由架构/代码行为直接推出，但没有 Anima 专项 A/B。
- **C（经验起点）**：社区报告或跨模型经验，只能用于小规模参数扫描。

下文所有“推荐”均附等级；没有标成 A 的数字，不应被描述成官方最优。

## 1. 底模与组件怎么选

| 项目 | 设置 | 等级 | 依据 |
| --- | --- | --- | --- |
| DiT | `anima-base-v1.0.safetensors` | A | 官方明确说 LoRA 应使用 Base；Aesthetic 是高质量默认风格微调，Turbo 是少步蒸馏，都会降低训练自由度。[模型卡](https://huggingface.co/circlestone-labs/Anima) |
| 文本编码器 | `qwen_3_06b_base.safetensors` | A | 官方组件。[文件目录](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/text_encoders) |
| VAE | `qwen_image_vae.safetensors` | A | 官方 Qwen-Image VAE。[文件目录](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/vae) |
| LLM Adapter | 加载但冻结 | A | 作者称其位于文本 embedding 进入 DiT 之前，影响过大且容易被小数据破坏。[模型卡 Finetuning Tips](https://huggingface.co/circlestone-labs/Anima#finetuning-tips) |
| 精度 | BF16 | A/B | 作者配置使用 BF16；上游为 Anima 做过 FP16 数值稳定修复，但 BF16 仍是参考路径。[作者配置](https://gist.github.com/tdrussell/3f79596efb8e27672da0881afd9c1d51)、[上游稳定性修复](https://github.com/kohya-ss/sd-scripts/commit/fa53f71) |
| FP8/量化底模 | 不用 | A | `--fp8_base` 对 Anima 不受支持；社区也有误用 FP8 checkpoint 导致异常训练的实例。[上游文档](https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md)、[排障讨论](https://huggingface.co/circlestone-labs/Anima/discussions/35) |

官方 ComfyUI 单文件 Base 已把约 269 MB 的 text conditioner/LLM Adapter 打包进 DiT 文件，这是根据单文件与 Diffusers 拆分文件布局得出的推断；通常不必另传 `--llm_adapter_path`。只有使用自定义拆分 checkpoint 时才传。[单文件目录](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/diffusion_models)、[Diffusers 拆分目录](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers/tree/main)

Anima 并不是 SDXL U-Net：其 Diffusers 组件是 Qwen3 + 六层 Anima text conditioner + Cosmos Transformer DiT + Qwen-Image VAE + FlowMatch scheduler；DiT 为 28 层、hidden 2048、16 heads、16-channel latent。作者说明 LLM Adapter 把 Qwen3 embedding 映射到 T5XXL embedding space，每轮 diffusion 前只运行一次。这解释了为什么应使用 flow-matching 训练参数，也解释了为什么小数据不应随意训练 LLM Adapter。[Diffusers PR](https://github.com/huggingface/diffusers/pull/13732)、[组件索引](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers/blob/main/modular_model_index.json)、[Transformer 配置](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers/blob/main/transformer/config.json)

许可提醒：Anima 模型及其衍生模型受 CircleStone Labs Non-Commercial License 限制；模型卡说明生成输出本身不受该非商用限制。发布 LoRA 前应按实际用途复核许可证。[许可证说明](https://huggingface.co/circlestone-labs/Anima#license)

## 2. 适配器类型怎么选

| 目标 | 首选 | 何时换 | 建议起点 | 证据边界 |
| --- | --- | --- | --- | --- |
| 单角色、身份准确且保持姿势自由 | 普通 LoRA | rank 32 明显学不住复杂身份细节时再试 LoKr；若角色“粘死”训练构图则试 LoHa | LoRA `32/32 @ 2e-5` | 普通 LoRA 有作者起点；角色专属最优未公开 |
| 简单角色、少图、强调泛化 | 普通 LoRA 或 T-LoRA A/B | 普通 LoRA 很快过拟合时试 T-LoRA | LoRA `32/32`；T-LoRA 保持 `32/32`，只加 `min_rank=4` mask | T-LoRA 论文在 SDXL/FLUX 验证，不是 Anima 官方结论 |
| 单一画风 | 普通 LoRA | 风格把人物/构图一起粘死时换 LoHa | LoRA `32/32 @ 2e-5` | 模型作者公开画风 LoRA 使用 rank 32 / 2e-5 |
| 多概念或希望更强泛化 | LoHa 实验 | 学习不足再升 dim，不先升 LR | `dim=16 alpha=8 @ 1e-5~2e-5` | LyCORIS 跨架构建议，非 Anima 官方最优 |
| 极小文件 | LoKr `factor=-1` | 细节不足改 `factor=4/8`；但文件和过拟合风险增加 | `dim=16~32 alpha=8~16` | LyCORIS 官方算法建议；没有 Anima A/B |
| 极高细节拟合 | LoKr `factor=4/8` 或更高容量 LoRA | 只有普通 LoRA 的数据/训练量排查完仍不足时 | LR 从 `1e-5` 小扫 | 高容量 LoKr 更易迁移差/过拟合，必须验证 |

LyCORIS 官方说明：LoHa 的有效更新秩上限可达到 `dim²`，具有更强 dampening，可能适合简单概念、多概念和泛化；建议 dim≤32、alpha 从 1 到 dim/2，高 dim 可能 loss 不稳或 NaN。LoKr 使用 Kronecker 分解；`factor=-1` 是平衡小文件，较小 factor 会增大容量/文件，但小 LoKr 在不同底模间可能更难迁移。[LyCORIS Algo List](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-List.md)、[LoHa 的 FedPara 论文](https://arxiv.org/abs/2108.06098)、[LyCORIS 论文](https://arxiv.org/abs/2309.14859)

## 3. 普通 LoRA 的共同基线

下面是本仓库可直接使用的保守命令。`max_train_steps=1500` 是第一轮的检查上限，不是保证最优的目标点；应从更早 checkpoint 开始选择。

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
  --gradient_accumulation_steps=4 \
  --max_grad_norm=1.0 \
  --timestep_sampling=sigmoid --sigmoid_scale=1.0 \
  --weighting_scheme=uniform --loss_type=l2 \
  --max_train_steps=1500 --save_every_n_steps=250 \
  --mixed_precision=bf16 --gradient_checkpointing \
  --cache_latents --cache_latents_to_disk \
  --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk \
  --qwen_image_vae_2d \
  --seed=42
```

证据对应：rank 32 / `2e-5` / 不训 LLM Adapter 来自作者；AdamW、betas、weight decay、BF16、effective batch 4、warmup 100 和 gradient clip 来自作者公开画风配置。该公开配置没有显式列出 scheduler；这里选 `constant_with_warmup` 是为了让 100-step warmup 真正生效的工程映射。多分辨率设置来自作者数据配置。[作者训练配置](https://gist.github.com/tdrussell/3f79596efb8e27672da0881afd9c1d51)、[作者数据配置](https://gist.github.com/tdrussell/d5651bcf4a3f0855c7f55f5148519d73)

本仓库中，冻结 LLM Adapter 的正确方法是**不要**传 `train_llm_adapter=True`；`llm_adapter_lr=0` 是 diffusion-pipe 的参数，不是本仓库普通 LoRA 的控制项。`--network_train_unet_only` 冻结 Qwen3；缓存文本输出也会阻止 Qwen3 LoRA 训练。[本仓库 Anima 文档](https://github.com/beautifulrem/sd-scripts/blob/remi/docs/anima_train_network.md)

## 4. 角色 LoRA：数据和参数

### 数据集

推荐第一轮 30–80 张高质量图（C，范围不是官方定值），宁可减少近重复图，也不要靠重复次数制造“更多数据”。身份学习最依赖覆盖：

- 近景、半身、全身均有；正面、3/4、侧面、俯仰角均有。
- 多姿势、多表情、多背景、多光照；如果希望服装可换，训练集中必须有服装变化。
- 身份不可变特征在 caption 中保持一致；姿势、镜头、服装、背景等可变因素逐图准确描述。
- 每张 caption 都带唯一触发词，例如 `zxremichar`；不要使用底模已有角色名作为新角色 trigger。
- Anima 官方支持 Danbooru tags、自然语言和混合 caption。已有 tags 应小写并以空格代替下划线（`score_*` 除外）；自然语言描述角色时，角色名后补基本外观。[模型卡 Prompting](https://huggingface.co/circlestone-labs/Anima#prompting)

示例 caption：

```text
zxremichar, 1girl, long silver hair, amber eyes, black jacket, cowboy shot, looking at viewer, city street, night
```

若所有图都只有头像，参数无法凭空补出全身/侧面身份；这类失败首先是数据覆盖问题，不应先暴力提高 LR。[角色泛化失败实例](https://huggingface.co/circlestone-labs/Anima/discussions/119)

### 角色参数建议

| 参数 | 第一轮 | 扫描范围 | 等级/理由 |
| --- | --- | --- | --- |
| LoRA dim/alpha | `32/32` | `16/16`, `32/32`, 复杂多角色可试 `64/64` | A/B；作者 rank32 起点，alpha=rank 为本仓库映射 |
| LR | `2e-5` | `1e-5`, `2e-5`, `5e-5` | A 起点；高于 `2e-5` 仅在明确欠拟合时试 |
| effective batch | 4 | 2–8 | A/C；作者画风配置 batch4，角色最优未公开 |
| steps | 保存 250/500/750/1000/1250/1500 | 不预设“必须 3000” | B；以最早通过验证者为准 |
| dropout | 0 | `rank_dropout=0.05` 或 `module_dropout=0.05` | A/C；作者基线无 dropout；过拟合先早停/降 LR |
| 分辨率 | 1024 面积桶 | 512+1024 多数据集 A/B | B；官方支持 512²–1536²，角色通常无需先上 1536 |
| timestep | sigmoid 1.0 | T-LoRA A/B | A/B；没有角色专属官方分布 |

角色最重要的评价不是训练 loss，而是：固定 trigger 后替换服装、姿势、背景、表情、镜头能否仍保持身份；去掉 trigger 后底模是否恢复正常。

## 5. 画风 LoRA：数据和参数

### 数据集

- 目标画风应覆盖不同人物、物体、场景、构图、色域和明暗；否则 LoRA 会同时记住题材和构图。
- caption 描述图中内容，但不要反复把希望由 trigger 承担的风格属性写成普通内容标签；所有图使用唯一 trigger，例如 `zxremistyle`。
- 如果训练已有艺术家标签，Anima 推理语法要求 `@artist`；自定义 LoRA trigger 不必冒充官方 artist tag。
- 作者的公开风格 LoRA 使用多分辨率 `[512,1024,1536]`，并说明 `[512,1024]` 通常可靠、1536 对细节型画风可有帮助但不是必需。[作者 dataset config](https://gist.github.com/tdrussell/d5651bcf4a3f0855c7f55f5148519d73)

### 画风参数建议

| 参数 | 第一轮 | 第二轮 A/B | 等级/理由 |
| --- | --- | --- | --- |
| LoRA | `32/32 @ 2e-5` | 简单风格可试 `16/16` | A/B；作者公开风格配置是 rank32 / 2e-5 |
| batch | effective 4 | 8（同时重新评估 LR） | A/B |
| 分辨率 | 512+1024 | 追加 1536 细节层 | A；作者公开配置 |
| sigmoid scale | 1.0 | 1.3 | A/B；作者配置把 1.3 作为可选细节向设置 |
| steps/epoch | 每 2–5 epoch 保存和评估 | 过拟合前早停 | A/B；作者配置有周期 eval，而不是只看最终 epoch |
| dropout | 0 | 0.05 | C；风格粘死先改数据、早停、降 LR，再用 dropout |

作者公开风格 LoRA 的完整证据链：[模型/API](https://civitai.com/api/v1/models/2536147)、[训练 config](https://gist.github.com/tdrussell/3f79596efb8e27672da0881afd9c1d51)、[dataset config](https://gist.github.com/tdrussell/d5651bcf4a3f0855c7f55f5148519d73)、[eval config](https://gist.github.com/tdrussell/6a1eea3aa0634d3d02c37462fefe5b9f)。该例最终取 40 epoch，但因每图训练于三种面积，不能把“40 epoch”脱离数据量、batch 和多分辨率直接复制到其他数据集。

## 6. Caption shuffle、dropout 与文本缓存的开关

本仓库有一个必须注意的互斥关系：`--cache_text_encoder_outputs` 不能与 `shuffle_caption`、`caption_tag_dropout_rate` 或 token warmup 同时使用；但 Anima 支持缓存文本输出时的整句 `caption_dropout_rate`。[实现检查](https://github.com/beautifulrem/sd-scripts/blob/remi/anima_train_network.py)

因此有两种合法路线：

1. **速度优先基线**：固定、准确 caption；开启文本缓存；`caption_dropout_rate=0`。这是上面命令采用的路线。
2. **caption 增广实验**：仍用 `--network_train_unet_only` 冻结 Qwen3，但关闭文本输出缓存，数据 TOML 开 `shuffle_caption=true`、`keep_tokens=1`、`caption_tag_dropout_rate=0.05~0.1`。trigger 放第一个并由 `keep_tokens` 固定。这更接近底模随机 tag dropout 的训练方式，但会增加运行显存/时间，且不是已证实优于固定 caption。

固定 caption 的基础数据 TOML：

```toml
[[datasets]]
resolution = [1024, 1024]
batch_size = 1
enable_bucket = true
bucket_free_fit = true
bucket_reso_steps = 16
min_bucket_reso = 512
max_bucket_reso = 1536

  [[datasets.subsets]]
  image_dir = "/data/images"
  caption_extension = ".txt"
  num_repeats = 1
  shuffle_caption = false
  flip_aug = false
  color_aug = false
  random_crop = false
```

`bucket_free_fit` 是本仓库 Anima-only 的面积保持桶；它保留原始纵横比，并与 `bucket_no_upscale` 互斥。[数据配置文档](https://github.com/beautifulrem/sd-scripts/blob/remi/docs/config_README-en.md)、[示例](https://github.com/beautifulrem/sd-scripts/blob/remi/examples/anima/dataset_free_fit.toml)

## 7. LoHa 和 LoKr 的可执行起点

以下数字是 **B/C 工程起点，不是 Anima 官方最优**。

### LoHa：泛化/多概念优先

把共同命令中的网络部分替换为：

```bash
--network_module=networks.loha \
--network_dim=16 --network_alpha=8 \
--learning_rate=1e-5
```

如果明确欠拟合，再试：

```bash
--network_dim=32 --network_alpha=16 --learning_rate=1e-5
```

不要用高 dim 配高 LR 起跑；LyCORIS 官方明确警告高 dim LoHa 可能出现不稳定或 NaN。[LyCORIS Algo List](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-List.md)

### LoKr：小文件或高细节实验

小文件起点：

```bash
--network_module=networks.lokr \
--network_dim=16 --network_alpha=8 \
--network_args "factor=-1" \
--learning_rate=1e-5
```

细节仍不足时只改变一个变量，试 `factor=8`，再试 `factor=4`；factor 越小通常容量/文件越大。不要一开始同时升 LR、dim 和降低 factor，否则无法判断收益来自哪里。[本仓库 LoKr 参数说明](https://github.com/beautifulrem/sd-scripts/blob/remi/docs/loha_lokr.md)、[LyCORIS Guidelines](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Guidelines.md)

LoHa/LoKr 也支持 `rank_dropout`、`module_dropout`、正则表达式分层 rank/LR 和 LoRA+，但本仓库只称 LoRA+ 经过基本测试；没有 Anima 效果 A/B，因此不要在第一轮叠加。[本仓库 LoHa/LoKr 文档](https://github.com/beautifulrem/sd-scripts/blob/remi/docs/loha_lokr.md)

## 8. Anima 专属和实验开关

| 开关 | 默认建议 | 何时开启 | 证据边界 |
| --- | --- | --- | --- |
| `train_llm_adapter=True` | **关** | 只有大数据、全新语义概念、且有严格验证时 | 官方明确建议不训练 |
| Qwen3 LoRA | **关** | trigger/语言绑定问题经 DiT-only 排除后 | 无官方角色/画风收益证据，且不能缓存 TE |
| `loraplus_lr_ratio=2` | 关 | 普通 LoRA 收敛很慢时单独 A/B | LoRA+ 论文报告同算力更快，但不是 Anima 实证。[LoRA+](https://arxiv.org/abs/2402.12354) |
| `down_init=weight_svd` | 关 | 第二轮普通 LoRA A/B | 本仓库实现，保持标准 LoRA 格式；无官方 Anima 最优证据 |
| `use_timestep_mask=true` | 关 | 单图/极少图易过拟合 | T-LoRA 论文证明高 timestep 更易过拟合，但只在 SDXL/FLUX 验证。[T-LoRA](https://arxiv.org/abs/2507.05964) |
| `train_adaln=true` | 关 | 画风/全局调制不足的严格 A/B | Anima 架构独有潜力，但高影响、无作者配方 |
| REPA | 关 | 有可靠视觉编码器 sidecar，想加关系保持正则 | 会限制 flip/color/random crop；非普通 LoRA 必需项 |
| Self-Flow | 关 | 研究一致性正则 | 约双倍 DiT forward，不会自动得到少步模型 |

本仓库这些开关的实际语义、保存格式和互斥关系见 [Anima advanced training](https://github.com/beautifulrem/sd-scripts/blob/remi/docs/anima_advanced_training.md)。建议一次只打开一个，并保留普通 LoRA 对照组。

一个合理的 T-LoRA A/B：

```bash
--network_module=networks.lora_anima \
--network_dim=32 --network_alpha=32 \
--network_args "use_timestep_mask=true" "min_rank=4" "alpha_rank_scale=1.0" \
--learning_rate=2e-5
```

这组与普通 LoRA 基线相比只改变 timestep rank mask，不同时改 dim 或加 SVD-Down，因此结果才可归因。

## 9. 性能和显存开关

按风险从低到高启用：

1. `--cache_latents --cache_latents_to_disk`。
2. DiT-only 时 `--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk`。
3. `--qwen_image_vae_2d`：上游单图测试称 latent 数值等价，encode/decode 约 2 倍快，峰值 VRAM 约为 3D VAE 的 1/3。[上游提交](https://github.com/kohya-ss/sd-scripts/commit/575da53)
4. BF16 + gradient checkpointing。
5. 有 Triton 时试 `--compile --compile_mode=default --compile_cache_size_limit=32 --cuda_allow_tf32 --cuda_cudnn_benchmark`。多 bucket 第一轮会编译多个图，第二轮后才容易看到收益；不要与 `--torch_compile` 同开。[本仓库 compile 文档](https://github.com/beautifulrem/sd-scripts/blob/remi/docs/anima_torch_compile.md)、[上游 per-block compile 提交](https://github.com/kohya-ss/sd-scripts/commit/ac67c12)
6. 仍 OOM 时，二选一：`--blocks_to_swap=N` 或 `--unsloth_offload_checkpointing`；两者互斥，都会牺牲速度。

普通 AdamW 是作者参考路径。LoRA 参数本身不大，显存允许时先用 AdamW；只有优化器状态确实构成瓶颈时再换 AdamW8bit，并保持其他参数不变做对照。

## 10. Timestep、loss 与 SDXL 遗留参数

- 第一轮：`timestep_sampling=sigmoid`、`sigmoid_scale=1.0`、`weighting_scheme=uniform`、`loss_type=l2`。
- 画风纹理不足可单独 A/B `sigmoid_scale=1.3`；这是作者公开画风配置中的可选项，不代表所有角色都更好。
- `discrete_flow_shift` 仅在对应 sampling 分支生效；`sigmoid + discrete_flow_shift=3` 中的 3 会被忽略。可先运行 `--show_timesteps=console --show_timesteps_resolution=1024` 检查实际分布。[本仓库文档](https://github.com/beautifulrem/sd-scripts/blob/remi/docs/anima_train_network.md)
- Diffusers scheduler 的 `shift=3` 是推理 scheduler 配置，不能据此推导训练时 sigmoid 也要设 3。[官方 scheduler 配置](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers/blob/main/scheduler/scheduler_config.json)
- 不要添加 SD1/SDXL 遗留项：`v2`、`v_parameterization`、`clip_skip` 对 Anima 不适用；`noise_offset`、debiased loss、Huber、min/max timestep 也没有官方角色/画风最佳证据，先保持关闭。

## 11. 如何挑选 checkpoint

每个 checkpoint 用同一组 seed、分辨率、prompt 和 LoRA 强度生成网格：

- Base 推理基线：30–50 steps、CFG 4–5；官方常用 `er_sde`，也列出 `euler_a`、`dpmpp_2m_sde_gpu`、`euler`。[模型卡 Generation settings](https://huggingface.co/circlestone-labs/Anima#generation-settings)
- LoRA strength 至少测 `0.6 / 0.8 / 1.0`。
- 角色网格：训练内外服装、全身/头像、正/侧/背、简单/复杂背景、单人/多人、不同表情。
- 画风网格：人物、动物、建筑、室内、风景、明/暗色域、线稿/厚涂，以及底模原有多个 `@artist` 标签。
- 必测负对照：不写 trigger；若仍强烈出现目标角色/风格，说明过拟合或 caption 绑定失败。
- 数据足够时留 5–10% 验证集，用 validation loss 辅助，但最终仍以生成网格的身份/风格、可编辑性和底模保持三项共同决定。[本仓库 validation 文档](https://github.com/beautifulrem/sd-scripts/blob/remi/docs/validation.md)

应优先选择最早满足目标的 checkpoint，而不是 loss 最低或训练最久的 checkpoint。

## 12. 最小实验矩阵

不要一次扫几十组。建议先做以下 4 组：

| 组 | 适配器 | dim/alpha | LR | 其他变化 |
| --- | --- | --- | --- | --- |
| A | LoRA | 32/32 | 2e-5 | 官方基线映射 |
| B | LoRA | 32/32 | 1e-5 | 判断是否过拟合/风格粘死来自 LR |
| C | LoRA | 16/16 | 2e-5 | 判断容量是否过剩 |
| D | LoHa | 16/8 | 1e-5 | 判断更强 dampening 是否改善泛化 |

只有 A–D 都无法学习细节时，才加 LoKr `factor=8`；只有普通 LoRA 很快过拟合时，才加 T-LoRA。每组使用相同数据划分、seed、保存点和评价网格。

## 13. 最新工程生态与来源热度

截至 2026-08-09 的 GitHub API 快照：`kohya-ss/sd-scripts` 约 7.2k stars、`LyCORIS` 约 2.5k、模型作者使用的 `diffusion-pipe` 约 2.0k；这些是当前最值得优先相信的实现来源，而不是“某个配置被转发很多次”。[sd-scripts API](https://api.github.com/repos/kohya-ss/sd-scripts)、[LyCORIS API](https://api.github.com/repos/KohakuBlueleaf/LyCORIS)、[diffusion-pipe API](https://api.github.com/repos/tdrussell/diffusion-pipe)

重要的 Anima 上游节点：

- 2026-02：加入 Anima LoHa/LoKr。[commit 2217704](https://github.com/kohya-ss/sd-scripts/commit/2217704)
- 2026-02：修复 Anima rank dropout 维度问题。[commit 609d129](https://github.com/kohya-ss/sd-scripts/commit/609d129)
- 2026-04：加强 Anima FP16 数值稳定性。[commit fa53f71](https://github.com/kohya-ss/sd-scripts/commit/fa53f71)
- 2026-06：加入 image-only 2D Qwen VAE 与 per-block compile。[commit 575da53](https://github.com/kohya-ss/sd-scripts/commit/575da53)、[commit ac67c12](https://github.com/kohya-ss/sd-scripts/commit/ac67c12)

社区专用训练器可用于观察工程实践，但不能替代作者证据：`sorryhyun/anima_lora` 提供 Anima 专用 compile/T-LoRA/实验堆栈；Citron Colab 提供低显存入门，默认 768/dim20 可在约 6 GB VRAM 运行，但其参数不是质量最优证明。[anima_lora](https://github.com/sorryhyun/anima_lora)、[Citron trainer](https://github.com/citronlegacy/citron-colab-anima-lora-trainer)

## 14. 明确没有找到的结论

截至本次检索，没有一手来源能证明：

- Anima v1.0 角色 LoRA 的固定最佳图数、epoch 或 steps；
- Anima 官方 LoHa 配方或 LoRA/LoHa/LoKr 质量对照；
- 角色和画风各自唯一最佳 timestep、loss weighting、Huber 或 min/max timestep；
- 训练 Qwen3 或 LLM Adapter 对小数据角色/画风有普遍正收益；
- Aesthetic/Turbo 作为训练底模比 Base 更好。

因此，最稳妥的生产顺序是：**Base v1.0 普通 LoRA 基线 → 低 LR/容量 A/B → LoHa 泛化 A/B → 仅在明确细节不足时试 LoKr → 最后才叠加 T-LoRA/AdaLN/REPA 等 Anima 专属实验项。**
