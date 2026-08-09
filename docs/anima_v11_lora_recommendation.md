# Anima Aesthetic v1.1 是否适合作为 LoRA / LoHa 训练底模

> 调查日期：2026-08-09  
> 结论适用于本仓库 `remi` 分支的 Anima LoRA、LoHa、LoKr 训练。

## 结论先行

**不建议把 Anima Aesthetic v1.1 当成角色或画风 LoRA 的默认训练底模。默认仍应使用 `anima-base-v1.0.safetensors`。**

这里最容易产生的误解是：官方新发布的不是 `Anima Base v1.1`，而是
`anima-aesthetic-v1.1.safetensors`。截至调查日，官方文件目录中仍只有
`anima-base-v1.0.safetensors` 这一份正式 Base 权重；v1.1 属于 Aesthetic 分支。
[官方文件目录](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/diffusion_models)

这个判断直接遵循作者的模型卡：Base 是未精修、灵活性和多样性最大的版本，且作者明确写明
“LoRAs should be trained using this version”；Aesthetic 则是为了更稳定和更高默认画质而继续微调的版本。
[官方模型卡 Versions](https://huggingface.co/circlestone-labs/Anima#versions)

但是，**Aesthetic v1.1 在技术上可以被本仓库加载并训练 LoRA / LoHa / LoKr**。如果你的 LoRA
只会部署在 Aesthetic v1.1 上，并且目标就是继承它的默认画风，可以把“直接在 v1.1 上训练”作为一个
受控 A/B 实验；不要未经比较就把它取代 Base v1.0。

## 1. v1.1 到底改变了什么

### 已确认事实

- 官方 Hugging Face 在 2026-07-13 上传了 `anima-aesthetic-v1.1.safetensors`；提交只新增这一份
  4.18 GB BF16 权重，没有发布新的 Base v1.1、训练配置或结构配置。
  [官方上传提交 `594c27f`](https://huggingface.co/circlestone-labs/Anima/commit/594c27fea35648b87c86a9b4d5436a6024c820b5)、
  [官方权重文件](https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/diffusion_models/anima-aesthetic-v1.1.safetensors)
- CircleStone Labs 在其官方 Civitai 版本说明中称，v1.1 是一次“部分重新训练”，目的是调整风格、让线条/画面
  更平滑、减少伪影，同时保留细节；作者预计多数用户会更喜欢这一版。这是生成侧的审美改进说明，不是
  LoRA 训练底模推荐。
  [官方 Civitai v1.1 页面](https://civitai.com/models/2458426?modelVersionId=3126581)、
  [官方版本 API](https://civitai.com/api/v1/model-versions/3126581)
- 作者在 Hugging Face discussion #211 中进一步解释：Aesthetic v1.0 是一份较大的 aesthetics full
  finetune，再合并了多个低权重自定义 LoRA；其中包括影响分布、风格与细节的 DMD-inspired LoRA、偏平涂的
  anime-coloring LoRA，以及用于抵消汗滴倾向的负权重 LoRA。v1.1 **没有改变底层的大规模 Aesthetic
  finetune**，主要是改动所合并的 LoRA，使风格更标准、线条更平滑、噪点更少。
  [作者回复：discussion #211](https://huggingface.co/circlestone-labs/Anima/discussions/211)

### 官方没有公开的内容

官方没有给出 v1.1 的数据集明细、训练步数、损失函数、优化器、精确学习率、各个合并 LoRA 的权重，
也没有发布 Base v1.0 与 Aesthetic v1.1 上训练角色/画风 LoRA 的质量对照。因此不能声称 v1.1 在 LoRA
训练上已经被证实优于或劣于 Base；本文的默认选择来自作者明确的底模建议和已公开的权重构成。

## 2. 架构、权重格式和本仓库兼容性

2026-08-09 对官方 safetensors 文件头进行了只读核对：

| 检查项 | Base v1.0 | Aesthetic v1.1 | 结论 |
| --- | --- | --- | --- |
| 文件精度 | BF16 | BF16 | 相同 |
| tensor 数量 | 685 | 685 | 相同 |
| tensor 名称前缀 | `net.` | `model.diffusion_model.` | 仅封装前缀不同 |
| 去除前缀后的名称、shape、dtype | — | — | 全部一致 |
| DiT / LLM Adapter 布局 | 同一套布局 | 同一套布局 | 没有发现结构变化 |

这项核对只能证明**结构兼容**，不能证明两个底模上的训练效果相同。文件证据：
[Base v1.0](https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/diffusion_models/anima-base-v1.0.safetensors)、
[Aesthetic v1.1](https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/diffusion_models/anima-aesthetic-v1.1.safetensors)。

本仓库的加载器会同时移除 `net.` 和 `model.diffusion_model.` 前缀，并从 tensor shape 自动推断 Anima
配置；因此 Aesthetic v1.1 可直接传给 `--pretrained_model_name_or_path`。普通 LoRA、LoHa 和 LoKr 都是
挂载到加载后的模块上，v1.1 没有新增一套网络实现。

相关实现：[`library/anima_utils.py`](../library/anima_utils.py)、
[`networks/lora_anima.py`](../networks/lora_anima.py)、[`docs/loha_lokr.md`](loha_lokr.md)。

需要区分两句话：

1. **可以训练**：结构和当前加载器兼容。
2. **推荐作为默认底模**：官方明确推荐 Base，而不是 Aesthetic；两者不能混为一谈。

## 3. 为什么默认推荐 Base v1.0

### 3.1 它是作者指定的 LoRA 底模

官方把 Base 描述为未精修版本，具有最大灵活性、多样性和风格服从能力，并直接要求 LoRA 使用 Base 训练。
Aesthetic 已经带有高质量数据微调和风格/构图/稳定化 LoRA 合并，不再是中性起点。
[官方模型卡](https://huggingface.co/circlestone-labs/Anima#versions)

### 3.2 角色 LoRA 更需要保留可迁移性

对角色训练而言，通常希望身份能跨服装、姿势、背景和画风保持，同时让使用者在 Base、Aesthetic 或社区
Anima checkpoint 上选择推理底模。Base 上训练避免把 Aesthetic v1.1 的既定平滑线条和分布收缩一起当成
角色的一部分。

这是基于权重构成的工程推断，不是官方角色 benchmark。Base 训练出的 LoRA 能否在 Aesthetic v1.1 上达到
同等身份相似度，仍需固定 seed/prompt 实测；结构相同只说明它可以加载，不保证效果完全可移植。

### 3.3 画风 LoRA 更容易与 Aesthetic 自带风格发生耦合

v1.1 本身就在调整风格、线条、噪点和细节。如果目标画风与它一致，直接训练可能更快得到顺眼结果；如果
目标画风与它冲突，LoRA 必须先抵消底模的审美偏置，再学习目标风格，更容易把“底模风格”和“目标风格”
纠缠起来。官方说明 Base 没有需要克服的 aggressive aesthetic tuning/RLHF，并建议 rank 32 LoRA 从
`2e-5` 的低学习率起步。
[官方 Finetuning Tips](https://huggingface.co/circlestone-labs/Anima#finetuning-tips)

### 3.4 LoHa / LoKr 不是改变底模选择的理由

LoHa、LoKr 容量和参数化方式不同，但它们仍然学习“相对于当前冻结底模的增量”。在 Aesthetic v1.1 上训练
不会自动把它们变成更可移植的适配器。官方没有发布 Anima Aesthetic v1.1 上 LoHa/LoKr 的 A/B 结果；
因此仍建议先在 Base 上做普通 LoRA，确认容量不足后再试 LoHa/LoKr。

## 4. 什么时候值得直接在 Aesthetic v1.1 上训练

只有同时满足下列条件时，我才建议将它加入实验：

- 最终产品明确固定使用 `anima-aesthetic-v1.1.safetensors`，不追求跨 Anima checkpoint 通用。
- 数据目标与 v1.1 的平滑、较干净、较标准的默认风格接近。
- 你愿意同时保留一组 Base v1.0 对照，而不是只看 v1.1 单次结果。
- 评价包含无触发词底模保持、角色身份、换装/换构图、目标风格强度和不同 LoRA 权重，而不只看训练 loss。

不建议直接用 v1.1 的情况：

- 要发布给不同 Anima 底模用户使用的通用角色/画风 LoRA。
- 目标是强烈偏离 v1.1 默认风格的独特画风。
- 数据很少且容易过拟合。
- 希望把训练结果用于分析 Anima Base 本身的能力。
- 没有预算做 Base 与 Aesthetic 的相同数据 A/B。

## 5. 最小 A/B 实验方案

不要同时改变底模、学习率、rank、数据和随机种子。建议先做：

| 组别 | 训练底模 | 网络 | rank / alpha | 学习率 | 用途 |
| --- | --- | --- | --- | --- | --- |
| A | Base v1.0 | 普通 LoRA | 32 / 32 | `2e-5` | 官方推荐基线 |
| B | Aesthetic v1.1 | 普通 LoRA | 32 / 32 | `2e-5` | 只隔离底模差异 |
| C（可选） | Aesthetic v1.1 | 普通 LoRA | 32 / 32 | `1e-5` | 检查已精修底模是否需要更轻触 |

`1e-5` 组是工程实验，不是作者给出的 v1.1 最优值。其余共同设置、数据要求和评估方法见
[`anima_lora_training_latest_research.md`](anima_lora_training_latest_research.md)。继续遵守作者的两个硬建议：

- 冻结 LLM Adapter；本仓库不要传 `train_llm_adapter=True`。
- 使用低学习率，从 rank 32 / `2e-5` 附近向上或向下小扫。

每组使用完全相同的图片、caption、bucket、effective batch、step、seed 和保存间隔。训练后至少交叉验证：

1. A 组 LoRA 分别加载到 Base v1.0 和 Aesthetic v1.1。
2. B/C 组分别加载到其训练底模，并额外尝试 Base v1.0。
3. LoRA 权重测试 `0.6 / 0.8 / 1.0`。
4. 角色用换装、换姿势、换背景和近景/全身 prompt；画风用不同人物、场景、色域和构图。
5. 同时记录目标相似度、画面质量、构图多样性、底模知识损伤和无 trigger 污染。

只有当 B/C 在固定部署场景持续胜过 A，才把 Aesthetic v1.1 设为该项目的训练底模。

## 6. 推理参数与训练参数不要混用

官方模型卡对标准 Anima 推理给出 512²–1536²、30–50 steps、CFG 4–5，并说明 Aesthetic 版本通常不需要
`score_*` quality tags；这些是生成建议，不是 LoRA 训练的 timestep、loss 或学习率配置。
[官方 Generation settings](https://huggingface.co/circlestone-labs/Anima#generation-settings)、
[官方 Aesthetic prompting](https://huggingface.co/circlestone-labs/Anima#aesthetic-version-prompting)

v1.1 没有公开新的训练 scheduler、flow objective 或 LoRA 专用参数。除底模路径外，不应仅因为文件名变成
v1.1 就擅自改变本仓库已经验证过的 Anima flow-matching 训练参数。

## 7. 许可证

截至 2026-08-09，官方仓库使用 CircleStone Labs Non-Commercial License v1.2，并同时受 NVIDIA
Open Model License 对 Derivative Model 的适用条款约束。许可证明确把 LoRA 和 textual inversion 定义为
Derivative：一般只授予非商业使用；个人身份可按 2(c) 的有限例外出售自己拥有或创建的衍生权重，但该例外
不延伸到集成模型的更大产品、工具或功能。生成 Outputs 可以商业使用；分发 LoRA 还需要附带许可证、归属
声明和修改声明。商业部署或组织用途应按实际场景复核全文或取得单独许可。
[官方许可证 v1.2](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md)、
[官方模型卡 License](https://huggingface.co/circlestone-labs/Anima#license)

## 最终建议

- **角色 LoRA：**训练在 Base v1.0；完成后把 Aesthetic v1.1 作为重点推理兼容目标测试。
- **画风 LoRA：**训练在 Base v1.0；只有目标风格明确依赖 v1.1 默认审美且部署固定时，才做 v1.1 直训 A/B。
- **LoHa / LoKr：**同样优先 Base v1.0；先证实普通 LoRA 容量不足，再换算法。
- **不要把 Aesthetic v1.1 误称为 Base v1.1。**它是结构兼容、生成效果更新的 Aesthetic checkpoint，
  不是官方替代 LoRA 训练底模的版本。

