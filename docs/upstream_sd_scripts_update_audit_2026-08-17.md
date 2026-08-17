# sd-scripts 上游更新对 remi 分支的可移植审计（2026-08-17）

## 结论

截至 2026-08-17，上游 kohya-ss/sd-scripts 的 **main** 与 **dev** 都仍指向
[37a1cbbc](https://github.com/kohya-ss/sd-scripts/commit/37a1cbbc5725ed2a3575506e7bd2001c9908ac92)。
这正是 remi 收紧时使用的上游基线，因此目前 **没有新的主线提交可以合并**，也没有新的主线依赖、安全、
safetensors、optimizer、cache 或 torch.compile 修复需要同步。

真正值得取长补短的是尚未合并的 PR。建议优先级如下：

1. **P0：移植 PR #2398 的 Qwen VAE 高分辨率分块卷积尺寸修复。**
2. **P0：不移植 PR #2418 的 28/40-block 特判；remi 已有更通用实现，只补 40-block/2.9B 测试与文档。**
3. **P1：在隔离实验分支移植 PR #2412 的 LLLite latent conditioning，并保留 pixel 为默认值。**
4. **P1：只从 PR #2413 提取 ControlNet target-alpha loss 的可选按样本归一化；semantic trunk v3 暂缓。**
5. **PR #2393 的 ICC、透明图与动画检测意图可取，但当前补丁含恒真条件，不能直接搬。**
6. FAD、KNN noise、CEP、KronA/CDKA 与额外 cache 方向均保持实验或低优先级。
7. PR #2415、#2416 只修改 remi 已删除的 resize_lora.py，不引入。

### 实施状态（remi）

- #2398 的两行输出尺寸公式已移植，并增加 4 组 chunked/普通 `Conv2d` 数值对照。
- 40-block 继续使用 remi 的通用 shape inference；新增分片权重、最大 block swap、真实公开 2.9B safetensors header 与本地完整权重 opt-in smoke。
- target alpha 已贯通 ControlNet TOML → dataset → batch → loss，并以 `--normalize_alpha_mask_loss` 作为默认关闭的显式开关；未引入 semantic trunk。
- LLLite latent conditioning 已与 pixel v2 隔离，pixel 仍为默认；训练、推理、metadata、save/load 与固定条件 CUDA A/B runner 已接通。当前开发机无 CUDA，真实质量和速度 A/B 仍待在目标 GPU、固定数据集上执行。

---

## 1. 审计范围与状态

本报告只使用上游 Git commit、PR、源码、文档和 Actions 检查作为一手来源。

| 对象 | 状态 | 结论 |
|---|---|---|
| 上游 main / dev | [37a1cbbc](https://github.com/kohya-ss/sd-scripts/commit/37a1cbbc5725ed2a3575506e7bd2001c9908ac92) | 相对 remi 基线无新增 |
| remi 审计时 HEAD | [cc968d48](https://github.com/beautifulrem/sd-scripts/commit/cc968d48b2a4283f0e2acc0aff162c8993ac2f0c) | Anima-only，保留 LoRA、全量微调、ControlNet-LLLite 与推理验证 |
| #2398 | open | Qwen VAE 高分辨率分块尺寸修复，P0 |
| #2418 | open | Anima 2.9B 40-block 检测，已被 remi 更通用覆盖 |
| #2412 | open draft | LLLite v2.1 latent conditioning，P1 实验 |
| #2413 | open draft | semantic trunk v3；只拆取 mask normalize |
| #2393 | open | ICC/透明图处理，需重写而非直接移植 |
| #2385/#2374/#2371/#2370 | open | FAD/KNN/CEP/KronA-CDKA，低优先实验 |
| #2415/#2416 | open | 只涉及已删除的 resize_lora.py，无关 |

---

## 2. 高优先级可移植项

### 2.1 P0：PR #2398，修复 Qwen VAE 高分辨率 spatial chunking

来源：[PR #2398](https://github.com/kohya-ss/sd-scripts/pull/2398)。

问题是 Qwen VAE 在对高分辨率图像做 spatial chunking 时，预分配输出尺寸只使用输入尺寸除以 stride，
没有采用 Conv2d 的完整输出尺寸公式。特定高宽和较小 chunk size 会让边界 chunk 进入后续 3×3 卷积时高度过小，
出现“kernel size 大于输入尺寸”的确定性崩溃。PR 作者以 1280×1856 图像复现。

补丁只修改 library/qwen_image_autoencoder_kl.py 中 ChunkedConv2d 的两行尺寸计算：

- 原逻辑：输入高宽整除 stride；
- 新逻辑：使用 input、padding、kernel 和 stride 的标准 Conv2d 输出公式。

#### 为什么适合 remi

- Anima 训练直接使用 Qwen-Image VAE；
- Free-fit bucket 和高分辨率原图会扩大遇到边界尺寸的概率；
- 修复不改变权重格式、训练参数或依赖；
- 修改局部，且不需要恢复任何已删除的跨模型模块。

#### 风险

- PR 没有上游 CI 检查记录，也没有提交自动化回归测试；
- 只修正了输出预分配公式，仍需验证分块拼接索引、最后一个 chunk 和非对称尺寸；
- 不应只用作者的一个 1280×1856 样例证明所有 kernel/stride/padding 组合正确。

#### 建议测试

1. 将 ChunkedConv2d 与同权重普通 Conv2d 对比，覆盖奇偶高宽、超长图、非方图和边界最小 chunk。
2. 覆盖 stride 1/2、padding 0/1、kernel 1/3，以及高或宽不足一个常规 chunk 的情况。
3. 对 1280×1856 和 Free-fit bucket 产生的多组尺寸做真实 VAE encode。
4. 比较 chunked 与非 chunked latent 的 shape、有限值和数值误差。
5. 在 BF16、FP16（如支持）和 FP32 下分别运行，记录峰值 VRAM。

结论：**这是本轮最值得先实现的实际稳定性修复。** 建议单独一个 commit，便于回滚。

### 2.2 P0：PR #2418 的需求已覆盖，只补 2.9B 回归测试和文档

来源：[PR #2418](https://github.com/kohya-ss/sd-scripts/pull/2418) /
[45dddfcc](https://github.com/kohya-ss/sd-scripts/commit/45dddfccb704b6b0591f65d98f0d695c997b3115)。

上游 loader 把 Anima block 数硬编码为 28，PR 通过 safetensors header 检查
blocks.39.mlp.layer1.weight，在 28 和 40 之间二选一，以加载 expanded Anima 2.9B。

remi 已在
[0ed5338](https://github.com/beautifulrem/sd-scripts/commit/0ed533813492320ae2daa385ae77dd1e65634c9f)
实现更通用的 checkpoint shape inference：

- 收集连续 block index，自动得到任意 num_blocks；
- 同时推断 model width、head 数、MLP ratio、patch 和 LLM Adapter 结构；
- 支持 net.、model.diffusion_model. 和无前缀 key；
- 支持 sharded safetensors，而上游 PR 直接 safe_open 单一路径；
- 对缺 key 和不连续 block 显式报错。

因此：

- **不要 cherry-pick #2418；**
- 不要把通用推断降级成 28/40 特判；
- 应补一个不实例化完整 2.9B 的 40-block synthetic shape 测试；
- 再用真实 40-block checkpoint 做 CPU/meta load 与一小步 CUDA LoRA smoke test；
- README 应明确“代码路径支持动态 block 数”，但真实 checkpoint 验证完成前不要写成生产保证。

建议测试矩阵：

| 项目 | 覆盖 |
|---|---|
| block | 28、40、非连续 block 显式失败 |
| key prefix | net.、model.diffusion_model.、无前缀 |
| 文件布局 | 单 safetensors、sharded safetensors |
| 训练 | 40-block LoRA forward/backward/save/load |
| 低显存 | block swap 上限随动态 num_blocks 正确变化 |

---

## 3. 需要适配的实验项

### 3.1 P1：PR #2412，ControlNet-LLLite v2.1 latent conditioning

来源：[PR #2412](https://github.com/kohya-ss/sd-scripts/pull/2412) /
[ec3dd9bc](https://github.com/kohya-ss/sd-scripts/commit/ec3dd9bc28059b55b388e45222feb81295535921)。

该 PR 增加可选的 latent conditioning：

conditioning image → Qwen-Image VAE → 16-channel latent → latent stem → LLLite。

可取之处：

- 默认仍为 pixel，不改变旧命令；
- pixel/latent stem 使用不同 state-dict key，错误混用会显式失败；
- metadata 保存 conditioning input space；
- 4-channel inpainting 通过独立 mask conv pyramid 支持；
- VAE encode 在 no_grad 且禁用 autocast 的区域执行，以规避 FP16 conv NaN；
- 上游 PyTorch 2.4/2.6 测试矩阵与 typos 检查通过。

上游作者在一个角色表情编辑任务上报告：latent 路径大约用十分之一训练步数达到相近效果，
单步从 6.0 s 增至 6.7 s（约 +11.7%）。这是
[#2412 作者的单数据集结果](https://github.com/kohya-ss/sd-scripts/pull/2412)，
不是多数据集、多 seed 或 remi 环境的独立验证。

本地 patch check 表明，#2412 的生产代码和文档相对当前 remi 基本可应用；唯一表面冲突是
tools/dev/manual_test_anima_lllite_dryrun.py 中 repo-root sys.path，而 remi 已提前修复同一问题。

#### 主要限制

- conditioning latent 尚无磁盘 cache，每 step 重新 VAE encode；
- 即使训练 latent 已缓存，VAE 仍需常驻 GPU；
- pixel 与 latent LLLite 权重不互通；
- 表情/inpainting 的收益不能推导到 lineart、depth、pose；
- 当前主要是手工 dry-run，应转成正式 pytest；
- cache key 将来必须包含 VAE 标识、预处理、分辨率和 mask 语义。

#### 建议 A/B

固定数据、seed、LR、batch、模块容量和训练时间，比较：

- 达到同一验证质量阈值的 step 与 wall-clock；
- 稳态 s/it 和峰值 VRAM；
- 身份保持、编辑强度、mask 边界；
- pixel 默认路径固定 seed 回归；
- 3-channel/4-channel save-load round trip；
- metadata 缺失和 pixel/latent 错配必须显式失败；
- BF16/FP16/FP32 VAE encode 无 NaN；
- 多 bucket 下的 shape 与 compile 行为。

建议先在独立实验分支移植，完成真实 CUDA A/B 后再决定是否列为稳定功能。

### 3.2 P1：从 PR #2413 只提取可选 target-alpha loss normalize

来源：draft [PR #2413](https://github.com/kohya-ss/sd-scripts/pull/2413) 中的
[f8e0dd64](https://github.com/kohya-ss/sd-scripts/commit/f8e0dd64ca0976336da9b6cb4e31a4f1e6e662c3)。

remi 当前训练循环会读取 alpha_masks，但标准 ControlNet TOML 路径并不完整：

- ControlNetSubsetParams 和 CN schema 没有 alpha_mask；
- ControlNetSubset 与 ControlNetDataset 委托路径把 alpha_mask 硬编码为 false；
- anima_loss.apply_masked_loss 没有按每个样本的 mask mean 归一化。

上游孤立 commit 做了三件有用的事：

1. 让 ControlNet subset 可以设置 alpha_mask = true；
2. 优先使用 **target/教师图的 alpha 通道**，而不是 conditioning image；
3. 可按样本 mask mean 归一化，避免小编辑区域因画面 framing 改变整体 loss 尺度。

注意：#2413 PR 摘要称“conditioning alpha”，但实际代码和
[上游文档](https://github.com/kohya-ss/sd-scripts/blob/f8e0dd64ca0976336da9b6cb4e31a4f1e6e662c3/docs/anima_train_control_net_lllite.md#L96-L113)
写的是 **target image alpha**。remi 文档必须采用后者。

建议不要强制全局改变 loss 语义，而是增加明确可选项，例如 normalize_alpha_mask_loss，
默认保持兼容；在 ControlNet-LLLite 中拒绝把 conditioning image 当 loss mask 的模糊路径。

测试应覆盖 TOML 传递、alpha 优先级、无 mask identity、不同 mask 面积的归一化、
全零/极小 mask 的 clamp、latent cache 失效，以及 CUDA grad norm。

### 3.3 暂缓：PR #2413 semantic trunk v3

semantic trunk 将 conditioning latent 送入冻结 Anima DiT 的部分 block，用 hidden states、
多 ref-block、zero/uncond/caption context、timestep FiLM 和 scalar/vector gate 支持语义编辑。
它有研究价值，但目前不适合进入 remi 默认能力面：

- 上游明确标为 draft/experimental，gate、multi-block 和 scale 仍在消融；
- 堆叠在 #2412 上，包含十个连续架构提交；
- control_net_lllite_anima.py 增加约千行，checkpoint 组合显著膨胀；
- PR 当前没有完整 pytest check 记录；显示的 typos check 因“PNGs”误判失败；
- 每 step 额外执行到 max(ref_block) 的冻结 DiT forward，默认约半个 DiT forward；
- reference hidden states 尚无 disk cache；
- 多 bucket compile 会多一种 graph/recompile；
- 不支持 inpainting，仅 T=1，并且点对点 gate 不适合跨位置构图变化；
- stem/semantic、single/multi ref-block 和 gate mode 之间权重不兼容；
- 文档引用的 references/anima_lllite_v3_semantic_trunk.md 不在 PR tree 内，部分 AUROC
  结论缺少仓库内完整实验记录。

结论：只可在隔离研究分支评估；应先稳定 #2412，并用 remi 自己的数据运行 ref-block probe。

---

## 4. 意图可取但补丁不可直接移植

### PR #2393：ICC profile、透明图和动画输入

来源：[PR #2393](https://github.com/kohya-ss/sd-scripts/pull/2393)。

意图是正确的：

- 有 ICC profile 时转换为 sRGB；
- indexed/P/CMYK 等模式更稳健地转换；
- 不训练 alpha 时用白底合成透明图，避免直接 RGB 转换留下错误颜色；
- 对 animated GIF/PNG 给出警告；
- 转换失败时保留旧行为并警告。

但当前补丁不能直接搬。核心条件：

image.mode != RGBA **or** image.mode != RGB

对任意单一 mode 都恒为 true，因为一个 mode 不可能同时等于 RGBA 和 RGB。结果是所有图片都会先转 RGBA，
即使本来已经是 RGB。这既是多余转换，也会使控制流难以推断，并表明补丁缺少针对模式矩阵的自动测试。

此外还存在：

- 裸 except；
- import 失败时直接 print，而非项目 logger；
- 没有上游 CI；
- ICC、alpha 和 target-alpha loss 的交互需要明确，不能在 image loader 中悄悄丢失 alpha。

建议在 remi 中按意图重写：

1. 明确区分保留 alpha 与丢弃 alpha 两条路径；
2. 条件改为 mode 不属于 RGB/RGBA 的集合判断；
3. ICC 转换使用窄化异常捕获和 logger；
4. 为 RGB、RGBA、P、PA、CMYK、灰度、带/不带 ICC、animated 输入建立 fixture；
5. target-alpha loss 开启时，必须验证 ICC 转换后 alpha bit-exact 或在容差内保持。

---

## 5. 实验或低优先级方向

### 5.1 FAD，PR #2385

来源：[PR #2385](https://github.com/kohya-ss/sd-scripts/pull/2385)。

Frequency Aware Dropout 试图按 token 频率动态 dropout，并修正 dataset constructor、
schedule、wildcard 统计确定性和 text-encoder cacheability；CI 已通过且测试较多。

但它重写 dataset/caption/cache 多处，而 remi 已对 dataset 与缓存做 Anima-only 收紧。
它还会使文本编码缓存不可用或受限，直接影响当前训练吞吐。没有 Anima 角色/画风 LoRA 的独立 A/B，
因此只应在小数据过拟合明确存在时做隔离实验，不进入默认路径。

### 5.2 KNN noise，PR #2374

来源：[PR #2374](https://github.com/kohya-ss/sd-scripts/pull/2374)。

每个样本生成 K 个噪声候选，再按 latent 与噪声的 L2 距离选最近者；K=1 保持旧行为，CI 通过。
风险是显存和计算随 K 增长，且“latent 与高斯噪声接近”的度量对 flow-matching Anima 是否有益没有
Anima 专项证据。只建议 K=2/4 的小规模 A/B，记录 wall-clock 到质量阈值，不采纳 PR 中未验证的 K=64 建议。

### 5.3 CEP noise，PR #2371

来源：[PR #2371](https://github.com/kohya-ss/sd-scripts/pull/2371)。

CEP 对浮点 condition/text embedding 加 gaussian 或 uniform 扰动，默认关闭，CI 通过。
潜在价值是小数据正则化；风险是 Anima 有 Qwen3、LLM Adapter、T5 IDs/mask 的复合条件，
必须精确限定只扰动真正的浮点 embedding，且与 text-output cache、caption dropout 和
是否训练 LLM Adapter 的语义一致。没有 Anima LoRA 专项结果，故低优先实验。

### 5.4 KronA/CDKA，PR #2370

来源：[PR #2370](https://github.com/kohya-ss/sd-scripts/pull/2370)。

新增两个约 600 行的 adapter 模块，并继续加入 DoRA、ALLoRA、resume 修复等多个概念。
虽然 CI 通过，但 PR 在短时间内多次修改 Kronecker 顺序、alpha 默认值和继续训练行为。
remi 已有 LoKr/LoHa，并修复了 LoKr factor inference 和 resume。没有 Anima-specific
兼容、转换、推理或质量测试前，不应再扩大 adapter 面；先把现有 LoKr 作为对照。

### 5.5 cache 相关

- 已合并的 [PR #2291](https://github.com/kohya-ss/sd-scripts/pull/2291)
  修复 Anima validation 与 text-encoder output cache/caption dropout，已包含在 37a1cbbc 基线，不需重复移植。
- PR #2412 的 conditioning latent cache 尚未实现；这是性能机会，但 cache key 必须覆盖
  VAE、预处理、尺寸、flip/crop/mask 与 dtype，不能先做一个不安全的路径缓存。
- FAD 会使 caption 动态变化，不能与静态 text-encoder cache 直接共存。

---

## 6. 明确不引入

### PR #2415

[PR #2415](https://github.com/kohya-ss/sd-scripts/pull/2415) 防止 resize_lora.py
把模块名里的 down.alpha 错认成 LoRA down weight。修复合理且 CI 通过，但 remi 已删除该跨模型 resize 工具，
当前 Anima loader 也不使用同一 key 匹配逻辑。因此不恢复文件、不移植。

### PR #2416

[PR #2416](https://github.com/kohya-ss/sd-scripts/pull/2416) 只把 resize_lora.py 的
svd_lowrank 默认迭代数从 2 提到 10。该文件不在 remi 中，不应为了四个默认值恢复。

唯一可借鉴之处是：remi 的 down_init=weight_svd 也使用 randomized SVD。
如需评估，应单独比较 niter=2/5/10 的子空间残差、初始化耗时和训练质量，并以显式参数记录，
不能把 #2416 当作直接补丁。

---

## 7. 推荐实施顺序

### 第一批：稳定性

1. 单独移植并测试 #2398；
2. 给动态 Anima loader 增加 40-block/2.9B、分片与 key-prefix 测试；
3. 更新 README/模型文档；
4. 跑全量 pytest、VAE 高分辨率 CUDA encode、28/40-block LoRA smoke。

### 第二批：ControlNet loss

1. 从 #2413 只手工提取 ControlNet alpha 配置链路；
2. 将 normalize 做成明确、兼容默认值的开关；
3. 补 cache invalidation 与 mask 数值测试；
4. 用小编辑区域数据做 CUDA A/B。

### 第三批：LLLite v2.1

1. 隔离分支移植 #2412，pixel 仍为默认；
2. 将 dry-run 断言转为 pytest；
3. 做 pixel 回归和 latent 3ch/4ch round-trip；
4. 多 seed 比较到质量阈值的 wall-clock，而非只看 train loss；
5. 复现后再决定是否标为稳定功能。

### 暂缓

- semantic trunk v3；
- FAD、KNN、CEP、KronA/CDKA；
- conditioning latent disk cache；
- 恢复 resize_lora.py。

---

## 8. 合并前测试矩阵

| 范围 | 必测项 |
|---|---|
| 基础回归 | 全量 pytest、import smoke、git diff --check |
| Qwen VAE | chunked 对普通 Conv2d；奇偶/超长/非方尺寸；BF16/FP16/FP32；真实高分辨率 encode |
| 28-block Base | LoRA forward/backward/save/load；full fine-tune smoke；LLLite pixel smoke |
| 40-block 2.9B | shape inference；单/分片 safetensors；CPU/meta load；CUDA LoRA 一步 |
| Alpha loss | TOML 传递；target alpha 优先；normalize；零/极小 mask；cache 失效；grad norm |
| LLLite v2.1 | pixel 回归；latent 3ch/4ch；mask boundary；metadata mismatch；VAE NaN |
| Compile | 固定尺寸与多 bucket；graph/recompile 数；split attention 组合检查 |
| 性能 | 预热后 s/it、峰值 VRAM、达到固定验证质量阈值的 wall-clock |

CPU/dummy test 只能证明接口与 shape 自洽。2.9B、latent conditioning、Free-fit bucket 和
torch.compile 的生产结论，必须以真实 Anima checkpoint 的 CUDA 训练为准。

---

## 9. 最终建议

当前不需要合并 upstream/main。最稳妥的取长补短路线是：

**#2398 VAE 稳定性修复 → 2.9B 测试/文档 → 可选 target-alpha normalize →
#2412 latent conditioning A/B。**

#2418 的需求已由 remi 更通用地解决；#2393 应按正确意图重写；semantic trunk、FAD、KNN、
CEP、KronA/CDKA 和额外 cache 都应等 Anima 专项证据后再进入稳定分支。
