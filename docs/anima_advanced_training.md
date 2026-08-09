# Anima-specific optimization and experimental training

All features on this page are opt-in. Start with a one-epoch ordinary LoRA smoke test, then enable one method at a time. These objectives use Anima's Rectified Flow and `B,T,H,W,D` DiT layout; they are not SDXL aliases.

## Conventional LoRA improvements

Use `--network_module=networks.lora_anima` and pass these through `--network_args`:

| Option | Purpose | Checkpoint compatibility |
| --- | --- | --- |
| `down_init=weight_svd` | Initialize `lora_down` from the base Linear's leading right-singular subspace; `lora_up` remains zero. | Standard LoRA |
| `use_timestep_mask=true min_rank=4 alpha_rank_scale=1.0` | T-LoRA: fewer rank channels at high noise, full rank near clean data. Validation and inference use full rank. | Standard LoRA |
| `train_adaln=true adaln_rank=16 adaln_alpha=16 adaln_lr=2e-5` | Include Anima AdaLN modulation with independent rank, alpha and LR. | Standard LoRA |
| `channel_scaling_alpha=0.5 channel_scaling_stats=/path/stats.safetensors` | SmoothQuant-style input-channel gradient rebalance. Stats map each LoRA module name to its calibrated `mean(abs(x))` vector. | Baked to standard LoRA when saved |

`--fused_adamw` requests PyTorch fused AdamW and rejects incompatible optimizer types instead of silently falling back.

## Compile and resolution path

The free-fit dataset example is [`examples/anima/dataset_free_fit.toml`](../examples/anima/dataset_free_fit.toml).

```bash
--compile \
--compile_dynamic_sequence \
--compile_dynamic_sequence_min_tokens=256 \
--compile_dynamic_sequence_max_tokens=8192 \
--activation_memory_budget=0.7 \
--fuse_qkv_projections
```

- `bucket_free_fit=true` targets the configured pixel area while preserving each source aspect ratio and allowing upscale. It is mutually exclusive with `bucket_no_upscale`.
- Dynamic-sequence compilation marks Anima's T/H/W token axes before every compiled block and enforces optional total-token bounds.
- A lower activation-memory budget recomputes more activations and may reduce VRAM at additional compute cost.
- Projection fusion retains the original q/k/v parameters and LoRA keys. Frozen base self-QKV and cross-KV are fused; adapter residuals remain separate.

## Relational REPA and DoG

Cache one vision-token sidecar per image first:

```bash
python tools/cache_anima_repa_features.py \
  --image_dir=/path/to/images \
  --vision_model=/path/to/vision-encoder \
  --target_pixels=1048576 \
  --resolution_step=16 \
  --max_bucket_reso=2048 \
  --overwrite
```

Then add:

```bash
--repa_weight=0.05 \
--repa_layer=8 \
--repa_feature_suffix=_anima_pe_spatial.safetensors \
--repa_dog_sigma_divisor=16 \
--repa_max_tokens=512 \
--repa_anneal_steps=0.3
```

The three geometry values must match the training dataset's resolution area, `bucket_reso_steps`, and `max_bucket_reso`. If the dataset sets `resize_interpolation`, pass the same value as `--resize_interpolation` to the cache tool. New sidecars record their selected bucket and training fails early on a mismatch; regenerate sidecars created by an older version of the tool. REPA captures the selected block from the primary DiT forward, pools it to the cached vision grid and matches pairwise token relations. It needs no second DiT forward. DoG band-passes and standardizes the vision target before Gram alignment. Flip, color and random-crop augmentation are rejected because they invalidate spatial sidecars. Values in `(0,1]` for `repa_anneal_steps` are fractions of total steps.

## Self-Flow

```bash
--self_flow_weight=0.1 --self_flow_delta=0.05
```

The main prediction transports `x_t` to a lower sigma with stop-gradient; a second DiT forward must reproduce the velocity. This approximately doubles DiT work per batch. It is a consistency regularizer, not a distilled few-step checkpoint by itself.

## AnyFlow-style arbitrary-interval distillation

```bash
--network_module=networks.flow_map_lora_anima \
--network_dim=32 --network_alpha=32 \
--anyflow_weight=1.0 \
--anyflow_teacher_steps=4 \
--anyflow_min_interval=0.05
```

The adapter adds a learned `(source_t,target_r)` interval embedding to every block. Frozen base Anima performs the teacher Euler transition; the student predicts its mean interval velocity in one call. Training costs about `1 + teacher_steps + 1` DiT forwards. The full checkpoint is required because the interval embedder cannot be represented by ordinary LoRA keys. The retained sampler detects an applied flow-map network and supplies each current/next sigma pair, so inference step count remains selectable.

Standalone inference for a full adapter uses:

```bash
python anima_minimal_inference.py \
  ... \
  --adapter_module=networks.flow_map_lora_anima \
  --adapter_weight=/path/to/flow_map.safetensors
```

The same loader supports full Soft Token, Hydra, Chimera, and EasyControl checkpoints and supplies their runtime context. Ordinary or lossy standard LoRA exports should continue to use `--lora_weight`. For EasyControl, add `--control_image /path/to/control.png`; batch and interactive prompt lines may override it with `--cn /path/to/control.png`. The condition image is resized to the requested output size and encoded with the same Anima VAE before denoising.

## DP-DMD Turbo

```bash
--network_module=networks.turbo_dmd_anima \
--network_dim=32 --network_alpha=32 \
--dp_dmd_weight=0.1 \
--dp_dmd_critic_weight=1.0 \
--dp_dmd_steps=4
```

This uses generator and fake-distribution critic LoRA branches over one frozen Anima base. The real-data velocity objective anchors diversity; DMD gradients are stopped at the first rollout step and refine later steps. Saving writes:

- `<name>.safetensors`: ordinary generator LoRA;
- `<name>_dmd_critic.safetensors`: critic sidecar required only to resume distillation.

`--dim_from_weights --network_weights=<name>.safetensors` restores both branches. Checkpoint retention and Hugging Face upload include the critic sidecar automatically. For ordinary image generation, load only the primary generator file through `networks.lora_anima`/`--lora_weight`.

Expect roughly `dp_dmd_steps + 3` total DiT forwards per batch (`dp_dmd_steps + 2` beyond the ordinary data forward). AnyFlow and DP-DMD use different adapter formats and cannot share one run.

## Advanced adapter modules

| Network module | Mechanism | Saved format |
| --- | --- | --- |
| `networks.soft_tokens_anima` | Per-layer, per-sigma-bin cross-attention tokens. Args: `num_tokens`, `num_timestep_bins`. | Full Soft Token adapter |
| `networks.hydra_lora_anima` | Shared down projection, routed up experts, load-balance and expert-orthogonality losses. | `export_mode=full`; `standard_mean` is explicitly lossy |
| `networks.chimera_lora_anima` | Additive content router plus sigma/frequency-energy router with two low-rank pools. | Full dual-pool checkpoint; `standard_mean` rank-concatenates both pools but loses routing |
| `networks.easycontrol_anima` | Zero-gated blend of ordinary and condition-K/V extended self-attention. | Full EasyControl adapter |

EasyControl requires a ControlNet-style dataset containing `conditioning_data_dir`; conditions are encoded with the same Anima VAE. Training-time sample prompt lines should include `--cn /path/to/control.png`; otherwise the sampler warns and renders the base DiT. It evaluates self-attention twice per block, so attention work is approximately doubled. Its zero gate makes the initial output identical to the base model.

All context-dependent adapters snapshot their active sigma, route, mode, or condition for gradient-checkpoint recomputation. They are therefore safe with ordinary, CPU-offloaded, and Unsloth-style Anima block checkpointing instead of silently recomputing with a later batch context. T-LoRA, Chimera, Turbo-DMD, and EasyControl require the default graph-break-capable compile mode for this restoration; `--compile_fullgraph` plus gradient checkpointing is rejected for those combinations.

## Suggested adoption order

1. Ordinary LoRA + SVD-Down + fused AdamW; verify a standard checkpoint round trip.
2. Add AdaLN and T-LoRA; compare fixed validation prompts.
3. Add free-fit/dynamic compile and tune the activation budget on the target GPU.
4. Try channel scaling or REPA independently before combining them.
5. Treat Self-Flow, AnyFlow, DP-DMD and routed adapters as separate experiments with their own baseline and wall-clock report.
