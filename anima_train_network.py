# Anima LoRA training script

import argparse
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import (
    anima_models,
    anima_flow_matching,
    anima_advanced_training,
    anima_prompt_utils,
    anima_train_utils,
    anima_utils,
    qwen_image_autoencoder_kl,
    strategy_anima,
    strategy_base,
)
import library.anima_args as args_util
import library.compile_utils as compile_utils
import library.anima_model_io as model_io
from library.dataset import DatasetGroup, MinimalDataset
from library import anima_network_trainer
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class AnimaNetworkTrainer(anima_network_trainer.AnimaNetworkTrainerBase):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self._repa_captured = None
        self._advanced_aux_loss = None
        self._train_micro_step = 0

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[DatasetGroup, MinimalDataset],
        val_dataset_group: Optional[DatasetGroup],
    ):
        anima_flow_matching.log_timestep_sampling_info(args)

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning("cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled")
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            assert train_dataset_group.is_text_encoder_output_cacheable(
                cache_supports_dropout=True
            ), "when caching Text Encoder output, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing"

        if args.unsloth_offload_checkpointing:
            if not args.gradient_checkpointing:
                logger.warning("unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled")
                args.gradient_checkpointing = True
            assert (
                not args.cpu_offload_checkpointing
            ), "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing"
            assert (
                args.blocks_to_swap is None or args.blocks_to_swap == 0
            ), "blocks_to_swap is not supported with unsloth_offload_checkpointing"

        if args.compile:
            assert not args.torch_compile, (
                "--compile (per-block torch.compile) and --torch_compile (accelerate dynamo) cannot be used together"
                " / --compile（ブロック単位torch.compile）と--torch_compile（accelerate dynamo）は併用できません"
            )
            assert not (args.compile_fullgraph and args.split_attn), (
                "--compile_fullgraph cannot be used with --split_attn (split attention uses dynamic control flow)"
                " / --compile_fullgraphは--split_attnと併用できません（split attentionは動的な制御フローを使用します）"
            )

        if args.activation_memory_budget is not None:
            assert args.compile, "--activation_memory_budget requires --compile"
            assert 0.0 <= args.activation_memory_budget <= 1.0, "--activation_memory_budget must be in [0, 1]"
        if args.compile_dynamic_sequence:
            assert args.compile, "--compile_dynamic_sequence requires --compile"
            if args.compile_dynamic_sequence_min_tokens is not None:
                assert args.compile_dynamic_sequence_min_tokens > 0
            if args.compile_dynamic_sequence_max_tokens is not None:
                assert args.compile_dynamic_sequence_max_tokens > 0
            if args.compile_dynamic_sequence_min_tokens is not None and args.compile_dynamic_sequence_max_tokens is not None:
                assert args.compile_dynamic_sequence_min_tokens <= args.compile_dynamic_sequence_max_tokens

        if args.compile and args.compile_fullgraph and args.gradient_checkpointing:
            network_args = {}
            for network_arg in args.network_args or []:
                key, value = network_arg.split("=", 1)
                network_args[key] = value
            uses_external_checkpoint_context = args.network_module in {
                "networks.chimera_lora_anima",
                "networks.turbo_dmd_anima",
                "networks.easycontrol_anima",
            } or str(network_args.get("use_timestep_mask", "false")).lower() in {"1", "true", "yes", "on"}
            if uses_external_checkpoint_context:
                raise ValueError(
                    "--compile_fullgraph cannot be combined with gradient checkpointing for T-LoRA, Chimera, "
                    "Turbo-DMD, or EasyControl because their checkpoint recomputation context requires a graph break; "
                    "omit --compile_fullgraph (ordinary --compile remains supported)"
                )

        assert args.repa_weight >= 0.0
        assert args.self_flow_weight >= 0.0
        assert 0.0 < args.self_flow_delta <= 1.0
        assert args.dp_dmd_weight >= 0.0
        assert args.dp_dmd_critic_weight >= 0.0
        assert args.dp_dmd_steps >= 2
        if args.dp_dmd_weight > 0:
            assert args.network_module == "networks.turbo_dmd_anima", (
                "--dp_dmd_weight requires --network_module=networks.turbo_dmd_anima"
            )
        assert args.anyflow_weight >= 0.0
        assert args.anyflow_teacher_steps >= 1
        assert 0.0 < args.anyflow_min_interval < 1.0
        if args.anyflow_weight > 0:
            assert args.network_module == "networks.flow_map_lora_anima", (
                "--anyflow_weight requires --network_module=networks.flow_map_lora_anima"
            )
            assert args.dp_dmd_weight == 0, "AnyFlow and DP-DMD use different adapter formats and cannot share one run"
        if args.repa_weight > 0:
            assert train_dataset_group.is_repa_feature_compatible(), (
                "REPA sidecars require flip_aug, color_aug and random_crop to be disabled"
            )
            assert args.repa_layer >= 0
            assert args.repa_dog_sigma_divisor > 0
            assert args.repa_max_tokens > 0
            train_dataset_group.enable_repa_features(args.repa_feature_suffix, args.repa_feature_key)

        train_dataset_group.verify_bucket_reso_steps(16)  # WanVAE spatial downscale = 8 and patch size = 2
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(16)

    def load_target_model(self, args, weight_dtype, accelerator):
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0

        # Load Qwen3 text encoder (tokenizers already loaded in get_tokenize_strategy)
        logger.info("Loading Qwen3 text encoder...")
        qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
        qwen3_text_encoder.eval()

        # Load VAE
        logger.info("Loading Anima VAE...")
        vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)
        vae.to(weight_dtype)
        vae.eval()

        # Return format: (model_type, text_encoders, vae, unet)
        return "anima", [qwen3_text_encoder], vae, None  # unet loaded lazily

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, list[nn.Module]]:
        loading_dtype = weight_dtype
        loading_device = "cpu" if self.is_swapping_blocks else accelerator.device

        attn_mode = "torch"
        if args.xformers:
            attn_mode = "xformers"
        if args.attn_mode is not None:
            attn_mode = args.attn_mode

        # Load DiT
        logger.info(f"Loading Anima DiT model with attn_mode={attn_mode}, split_attn: {args.split_attn}...")
        model = anima_utils.load_anima_model(
            accelerator.device,
            args.pretrained_model_name_or_path,
            attn_mode,
            args.split_attn,
            loading_device,
            loading_dtype,
            False,
        )

        # Store unsloth preference so that the Anima trainer base can
        # dit.enable_gradient_checkpointing(cpu_offload=...), we can override to use unsloth.
        # The base trainer only passes cpu_offload, so we store the flag on the model.
        self._use_unsloth_offload_checkpointing = args.unsloth_offload_checkpointing

        # Block swap
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        if self.is_swapping_blocks:
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        return model, text_encoders

    def get_tokenize_strategy(self, args):
        # Load tokenizers from paths (called before load_target_model, so self.qwen3_tokenizer isn't set yet)
        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.qwen3,
            t5_tokenizer_path=args.t5_tokenizer_path,
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
        )
        return tokenize_strategy

    def get_tokenizers(self, tokenize_strategy: strategy_anima.AnimaTokenizeStrategy):
        return [tokenize_strategy.qwen3_tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_anima.AnimaLatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check)

    def get_text_encoding_strategy(self, args):
        return strategy_anima.AnimaTextEncodingStrategy()

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        if args.repa_weight <= 0:
            return
        if args.repa_layer >= len(unet.blocks):
            raise ValueError(f"repa_layer={args.repa_layer} but Anima has only {len(unet.blocks)} blocks")

        def capture_repa(_module, _inputs, output):
            self._repa_captured = output

        unet.blocks[args.repa_layer].register_forward_hook(capture_repa)
        accelerator.print(
            f"Relational REPA enabled: block={args.repa_layer}, weight={args.repa_weight}, "
            f"DoG={not args.repa_disable_dog}, max_tokens={args.repa_max_tokens}"
        )

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            return None  # no text encoders needed for encoding
        return text_encoders

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, False
            )
        return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # We cannot move DiT to CPU because of block swap, so only move VAE
                logger.info("move vae to cpu to save memory")
                org_vae_device = vae.device
                vae.to("cpu")
                clean_memory_on_device(accelerator.device)

            logger.info("move text encoder to gpu")
            text_encoders[0].to(accelerator.device)

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # cache sample prompts
            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}")

                tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
                text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

                prompts = anima_prompt_utils.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"  cache TE outputs for: {p}")
                                tokens_and_masks = tokenize_strategy.tokenize(p)
                                sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy, text_encoders, tokens_and_masks
                                )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            # move text encoder back to cpu
            logger.info("move text encoder back to cpu")
            text_encoders[0].to("cpu")

            if not args.lowram:
                logger.info("move vae back to original device")
                vae.to(org_vae_device)

            clean_memory_on_device(accelerator.device)
        else:
            # move text encoder to device for encoding during training/validation
            text_encoders[0].to(accelerator.device)

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]  # compatibility
        te = self.get_models_for_text_encoding(args, accelerator, text_encoders)
        qwen3_te = te[0] if te is not None else None

        text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        dit = accelerator.unwrap_model(unet)
        adapter_ref = getattr(dit, "_anima_adapter_network", None)
        adapter = adapter_ref() if adapter_ref is not None else None
        on_prompt_start = on_prompt_end = None
        if adapter is not None and hasattr(adapter, "set_condition_latents"):

            def on_prompt_start(prompt_dict, callback_accelerator):
                import os

                import numpy as np
                from PIL import Image

                from library.utils import IMAGE_TRANSFORMS

                condition_path = prompt_dict.get("controlnet_image")
                if condition_path is None or not os.path.isfile(condition_path):
                    logger.warning(
                        "EasyControl sample has no valid control image; add '--cn <path>' to the sample prompt"
                    )
                    adapter.clear_condition_latents()
                    return
                width = max(64, int(prompt_dict.get("width", 512)) // 16 * 16)
                height = max(64, int(prompt_dict.get("height", 512)) // 16 * 16)
                with Image.open(condition_path) as image:
                    image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                    pixels = IMAGE_TRANSFORMS(np.asarray(image).copy()).unsqueeze(0)
                original_vae_device = vae.device
                vae.to(callback_accelerator.device)
                try:
                    with torch.no_grad():
                        condition_latents = self.encode_images_to_latents(
                            args,
                            vae,
                            pixels.to(callback_accelerator.device, dtype=vae.dtype),
                        )
                finally:
                    vae.to(original_vae_device)
                    clean_memory_on_device(callback_accelerator.device)
                adapter.set_condition_latents(condition_latents)

            def on_prompt_end(_prompt_dict):
                adapter.clear_condition_latents()

        anima_train_utils.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            unet,
            vae,
            qwen3_te,
            tokenize_strategy,
            text_encoding_strategy,
            self.sample_prompts_te_outputs,
            on_prompt_start=on_prompt_start,
            on_prompt_end=on_prompt_end,
        )

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = anima_flow_matching.AnimaFlowMatchScheduler(
            num_train_timesteps=1000, shift=args.discrete_flow_shift
        )
        return noise_scheduler

    def encode_images_to_latents(self, args, vae, images):
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage
        return vae.encode_pixels_to_latents(images)  # Keep 4D for input/output

    def shift_scale_latents(self, args, latents):
        # Latents already normalized by vae.encode with scale
        return latents

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        anima: anima_models.Anima = unet

        # Sample noise
        if latents.ndim == 5:  # Fallback for 5D latents (old cache)
            latents = latents.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]
        noise = torch.randn_like(latents)

        # Per-sample timestep sampling offset from custom_attributes: timestep_sampling = { offset = ... }
        # Only applied during training; validation uses unbiased sampling for comparable loss.
        tso = None
        if is_train and "custom_attributes" in batch:
            offsets = [ca.get("timestep_sampling", {}).get("offset", 0.0) for ca in batch["custom_attributes"]]
            t = torch.tensor(offsets, dtype=torch.float32)
            if t.abs().sum() > 0:
                tso = t
        noisy_model_input, timesteps, sigmas = anima_flow_matching.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype,
            timestep_sampling_offset=tso,
        )
        timesteps = timesteps / 1000.0  # scale to [0, 1] range. timesteps is float32

        # Gradient checkpointing support
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)

        # Unpack text encoder conditions
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_conds[
            :4
        ]  # ignore caption_dropout_rate which is not needed for training step

        # Move to device
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        attn_mask = attn_mask.to(accelerator.device)
        t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
        t5_attn_mask = t5_attn_mask.to(accelerator.device)

        # Create padding mask
        bs = latents.shape[0]
        h_latent = latents.shape[-2]
        w_latent = latents.shape[-1]
        padding_mask = torch.zeros(bs, 1, h_latent, w_latent, dtype=weight_dtype, device=accelerator.device)

        # Call model
        noisy_model_input = noisy_model_input.unsqueeze(2)  # 4D to 5D, [B, C, H, W] -> [B, C, 1, H, W]
        self._advanced_aux_loss = None
        self._repa_captured = None
        unwrapped_network = None
        if network is not None:
            unwrap_model = getattr(accelerator, "unwrap_model", None)
            unwrapped_network = unwrap_model(network) if unwrap_model is not None else network
        if is_train and unwrapped_network is not None and hasattr(unwrapped_network, "set_timestep_mask"):
            unwrapped_network.set_timestep_mask(sigmas.flatten())
        if unwrapped_network is not None and hasattr(unwrapped_network, "set_step_sigmas"):
            unwrapped_network.set_step_sigmas(sigmas.flatten())
        if unwrapped_network is not None and hasattr(unwrapped_network, "set_frequency_context"):
            unwrapped_network.set_frequency_context(sigmas.flatten(), noisy_model_input)
        if unwrapped_network is not None and hasattr(unwrapped_network, "set_adapter_mode"):
            unwrapped_network.set_adapter_mode("generator")
        try:
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                model_pred = anima(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    padding_mask=padding_mask,
                    target_input_ids=t5_input_ids,
                    target_attention_mask=t5_attn_mask,
                    source_attention_mask=attn_mask,
                )

                auxiliary_terms = []
                if is_train and unwrapped_network is not None and hasattr(unwrapped_network, "get_auxiliary_loss"):
                    network_aux = unwrapped_network.get_auxiliary_loss()
                    if network_aux is not None:
                        auxiliary_terms.append(network_aux)
                repa_weight = float(getattr(args, "repa_weight", 0.0) or 0.0)
                if is_train and repa_weight > 0 and self._repa_captured is not None:
                    repa_features = batch.get("repa_features")
                    if repa_features is None:
                        raise RuntimeError("REPA is enabled but this batch has no repa_features")
                    cutoff = float(getattr(args, "repa_anneal_steps", 0.0) or 0.0)
                    if 0 < cutoff <= 1:
                        cutoff *= args.max_train_steps
                    optimizer_step = self._train_micro_step // max(1, args.gradient_accumulation_steps)
                    if cutoff <= 0 or optimizer_step < cutoff:
                        repa_loss = anima_advanced_training.relational_repa_loss(
                            self._repa_captured,
                            repa_features.to(accelerator.device),
                            (latents.shape[-2], latents.shape[-1]),
                            patch_size=anima.patch_spatial,
                            has_cls_token=not args.repa_no_cls_token,
                            use_dog=not args.repa_disable_dog,
                            dog_sigma_divisor=args.repa_dog_sigma_divisor,
                            max_tokens=args.repa_max_tokens,
                        )
                        auxiliary_terms.append(repa_weight * repa_loss)

                self_flow_weight = float(getattr(args, "self_flow_weight", 0.0) or 0.0)
                if is_train and self_flow_weight > 0:
                    transported, next_sigmas = anima_advanced_training.transported_self_flow_input(
                        noisy_model_input, model_pred, sigmas.flatten(), args.self_flow_delta
                    )
                    if unwrapped_network is not None and hasattr(unwrapped_network, "set_timestep_mask"):
                        unwrapped_network.set_timestep_mask(next_sigmas)
                    self_flow_pred = anima(
                        transported,
                        next_sigmas,
                        prompt_embeds,
                        padding_mask=padding_mask,
                        target_input_ids=t5_input_ids,
                        target_attention_mask=t5_attn_mask,
                        source_attention_mask=attn_mask,
                    )
                    self_flow_loss = torch.nn.functional.mse_loss(self_flow_pred.float(), model_pred.detach().float())
                    auxiliary_terms.append(self_flow_weight * self_flow_loss)

                anyflow_weight = float(getattr(args, "anyflow_weight", 0.0) or 0.0)
                if is_train and anyflow_weight > 0:
                    if unwrapped_network is None or not hasattr(unwrapped_network, "set_flow_interval"):
                        raise RuntimeError("AnyFlow distillation requires FlowMapLoRANetwork")
                    source_t = sigmas.flatten().detach()
                    max_interval = source_t.clamp_min(args.anyflow_min_interval)
                    interval = args.anyflow_min_interval + torch.rand_like(source_t) * (
                        max_interval - args.anyflow_min_interval
                    ).clamp_min(0)
                    target_r = (source_t - interval).clamp_min(0.0)

                    # Frozen base Anima supplies an ODE transition target over
                    # the arbitrary [t,r] interval.
                    unwrapped_network.set_enabled(False)
                    unwrapped_network.clear_flow_interval()
                    teacher_state = noisy_model_input.detach()
                    with torch.no_grad():
                        for teacher_index in range(args.anyflow_teacher_steps):
                            fraction = teacher_index / args.anyflow_teacher_steps
                            next_fraction = (teacher_index + 1) / args.anyflow_teacher_steps
                            teacher_t = source_t + (target_r - source_t) * fraction
                            teacher_next = source_t + (target_r - source_t) * next_fraction
                            teacher_velocity = anima(
                                teacher_state,
                                teacher_t,
                                prompt_embeds,
                                padding_mask=padding_mask,
                                target_input_ids=t5_input_ids,
                                target_attention_mask=t5_attn_mask,
                                source_attention_mask=attn_mask,
                            )
                            dt = (teacher_next - teacher_t).view(-1, 1, 1, 1, 1).to(teacher_state)
                            teacher_state = teacher_state + teacher_velocity * dt
                    mean_velocity_target = (teacher_state - noisy_model_input.detach()) / (
                        (target_r - source_t).view(-1, 1, 1, 1, 1).to(noisy_model_input).clamp_max(-1e-6)
                    )

                    unwrapped_network.set_enabled(True)
                    unwrapped_network.set_flow_interval(source_t, target_r)
                    flow_map_velocity = anima(
                        noisy_model_input,
                        source_t,
                        prompt_embeds,
                        padding_mask=padding_mask,
                        target_input_ids=t5_input_ids,
                        target_attention_mask=t5_attn_mask,
                        source_attention_mask=attn_mask,
                    )
                    unwrapped_network.clear_flow_interval()
                    per_sample_flow_loss = (flow_map_velocity.float() - mean_velocity_target.float()).square().mean(
                        dim=tuple(range(1, flow_map_velocity.ndim))
                    )
                    valid_intervals = ((source_t - target_r) > 1e-6).to(per_sample_flow_loss)
                    # A zero-length [t,r] interval has no velocity target.
                    anyflow_loss = (per_sample_flow_loss * valid_intervals).sum() / valid_intervals.sum().clamp_min(1)
                    auxiliary_terms.append(anyflow_weight * anyflow_loss)

                dp_dmd_weight = float(getattr(args, "dp_dmd_weight", 0.0) or 0.0)
                if is_train and dp_dmd_weight > 0:
                    if unwrapped_network is None or not hasattr(unwrapped_network, "set_adapter_mode"):
                        raise RuntimeError("DP-DMD requires the dual-branch TurboDMD network")
                    step_grid = torch.linspace(
                        1.0, 0.0, int(args.dp_dmd_steps) + 1, device=noise.device, dtype=torch.float32
                    )
                    generated = noise.unsqueeze(2)
                    for step_index in range(args.dp_dmd_steps):
                        step_sigma = step_grid[step_index].expand(bs)
                        if hasattr(unwrapped_network, "set_timestep_mask"):
                            unwrapped_network.set_timestep_mask(step_sigma)
                        generated_velocity = anima(
                            generated,
                            step_sigma,
                            prompt_embeds,
                            padding_mask=padding_mask,
                            target_input_ids=t5_input_ids,
                            target_attention_mask=t5_attn_mask,
                            source_attention_mask=attn_mask,
                        )
                        generated = generated + generated_velocity * (step_grid[step_index + 1] - step_grid[step_index])
                        if step_index == 0:
                            # DP-DMD role separation: no DMD gradient reaches
                            # the diversity-anchor first step.
                            generated = generated.detach()

                    critic_sigmas = sigmas.flatten().detach()
                    critic_eps = torch.randn_like(generated)
                    sigma_view = critic_sigmas.view(-1, 1, 1, 1, 1).to(generated)
                    fake_noisy = (1 - sigma_view) * generated + sigma_view * critic_eps
                    if hasattr(unwrapped_network, "set_timestep_mask"):
                        unwrapped_network.set_timestep_mask(critic_sigmas)
                    unwrapped_network.set_adapter_mode("critic")
                    fake_velocity = anima(
                        fake_noisy.detach(),
                        critic_sigmas,
                        prompt_embeds,
                        padding_mask=padding_mask,
                        target_input_ids=t5_input_ids,
                        target_attention_mask=t5_attn_mask,
                        source_attention_mask=attn_mask,
                    )
                    unwrapped_network.set_adapter_mode("base")
                    with torch.no_grad():
                        teacher_velocity = anima(
                            fake_noisy.detach(),
                            critic_sigmas,
                            prompt_embeds,
                            padding_mask=padding_mask,
                            target_input_ids=t5_input_ids,
                            target_attention_mask=t5_attn_mask,
                            source_attention_mask=attn_mask,
                        )
                    unwrapped_network.set_adapter_mode("generator")

                    critic_target = critic_eps - generated.detach()
                    critic_loss = torch.nn.functional.mse_loss(fake_velocity.float(), critic_target.float())
                    dmd_direction = fake_velocity.detach().float() - teacher_velocity.detach().float()
                    norm_dims = tuple(range(1, dmd_direction.ndim))
                    dmd_direction = dmd_direction / dmd_direction.abs().mean(dim=norm_dims, keepdim=True).clamp_min(1e-6)
                    dmd_loss = (generated.float() * dmd_direction).mean()
                    auxiliary_terms.append(dp_dmd_weight * dmd_loss)
                    auxiliary_terms.append(float(args.dp_dmd_critic_weight) * critic_loss)

                if auxiliary_terms:
                    self._advanced_aux_loss = torch.stack(auxiliary_terms).sum()
                if is_train:
                    self._train_micro_step += 1
        finally:
            # T-LoRA is training-only. Clearing here guarantees ordinary full-rank
            # validation, sampling, saving, and inference behavior.
            if unwrapped_network is not None and hasattr(unwrapped_network, "clear_timestep_mask"):
                unwrapped_network.clear_timestep_mask()
            if unwrapped_network is not None and hasattr(unwrapped_network, "clear_step_sigmas"):
                unwrapped_network.clear_step_sigmas()
            if unwrapped_network is not None and hasattr(unwrapped_network, "clear_frequency_context"):
                unwrapped_network.clear_frequency_context()
            if unwrapped_network is not None and hasattr(unwrapped_network, "set_adapter_mode"):
                unwrapped_network.set_adapter_mode("generator")
            if unwrapped_network is not None and hasattr(unwrapped_network, "clear_flow_interval"):
                unwrapped_network.clear_flow_interval()
                unwrapped_network.set_enabled(True)
        model_pred = model_pred.squeeze(2)  # 5D to 4D, [B, C, 1, H, W] -> [B, C, H, W]

        # Rectified flow target: noise - latents
        target = noise - latents

        # Loss weighting
        weighting = anima_train_utils.compute_loss_weighting_for_anima(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

        return model_pred, target, timesteps, weighting

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy,
        tokenize_strategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
    ) -> torch.Tensor:
        """Override base process_batch for caption dropout with cached text encoder outputs."""

        unwrapped_network = None
        if network is not None:
            unwrap_model = getattr(accelerator, "unwrap_model", None)
            unwrapped_network = unwrap_model(network) if unwrap_model is not None else network
        if unwrapped_network is not None and hasattr(unwrapped_network, "set_condition_latents"):
            conditioning_images = batch.get("conditioning_images")
            if conditioning_images is None:
                raise RuntimeError("EasyControl requires a ControlNet dataset with conditioning_images")
            with torch.no_grad():
                condition_latents = self.encode_images_to_latents(
                    args, vae, conditioning_images.to(accelerator.device, dtype=vae_dtype)
                )
            unwrapped_network.set_condition_latents(condition_latents)

        # Text encoder conditions
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        anima_text_encoding_strategy: strategy_anima.AnimaTextEncodingStrategy = text_encoding_strategy
        if text_encoder_outputs_list is not None:
            caption_dropout_rates = text_encoder_outputs_list[-1]
            text_encoder_outputs_list = text_encoder_outputs_list[:-1]

            # Apply caption dropout to cached outputs
            text_encoder_outputs_list = anima_text_encoding_strategy.drop_cached_text_encoder_outputs(
                *text_encoder_outputs_list, caption_dropout_rates=caption_dropout_rates
            )
            # Add the caption dropout rates back to the list for validation dataset (which is re-used batch items)
            batch["text_encoder_outputs_list"] = text_encoder_outputs_list + [caption_dropout_rates]

        try:
            return super().process_batch(
                batch,
                text_encoders,
                unet,
                network,
                vae,
                noise_scheduler,
                vae_dtype,
                weight_dtype,
                accelerator,
                args,
                text_encoding_strategy,
                tokenize_strategy,
                is_train,
                train_text_encoder,
                train_unet,
            )
        finally:
            if unwrapped_network is not None and hasattr(unwrapped_network, "clear_condition_latents"):
                unwrapped_network.clear_condition_latents()

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        auxiliary = self._advanced_aux_loss
        self._advanced_aux_loss = None
        return loss if auxiliary is None else loss + auxiliary

    def get_sai_model_spec(self, args):
        return model_io.get_anima_model_spec_dataclass(args, lora=True).to_metadata_dict()

    def update_metadata(self, metadata, args):
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # Set first parameter's requires_grad to True to workaround Accelerate gradient checkpointing bug
        first_param = next(text_encoder.parameters())
        first_param.requires_grad_(True)

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        # The Anima trainer base only calls enable_gradient_checkpointing(cpu_offload=True/False),
        # so we re-apply with unsloth_offload if needed (after base has already enabled it).
        if self._use_unsloth_offload_checkpointing and args.gradient_checkpointing:
            unet.enable_gradient_checkpointing(unsloth_offload=True)

        if not self.is_swapping_blocks:
            model = super().prepare_unet_with_accelerator(args, accelerator, unet)
        else:
            model = unet
            model = accelerator.prepare(model, device_placement=[not self.is_swapping_blocks])
            accelerator.unwrap_model(model).move_to_device_except_swap_blocks(accelerator.device)
            accelerator.unwrap_model(model).prepare_block_swap_before_forward()

        # CUDA perf switches are independent of torch.compile; apply whenever requested.
        compile_utils.apply_cuda_optimizations(args)

        if args.fuse_qkv_projections:
            dit = accelerator.unwrap_model(model)
            enabled, skipped = anima_models.enable_attention_projection_fusion(dit)
            logger.info(f"Enabled fused Anima attention projections for {enabled} modules")
            if skipped:
                logger.warning(
                    f"Skipped projection fusion for {skipped} attention modules with unsupported adapter monkey-patches"
                )

        if args.compile:
            # Apply per-block torch.compile to the DiT blocks. Reach the real Anima via
            # unwrap_model so we mutate the underlying ModuleList regardless of any DDP wrapper.
            dit = accelerator.unwrap_model(model)
            compile_utils.compile_transformer(args, dit, [dit.blocks], disable_linear=self.is_swapping_blocks)

        return model

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            # prepare for next forward: because backward pass is not called, we need to prepare it here
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()


def setup_parser() -> argparse.ArgumentParser:
    parser = anima_network_trainer.setup_parser()
    args_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    # parser.add_argument("--fp8_scaled", action="store_true", help="Use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). "
        "Cannot be used with --cpu_offload_checkpointing or --blocks_to_swap.",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    if args.show_timesteps:
        anima_train_utils.show_timesteps(args)
    else:
        trainer = AnimaNetworkTrainer()
        trainer.train(args)
