import gc
import importlib
import argparse
import math
import os
import typing
from typing import Literal, Optional
import sys
import random
import time
import json
from multiprocessing import Value
import numpy as np

from tqdm import tqdm

import torch
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from accelerate import Accelerator
from library import anima_model_spec, deepspeed_utils, strategy_base

import library.accelerator_setup as accelerator_setup
import library.anima_args as args_util
import library.dataset as dataset_util
import library.anima_model_io as model_io
import library.optimizer as optimizer_util
from library.dreambooth_dataset import DreamBoothDataset
from library.anima_model_io import SS_METADATA_MINIMUM_KEYS
import library.logging_util as logging_util
import library.checkpoint_io as checkpoint_io
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.huggingface_util as huggingface_util
import library.anima_loss as loss_util
from library.anima_loss import apply_masked_loss
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


class AnimaNetworkTrainerBase:
    def __init__(self):
        self.vae_scale_factor = 0.18215

    # TODO 他のスクリプトと共通化する
    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        mean_grad_norm=None,
        mean_combined_norm=None,
    ):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/max_key_norm"] = maximum_norm
        if mean_norm is not None:
            logs["norm/avg_key_norm"] = mean_norm
        if mean_grad_norm is not None:
            logs["norm/avg_grad_norm"] = mean_grad_norm
        if mean_combined_norm is not None:
            logs["norm/avg_combined_norm"] = mean_combined_norm

        lrs = lr_scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i - (0 if args.network_train_unet_only else 1)
                if idx == -1:
                    lr_desc = "textencoder"
                else:
                    if len(lrs) > 2:
                        lr_desc = f"group{i}"
                    else:
                        lr_desc = "unet"

            logs[f"lr/{lr_desc}"] = lr

            if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower().startswith("Prodigy".lower()):
                opt = lr_scheduler.optimizers[-1] if hasattr(lr_scheduler, "optimizers") else optimizer
                if opt is not None:
                    logs[f"lr/d*lr/{lr_desc}"] = opt.param_groups[i]["d"] * opt.param_groups[i]["lr"]
                    if "effective_lr" in opt.param_groups[i]:
                        logs[f"lr/d*eff_lr/{lr_desc}"] = opt.param_groups[i]["d"] * opt.param_groups[i]["effective_lr"]

        return logs

    def step_logging(self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
        self.accelerator_logging(accelerator, logs, global_step, global_step, epoch)

    def epoch_logging(self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
        self.accelerator_logging(accelerator, logs, epoch, global_step, epoch)

    def val_logging(self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int, val_step: int):
        self.accelerator_logging(accelerator, logs, global_step + val_step, global_step, epoch, val_step)

    def accelerator_logging(
        self, accelerator: Accelerator, logs: dict, step_value: int, global_step: int, epoch: int, val_step: Optional[int] = None
    ):
        """
        step_value is for tensorboard, other values are for wandb
        """
        tensorboard_tracker = None
        wandb_tracker = None
        other_trackers = []
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tensorboard_tracker = accelerator.get_tracker("tensorboard")
            elif tracker.name == "wandb":
                wandb_tracker = accelerator.get_tracker("wandb")
            else:
                other_trackers.append(accelerator.get_tracker(tracker.name))

        if tensorboard_tracker is not None:
            tensorboard_tracker.log(logs, step=step_value)

        if wandb_tracker is not None:
            logs["global_step"] = global_step
            logs["epoch"] = epoch
            if val_step is not None:
                logs["val_step"] = val_step
            wandb_tracker.log(logs)

        for tracker in other_trackers:
            tracker.log(logs, step=step_value)

    # returns a list of bool values indicating whether each text encoder should be trained
    def get_text_encoders_train_flags(self, args, text_encoders):
        return [True] * len(text_encoders) if self.is_train_text_encoder(args) else [False] * len(text_encoders)

    def is_train_text_encoder(self, args):
        return not args.network_train_unet_only

    def all_reduce_network(self, accelerator, network):
        for param in network.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

    def on_step_start(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train: bool = True):
        pass

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
        text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
    ) -> torch.Tensor:
        """Encode one Anima batch and compute its flow-matching loss."""

        with torch.no_grad():
            if batch.get("latents") is not None:
                latents = typing.cast(torch.FloatTensor, batch["latents"].to(accelerator.device))
            else:
                images = batch["images"]
                if args.vae_batch_size is None or len(images) <= args.vae_batch_size:
                    latents = self.encode_images_to_latents(args, vae, images.to(accelerator.device, dtype=vae_dtype))
                else:
                    latent_chunks = []
                    for start in range(0, len(images), args.vae_batch_size):
                        chunk = images[start : start + args.vae_batch_size].to(accelerator.device, dtype=vae_dtype)
                        latent_chunks.append(self.encode_images_to_latents(args, vae, chunk))
                    latents = torch.cat(latent_chunks, dim=0)
                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = typing.cast(torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents))
            latents = self.shift_scale_latents(args, latents)

        text_encoder_conditions = batch.get("text_encoder_outputs_list") or []
        if not text_encoder_conditions or text_encoder_conditions[0] is None or train_text_encoder:
            with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
                input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                encoded_conditions = text_encoding_strategy.encode_tokens(
                    tokenize_strategy,
                    self.get_models_for_text_encoding(args, accelerator, text_encoders),
                    input_ids,
                )
                if args.full_fp16:
                    encoded_conditions = [condition.to(weight_dtype) for condition in encoded_conditions]
            if not text_encoder_conditions:
                text_encoder_conditions = encoded_conditions
            else:
                for index, condition in enumerate(encoded_conditions):
                    if condition is not None:
                        text_encoder_conditions[index] = condition

        prediction, target, timesteps, weighting = self.get_noise_pred_and_target(
            args,
            accelerator,
            noise_scheduler,
            latents,
            batch,
            text_encoder_conditions,
            unet,
            network,
            weight_dtype,
            train_unet,
            is_train=is_train,
        )
        huber_c = loss_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
        loss = loss_util.conditional_loss(prediction.float(), target.float(), args.loss_type, "none", huber_c)
        if args.masked_loss or batch.get("alpha_masks") is not None:
            loss = apply_masked_loss(loss, batch)
        loss = loss_util.reduce_weighted_loss(loss, weighting, batch["loss_weights"])
        return self.post_process_loss(loss, args, timesteps, noise_scheduler).mean()

    def cast_text_encoder(self, args):
        return True

    def cast_vae(self, args):
        return True

    def cast_unet(self, args):
        return True

    def _build_metadata(
        self,
        args,
        *,
        session_id: int,
        training_started_at: float,
        model_version: str,
        text_encoder_lr,
        optimizer_name: str,
        optimizer_args: str,
        num_train_epochs: int,
        num_batches_per_epoch: int,
        net_kwargs: dict,
        total_batch_size: int,
        train_dataset_group,
        val_dataset_group,
        use_user_config: bool,
        use_dreambooth_method: bool,
    ) -> None:
        """Build training metadata dict and the minimum_metadata subset.

        Stores the result on ``self._metadata`` / ``self._minimum_metadata``. The
        ``_metadata`` dict is later mutated by ``_save_model`` (ss_training_finished_at /
        ss_steps / ss_epoch).
        """
        # TODO refactor metadata creation and move to util
        metadata = {
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_text_encoder_lr": text_encoder_lr,
            "ss_unet_lr": args.unet_lr,
            "ss_num_train_images": train_dataset_group.num_train_images,
            "ss_num_validation_images": val_dataset_group.num_train_images if val_dataset_group is not None else 0,
            "ss_num_reg_images": train_dataset_group.num_reg_images,
            "ss_num_batches_per_epoch": num_batches_per_epoch,
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_network_module": args.network_module,
            "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
            "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
            "ss_mixed_precision": args.mixed_precision,
            "ss_full_fp16": bool(args.full_fp16),
            "ss_base_model_version": model_version,
            "ss_cache_latents": bool(args.cache_latents),
            "ss_seed": args.seed,
            "ss_lowram": args.lowram,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            "ss_sd_scripts_commit_hash": model_io.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_caption_dropout_rate": args.caption_dropout_rate,
            "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
            "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
            "ss_face_crop_aug_range": args.face_crop_aug_range,
            "ss_prior_loss_weight": args.prior_loss_weight,
            "ss_scale_weight_norms": args.scale_weight_norms,
            "ss_ip_noise_gamma": args.ip_noise_gamma,
            "ss_ip_noise_gamma_random_strength": args.ip_noise_gamma_random_strength,
            "ss_loss_type": args.loss_type,
            "ss_huber_schedule": args.huber_schedule,
            "ss_huber_scale": args.huber_scale,
            "ss_huber_c": args.huber_c,
            "ss_validation_seed": args.validation_seed,
            "ss_validation_split": args.validation_split,
            "ss_max_validation_steps": args.max_validation_steps,
            "ss_validate_every_n_epochs": args.validate_every_n_epochs,
            "ss_validate_every_n_steps": args.validate_every_n_steps,
            "ss_resize_interpolation": args.resize_interpolation,
        }

        self.update_metadata(metadata, args)  # architecture specific metadata

        if use_user_config:
            # save metadata of multiple datasets
            # NOTE: pack "ss_datasets" value as json one time
            #   or should also pack nested collections as json?
            datasets_metadata = []
            tag_frequency = {}  # merge tag frequency for metadata editor
            dataset_dirs_info = {}  # merge subset dirs for metadata editor

            for dataset in train_dataset_group.datasets:
                is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
                dataset_metadata = {
                    "is_dreambooth": is_dreambooth_dataset,
                    "batch_size_per_device": dataset.batch_size,
                    "num_train_images": dataset.num_train_images,  # includes repeating
                    "num_reg_images": dataset.num_reg_images,
                    "resolution": (dataset.width, dataset.height),
                    "enable_bucket": bool(dataset.enable_bucket),
                    "min_bucket_reso": dataset.min_bucket_reso,
                    "max_bucket_reso": dataset.max_bucket_reso,
                    "skip_image_resolution": dataset.skip_image_resolution,
                    "tag_frequency": dataset.tag_frequency,
                    "bucket_info": dataset.bucket_info,
                    "resize_interpolation": dataset.resize_interpolation,
                }

                subsets_metadata = []
                for subset in dataset.subsets:
                    subset_metadata = {
                        "img_count": subset.img_count,
                        "num_repeats": subset.num_repeats,
                        "color_aug": bool(subset.color_aug),
                        "flip_aug": bool(subset.flip_aug),
                        "random_crop": bool(subset.random_crop),
                        "shuffle_caption": bool(subset.shuffle_caption),
                        "keep_tokens": subset.keep_tokens,
                        "keep_tokens_separator": subset.keep_tokens_separator,
                        "secondary_separator": subset.secondary_separator,
                        "enable_wildcard": bool(subset.enable_wildcard),
                        "caption_prefix": subset.caption_prefix,
                        "caption_suffix": subset.caption_suffix,
                        "resize_interpolation": subset.resize_interpolation,
                    }

                    image_dir_or_metadata_file = None
                    if subset.image_dir:
                        image_dir = os.path.basename(subset.image_dir)
                        subset_metadata["image_dir"] = image_dir
                        image_dir_or_metadata_file = image_dir

                    if is_dreambooth_dataset:
                        subset_metadata["class_tokens"] = subset.class_tokens
                        subset_metadata["is_reg"] = subset.is_reg
                        if subset.is_reg:
                            image_dir_or_metadata_file = None  # not merging reg dataset
                    else:
                        metadata_file = os.path.basename(subset.metadata_file)
                        subset_metadata["metadata_file"] = metadata_file
                        image_dir_or_metadata_file = metadata_file  # may overwrite

                    subsets_metadata.append(subset_metadata)

                    # merge dataset dir: not reg subset only
                    # TODO update additional-network extension to show detailed dataset config from metadata
                    if image_dir_or_metadata_file is not None:
                        # datasets may have a certain dir multiple times
                        v = image_dir_or_metadata_file
                        i = 2
                        while v in dataset_dirs_info:
                            v = image_dir_or_metadata_file + f" ({i})"
                            i += 1
                        image_dir_or_metadata_file = v

                        dataset_dirs_info[image_dir_or_metadata_file] = {
                            "n_repeats": subset.num_repeats,
                            "img_count": subset.img_count,
                        }

                dataset_metadata["subsets"] = subsets_metadata
                datasets_metadata.append(dataset_metadata)

                # merge tag frequency:
                for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                    # あるディレクトリが複数のdatasetで使用されている場合、一度だけ数える
                    # もともと繰り返し回数を指定しているので、キャプション内でのタグの出現回数と、それが学習で何度使われるかは一致しない
                    # なので、ここで複数datasetの回数を合算してもあまり意味はない
                    if ds_dir_name in tag_frequency:
                        continue
                    tag_frequency[ds_dir_name] = ds_freq_for_dir

            metadata["ss_datasets"] = json.dumps(datasets_metadata)
            metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
            metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
        else:
            # conserving backward compatibility when using train_dataset_dir and reg_dataset_dir
            assert (
                len(train_dataset_group.datasets) == 1
            ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug. / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"

            dataset = train_dataset_group.datasets[0]

            dataset_dirs_info = {}
            reg_dataset_dirs_info = {}
            if use_dreambooth_method:
                for subset in dataset.subsets:
                    info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
                    info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}
            else:
                for subset in dataset.subsets:
                    dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,
                    }

            metadata.update(
                {
                    "ss_batch_size_per_device": args.train_batch_size,
                    "ss_total_batch_size": total_batch_size,
                    "ss_resolution": args.resolution,
                    "ss_color_aug": bool(args.color_aug),
                    "ss_flip_aug": bool(args.flip_aug),
                    "ss_random_crop": bool(args.random_crop),
                    "ss_shuffle_caption": bool(args.shuffle_caption),
                    "ss_enable_bucket": bool(dataset.enable_bucket),
                    "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
                    "ss_bucket_free_fit": bool(dataset.bucket_free_fit),
                    "ss_min_bucket_reso": dataset.min_bucket_reso,
                    "ss_max_bucket_reso": dataset.max_bucket_reso,
                    "ss_skip_image_resolution": dataset.skip_image_resolution,
                    "ss_keep_tokens": args.keep_tokens,
                    "ss_dataset_dirs": json.dumps(dataset_dirs_info),
                    "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
                    "ss_tag_frequency": json.dumps(dataset.tag_frequency),
                    "ss_bucket_info": json.dumps(dataset.bucket_info),
                }
            )

        # add extra args
        if args.network_args:
            metadata["ss_network_args"] = json.dumps(net_kwargs)

        # model name and hash
        if args.pretrained_model_name_or_path is not None:
            sd_model_name = args.pretrained_model_name_or_path
            if os.path.exists(sd_model_name):
                metadata["ss_sd_model_hash"] = model_io.model_hash(sd_model_name)
                metadata["ss_new_sd_model_hash"] = model_io.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        if args.vae is not None:
            vae_name = args.vae
            if os.path.exists(vae_name):
                metadata["ss_vae_hash"] = model_io.model_hash(vae_name)
                metadata["ss_new_vae_hash"] = model_io.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name

        metadata = {k: str(v) for k, v in metadata.items()}

        # make minimum metadata for filtering
        minimum_metadata = {}
        for key in SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        self._metadata = metadata
        self._minimum_metadata = minimum_metadata

    def _run_validation_loop(
        self,
        *,
        mode: Literal["step", "epoch"],
        accelerator: Accelerator,
        args: argparse.Namespace,
        network,
        text_encoders,
        unet,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy,
        val_dataloader,
        validation_steps: int,
        validation_timesteps,
        validation_total_steps: int,
        train_text_encoder: bool,
        train_unet: bool,
        epoch: int,
        global_step: int,
        is_tracking: bool,
        loss_recorder: "logging_util.LossRecorder",
        train_loss_recorder: "logging_util.LossRecorder",
    ) -> None:
        """Run one validation pass: dataloader x validation_timesteps.

        ``mode`` selects the per-step vs per-epoch variant which only differs in
        tqdm desc / progress postfix key / log keys / log function. The core
        evaluation loop and ``on_step_start`` -> ``args.min/max_timestep``
        assignment order are identical across modes.

        Caller is responsible for the surrounding setup/teardown:
        ``optimizer_eval_fn`` / ``network.eval()`` / ``switch_rng_state`` before,
        and ``restore_rng_state`` / restoring ``args.min/max_timestep`` /
        ``optimizer_train_fn`` / ``network.train()`` / ``progress_bar.unpause()``
        after.
        """
        if mode == "step":
            tqdm_desc = "validation steps"
            progress_postfix_key = "val_avg_loss"
            log_key_average = "loss/validation/step_average"
            log_key_divergence = "loss/validation/step_divergence"
            log_fn = self.step_logging
        else:  # mode == "epoch"
            tqdm_desc = "epoch validation steps"
            progress_postfix_key = "val_epoch_avg_loss"
            log_key_average = "loss/validation/epoch_average"
            log_key_divergence = "loss/validation/epoch_divergence"
            log_fn = self.epoch_logging

        val_progress_bar = tqdm(
            range(validation_total_steps),
            smoothing=0,
            disable=not accelerator.is_local_main_process,
            desc=tqdm_desc,
        )

        val_timesteps_step = 0
        for val_step, batch in enumerate(val_dataloader):
            if val_step >= validation_steps:
                break

            for timestep in validation_timesteps:
                self.on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=False)

                args.min_timestep = args.max_timestep = timestep  # dirty hack to change timestep

                loss = self.process_batch(
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
                    is_train=False,
                    train_text_encoder=train_text_encoder,  # this is needed for validation because Text Encoders must be called if train_text_encoder is True
                    train_unet=train_unet,
                )

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=val_timesteps_step, loss=current_loss)
                val_progress_bar.update(1)
                val_progress_bar.set_postfix(
                    {progress_postfix_key: loss_recorder.moving_average, "timestep": timestep}
                )

                self.on_validation_step_end(args, accelerator, network, text_encoders, unet, batch, weight_dtype)
                val_timesteps_step += 1

        if is_tracking:
            loss_validation_divergence = loss_recorder.moving_average - train_loss_recorder.moving_average
            logs = {
                log_key_average: loss_recorder.moving_average,
                log_key_divergence: loss_validation_divergence,
            }
            log_fn(accelerator, logs, global_step, epoch + 1)

    def _save_model(
        self,
        *,
        args: argparse.Namespace,
        accelerator: Accelerator,
        save_dtype,
        ckpt_name: str,
        unwrapped_nw,
        steps: int,
        epoch_no: int,
        force_sync_upload: bool = False,
    ):
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        self._metadata["ss_training_finished_at"] = str(time.time())
        self._metadata["ss_steps"] = str(steps)
        self._metadata["ss_epoch"] = str(epoch_no)

        metadata_to_save = self._minimum_metadata if args.no_metadata else self._metadata
        sai_metadata = self.get_sai_model_spec(args)
        metadata_to_save.update(sai_metadata)

        unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)
        sidecar_paths = []
        if hasattr(unwrapped_nw, "get_checkpoint_sidecar_paths"):
            sidecar_paths = unwrapped_nw.get_checkpoint_sidecar_paths(ckpt_file)
            missing = [path for path in sidecar_paths if not os.path.isfile(path)]
            if missing:
                raise RuntimeError(f"network did not write its declared checkpoint sidecars: {missing}")
        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)
            for sidecar_path in sidecar_paths:
                huggingface_util.upload(
                    args,
                    sidecar_path,
                    "/" + os.path.basename(sidecar_path),
                    force_sync_upload=force_sync_upload,
                )

    def _remove_model(self, *, args: argparse.Namespace, accelerator: Accelerator, old_ckpt_name: str, unwrapped_nw=None):
        old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
        artifact_paths = [old_ckpt_file]
        if unwrapped_nw is not None and hasattr(unwrapped_nw, "get_checkpoint_sidecar_paths"):
            artifact_paths.extend(unwrapped_nw.get_checkpoint_sidecar_paths(old_ckpt_file))
        for artifact_path in artifact_paths:
            if os.path.exists(artifact_path):
                accelerator.print(f"removing old checkpoint artifact: {artifact_path}")
                os.remove(artifact_path)

    def train(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        args_util.verify_training_args(args)
        accelerator_setup.prepare_dataset_args(args, True)
        deepspeed_utils.prepare_deepspeed_args(args)
        setup_logging(args, reset=True)

        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        tokenize_strategy = self.get_tokenize_strategy(args)
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
        tokenizers = self.get_tokenizers(tokenize_strategy)  # will be removed after sample_image is refactored

        # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
        latents_caching_strategy = self.get_latents_caching_strategy(args)
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

        # データセットを準備する
        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
            if use_user_config:
                logger.info(f"Loading dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    logger.warning(
                        "ignoring the following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                            ", ".join(ignored)
                        )
                    )
            else:
                if use_dreambooth_method:
                    logger.info("Using DreamBooth method.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                    args.train_data_dir, args.reg_data_dir
                                )
                            }
                        ]
                    }
                else:
                    logger.info("Training with captions.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": [
                                    {
                                        "image_dir": args.train_data_dir,
                                        "metadata_file": args.in_json,
                                    }
                                ]
                            }
                        ]
                    }

            blueprint = blueprint_generator.generate(user_config, args)
            train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        else:
            # use arbitrary dataset class
            train_dataset_group = dataset_util.load_arbitrary_dataset(args)
            val_dataset_group = None  # placeholder until validation dataset supported for arbitrary

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = dataset_util.collator_class(current_epoch, current_step, ds_for_collator)

        if args.debug_dataset:
            train_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
            dataset_util.debug_dataset(train_dataset_group)

            if val_dataset_group is not None:
                val_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
                dataset_util.debug_dataset(val_dataset_group)
            return
        if len(train_dataset_group) == 0:
            logger.error(
                "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
            )
            return

        if cache_latents:
            assert (
                train_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"
            if val_dataset_group is not None:
                assert (
                    val_dataset_group.is_latent_cacheable()
                ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

        self.assert_extra_args(args, train_dataset_group, val_dataset_group)  # may change some args

        # acceleratorを準備する
        logger.info("preparing accelerator")
        accelerator = accelerator_setup.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        # mixed precisionに対応した型を用意しておき適宜castする
        weight_dtype, save_dtype = accelerator_setup.prepare_dtype(args)
        vae_dtype = (torch.float32 if args.no_half_vae else weight_dtype) if self.cast_vae(args) else None

        # load target models: unet may be None for lazy loading
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)
        if vae_dtype is None:
            vae_dtype = vae.dtype
            logger.info(f"vae_dtype is set to {vae_dtype} by the model since cast_vae() is false")

        # Normalize the Anima text encoder to a list for shared loop handling.
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        # prepare dataset for latents caching if needed
        if cache_latents:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()

            train_dataset_group.new_cache_latents(vae, accelerator)
            if val_dataset_group is not None:
                val_dataset_group.new_cache_latents(vae, accelerator)

            vae.to("cpu")
            clean_memory_on_device(accelerator.device)

            accelerator.wait_for_everyone()

        # 必要ならテキストエンコーダーの出力をキャッシュする: Text Encoderはcpuまたはgpuへ移される
        # cache text encoder outputs if needed: Text Encoder is moved to cpu or gpu
        text_encoding_strategy = self.get_text_encoding_strategy(args)
        strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

        text_encoder_outputs_caching_strategy = self.get_text_encoder_outputs_caching_strategy(args)
        if text_encoder_outputs_caching_strategy is not None:
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_outputs_caching_strategy)
        self.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, train_dataset_group, weight_dtype)
        if val_dataset_group is not None:
            self.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, val_dataset_group, weight_dtype)

        if unet is None:
            # lazy load unet if needed. text encoders may be freed or replaced with dummy models for saving memory
            unet, text_encoders = self.load_unet_lazily(args, weight_dtype, accelerator, text_encoders)

        # 差分追加学習のためにモデルを読み込む
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

        if args.base_weights is not None:
            # base_weights が指定されている場合は、指定された重みを読み込みマージする
            for i, weight_path in enumerate(args.base_weights):
                if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]

                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

                module, weights_sd = network_module.create_network_from_weights(
                    multiplier, weight_path, vae, text_encoder, unet, for_inference=True
                )
                is_mergeable = getattr(module, "is_mergeable", None)
                if not callable(is_mergeable) or not is_mergeable() or not callable(getattr(module, "merge_to", None)):
                    raise ValueError(
                        f"{args.network_module} checkpoint {weight_path} contains a full Anima adapter and cannot be "
                        "merged with --base_weights; load it as --network_weights instead"
                    )
                module.merge_to(text_encoder, unet, weights_sd, weight_dtype, accelerator.device if args.lowram else "cpu")

            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=", 1)
                net_kwargs[key] = value

        if "dropout" not in net_kwargs:
            # Keep resume/dim-from-weights behavior consistent with newly
            # created networks and with LyCORIS-style module arguments.
            net_kwargs["dropout"] = args.network_dropout

        # if a new network is added in future, add if ~ then blocks for each network (;'∀')
        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet, **net_kwargs)
        else:
            network = network_module.create_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                text_encoder,
                unet,
                neuron_dropout=args.network_dropout,
                **net_kwargs,
            )
        if network is None:
            return
        # TODO remove `hasattr` by setting up methods if not defined in the network like below  (hacky but will work):
        # if not hasattr(network, "prepare_network"):
        #    network.prepare_network = lambda args: None

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
            logger.warning(
                "warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません"
            )
            args.scale_weight_norms = False

        self.post_process_network(args, accelerator, network, text_encoders, unet)

        # apply network to unet and text_encoder
        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = self.is_train_text_encoder(args)
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        if args.network_weights is not None:
            # FIXME consider alpha of weights: this assumes that the alpha is not changed
            info = network.load_weights(args.network_weights)
            accelerator.print(f"load network weights from {args.network_weights}: {info}")

        if args.gradient_checkpointing:
            if args.cpu_offload_checkpointing:
                unet.enable_gradient_checkpointing(cpu_offload=True)
            else:
                unet.enable_gradient_checkpointing()

            for t_enc, flag in zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders)):
                if flag:
                    if t_enc.supports_gradient_checkpointing:
                        t_enc.gradient_checkpointing_enable()
            del t_enc
            network.enable_gradient_checkpointing()  # may have no effect

        # 学習に必要なクラスを準備する
        accelerator.print("prepare optimizer, data loader etc.")

        # make backward compatibility for text_encoder_lr
        support_multiple_lrs = hasattr(network, "prepare_optimizer_params_with_multiple_te_lrs")
        if support_multiple_lrs:
            text_encoder_lr = args.text_encoder_lr
        else:
            # toml backward compatibility
            if args.text_encoder_lr is None or isinstance(args.text_encoder_lr, float) or isinstance(args.text_encoder_lr, int):
                text_encoder_lr = args.text_encoder_lr
            else:
                text_encoder_lr = None if len(args.text_encoder_lr) == 0 else args.text_encoder_lr[0]
        try:
            if support_multiple_lrs:
                results = network.prepare_optimizer_params_with_multiple_te_lrs(text_encoder_lr, args.unet_lr, args.learning_rate)
            else:
                results = network.prepare_optimizer_params(text_encoder_lr, args.unet_lr, args.learning_rate)
            if type(results) is tuple:
                trainable_params = results[0]
                lr_descriptions = results[1]
            else:
                trainable_params = results
                lr_descriptions = None
        except TypeError:
            trainable_params = network.prepare_optimizer_params(text_encoder_lr, args.unet_lr)
            lr_descriptions = None

        # if len(trainable_params) == 0:
        #     accelerator.print("no trainable parameters found / 学習可能なパラメータが見つかりませんでした")
        # for params in trainable_params:
        #     for k, v in params.items():
        #         if type(v) == float:
        #             pass
        #         else:
        #             v = len(v)
        #         accelerator.print(f"trainable_params: {k} = {v}")

        optimizer_name, optimizer_args, optimizer = optimizer_util.get_optimizer(args, trainable_params)
        optimizer_train_fn, optimizer_eval_fn = optimizer_util.get_optimizer_train_eval_fn(optimizer, args)

        # prepare dataloader
        # strategies are set here because they cannot be referenced in another process. Copy them with the dataset
        # some strategies can be None
        train_dataset_group.set_current_strategies()
        if val_dataset_group is not None:
            val_dataset_group.set_current_strategies()

        # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset_group if val_dataset_group is not None else [],
            shuffle=False,
            batch_size=1,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        # 学習ステップ数を計算する
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
            )

        # データセット側にも学習ステップを送信
        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # lr schedulerを用意する
        lr_scheduler = optimizer_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

        # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
        if args.full_fp16:
            assert (
                args.mixed_precision == "fp16"
            ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert (
                args.mixed_precision == "bf16"
            ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)

        unet_weight_dtype = te_weight_dtype = weight_dtype

        unet.requires_grad_(False)
        if self.cast_unet(args):
            unet.to(dtype=unet_weight_dtype)
        for i, t_enc in enumerate(text_encoders):
            t_enc.requires_grad_(False)

            # in case of cpu, dtype is already set to fp32 because cpu does not support fp8/fp16/bf16
            if t_enc.device.type != "cpu" and self.cast_text_encoder(args):
                t_enc.to(dtype=te_weight_dtype)


        # acceleratorがなんかよろしくやってくれるらしい / accelerator will do something good
        if args.deepspeed:
            flags = self.get_text_encoders_train_flags(args, text_encoders)
            ds_model = deepspeed_utils.prepare_deepspeed_model(
                args,
                unet=unet if train_unet else None,
                text_encoder1=text_encoders[0] if flags[0] else None,
                text_encoder2=(text_encoders[1] if flags[1] else None) if len(text_encoders) > 1 else None,
                network=network,
            )
            ds_model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
                ds_model, optimizer, train_dataloader, val_dataloader, lr_scheduler
            )
            training_model = ds_model
        else:
            if train_unet:
                # default implementation is:  unet = accelerator.prepare(unet)
                unet = self.prepare_unet_with_accelerator(args, accelerator, unet)  # accelerator does some magic here
            else:
                # move to device because unet is not prepared by accelerator
                unet.to(accelerator.device, dtype=unet_weight_dtype if self.cast_unet(args) else None)
            if train_text_encoder:
                text_encoders = [
                    (accelerator.prepare(t_enc) if flag else t_enc)
                    for t_enc, flag in zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders))
                ]
                if len(text_encoders) > 1:
                    text_encoder = text_encoders
                else:
                    text_encoder = text_encoders[0]
            else:
                pass  # if text_encoder is not trained, no need to prepare. and device and dtype are already set

            network, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
                network, optimizer, train_dataloader, val_dataloader, lr_scheduler
            )
            training_model = network

        if args.gradient_checkpointing:
            # ``train()`` is required for the prepared text-encoder path.
            unet.train()
            for i, (t_enc, frag) in enumerate(zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders))):
                t_enc.train()

                # set top parameter requires_grad = True for gradient checkpointing works
                if frag:
                    self.prepare_text_encoder_grad_ckpt_workaround(i, t_enc)

        else:
            unet.eval()
            for t_enc in text_encoders:
                t_enc.eval()

        del t_enc

        accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)

        needs_vae_during_training = hasattr(accelerator.unwrap_model(network), "set_condition_latents")
        if not cache_latents or needs_vae_during_training:
            # EasyControl still encodes its conditioning image every step even
            # when target latents have been cached.
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
        if args.full_fp16:
            accelerator_setup.patch_accelerator_for_fp16_training(accelerator)

        # before resuming make hook for saving/loading to save/load the network weights only
        def save_model_hook(models, weights, output_dir):
            # pop weights of other models than network to save only network weights
            # only main process or deepspeed https://github.com/huggingface/diffusers/issues/2606
            if accelerator.is_main_process or args.deepspeed:
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, type(accelerator.unwrap_model(network))):
                        remove_indices.append(i)
                for i in reversed(remove_indices):
                    if len(weights) > i:
                        weights.pop(i)
                # print(f"save model hook: {len(weights)} weights will be saved")

            # save current ecpoch and step
            train_state_file = os.path.join(output_dir, "train_state.json")
            # +1 is needed because the state is saved before current_step is set from global_step
            logger.info(f"save train state to {train_state_file} at epoch {current_epoch.value} step {current_step.value+1}")
            with open(train_state_file, "w", encoding="utf-8") as f:
                json.dump({"current_epoch": current_epoch.value, "current_step": current_step.value + 1}, f)

        steps_from_state = None

        def load_model_hook(models, input_dir):
            # remove models except network
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(network))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                models.pop(i)
            # print(f"load model hook: {len(models)} models will be loaded")

            # load current epoch and step to
            nonlocal steps_from_state
            train_state_file = os.path.join(input_dir, "train_state.json")
            if os.path.exists(train_state_file):
                with open(train_state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                steps_from_state = data["current_step"]
                logger.info(f"load train state from {train_state_file}: {data}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        # resumeする
        args_util.resume_from_local_or_hf_if_specified(accelerator, args)

        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

        # 学習する
        # TODO: find a way to handle total batch size when there are multiple datasets
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        accelerator.print("running training / 学習開始")
        accelerator.print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
        accelerator.print(
            f"  num validation images * repeats / 学習画像の数×繰り返し回数: {val_dataset_group.num_train_images if val_dataset_group is not None else 0}"
        )
        accelerator.print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
        accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
        accelerator.print(
            f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
        )
        # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
        accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

        self._build_metadata(
            args,
            session_id=session_id,
            training_started_at=training_started_at,
            model_version=model_version,
            text_encoder_lr=text_encoder_lr,
            optimizer_name=optimizer_name,
            optimizer_args=optimizer_args,
            num_train_epochs=num_train_epochs,
            num_batches_per_epoch=len(train_dataloader),
            net_kwargs=net_kwargs,
            total_batch_size=total_batch_size,
            train_dataset_group=train_dataset_group,
            val_dataset_group=val_dataset_group,
            use_user_config=use_user_config,
            use_dreambooth_method=use_dreambooth_method,
        )

        # calculate steps to skip when resuming or starting from a specific step
        initial_step = 0
        if args.initial_epoch is not None or args.initial_step is not None:
            # if initial_epoch or initial_step is specified, steps_from_state is ignored even when resuming
            if steps_from_state is not None:
                logger.warning(
                    "steps from the state is ignored because initial_step is specified / initial_stepが指定されているため、stateからのステップ数は無視されます"
                )
            if args.initial_step is not None:
                initial_step = args.initial_step
            else:
                # num steps per epoch is calculated by num_processes and gradient_accumulation_steps
                initial_step = (args.initial_epoch - 1) * math.ceil(
                    len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
                )
        else:
            # if initial_epoch and initial_step are not specified, steps_from_state is used when resuming
            if steps_from_state is not None:
                initial_step = steps_from_state
                steps_from_state = None

        if initial_step > 0:
            assert (
                args.max_train_steps > initial_step
            ), f"max_train_steps should be greater than initial step / max_train_stepsは初期ステップより大きい必要があります: {args.max_train_steps} vs {initial_step}"

        epoch_to_start = 0
        if initial_step > 0:
            if args.skip_until_initial_step:
                # if skip_until_initial_step is specified, load data and discard it to ensure the same data is used
                if not args.resume:
                    logger.info(
                        f"initial_step is specified but not resuming. lr scheduler will be started from the beginning / initial_stepが指定されていますがresumeしていないため、lr schedulerは最初から始まります"
                    )
                logger.info(f"skipping {initial_step} steps / {initial_step}ステップをスキップします")
                initial_step *= args.gradient_accumulation_steps

                # set epoch to start to make initial_step less than len(train_dataloader)
                epoch_to_start = initial_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            else:
                # if not, only epoch no is skipped for informative purpose
                epoch_to_start = initial_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
                initial_step = 0  # do not skip

        global_step = 0

        noise_scheduler = self.get_noise_scheduler(args, accelerator.device)

        logging_util.init_trackers(accelerator, args, "network_train")

        loss_recorder = logging_util.LossRecorder()
        val_step_loss_recorder = logging_util.LossRecorder()
        val_epoch_loss_recorder = logging_util.LossRecorder()

        del train_dataset_group
        if val_dataset_group is not None:
            del val_dataset_group

        # callback for step start
        if hasattr(accelerator.unwrap_model(network), "on_step_start"):
            on_step_start_for_network = accelerator.unwrap_model(network).on_step_start
        else:
            on_step_start_for_network = lambda *args, **kwargs: None

        # if text_encoder is not needed for training, delete it to save memory.
        if self.is_text_encoder_not_needed_for_training(args):
            logger.info("text_encoder is not needed for training. deleting to save memory.")
            for t_enc in text_encoders:
                del t_enc
            text_encoders = []
            text_encoder = None
            gc.collect()
            clean_memory_on_device(accelerator.device)

        # For --sample_at_first
        optimizer_eval_fn()
        self.sample_images(accelerator, args, 0, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)
        optimizer_train_fn()
        is_tracking = len(accelerator.trackers) > 0
        if is_tracking:
            # log empty object to commit the sample images to wandb
            accelerator.log({}, step=0)

        # training loop
        if initial_step > 0:  # only if skip_until_initial_step is specified
            for skip_epoch in range(epoch_to_start):  # skip epochs
                logger.info(f"skipping epoch {skip_epoch+1} because initial_step (multiplied) is {initial_step}")
                initial_step -= len(train_dataloader)
            global_step = initial_step

        # log device and dtype for each model
        logger.info(f"Anima DiT dtype: {unet_weight_dtype}, device: {unet.device}")
        for i, t_enc in enumerate(text_encoders):
            parameter = next(t_enc.parameters())
            logger.info(f"text_encoder [{i}] dtype: {parameter.dtype}, device: {parameter.device}")

        clean_memory_on_device(accelerator.device)

        progress_bar = tqdm(
            range(args.max_train_steps - initial_step), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps"
        )

        validation_steps = (
            min(args.max_validation_steps, len(val_dataloader)) if args.max_validation_steps is not None else len(val_dataloader)
        )
        NUM_VALIDATION_TIMESTEPS = 4  # 200, 400, 600, 800 TODO make this configurable
        min_timestep = 0 if args.min_timestep is None else args.min_timestep
        max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep
        validation_timesteps = np.linspace(min_timestep, max_timestep, (NUM_VALIDATION_TIMESTEPS + 2), dtype=int)[1:-1]
        validation_total_steps = validation_steps * len(validation_timesteps)
        original_args_min_timestep = args.min_timestep
        original_args_max_timestep = args.max_timestep

        def switch_rng_state(seed: int) -> tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple]:
            cpu_rng_state = torch.get_rng_state()
            if accelerator.device.type == "cuda":
                gpu_rng_state = torch.cuda.get_rng_state()
            elif accelerator.device.type == "xpu":
                gpu_rng_state = torch.xpu.get_rng_state()
            elif accelerator.device.type == "mps":
                gpu_rng_state = torch.mps.get_rng_state()
            else:
                gpu_rng_state = None
            python_rng_state = random.getstate()

            torch.manual_seed(seed)
            random.seed(seed)

            return (cpu_rng_state, gpu_rng_state, python_rng_state)

        def restore_rng_state(rng_states: tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple]):
            cpu_rng_state, gpu_rng_state, python_rng_state = rng_states
            torch.set_rng_state(cpu_rng_state)
            if gpu_rng_state is not None:
                if accelerator.device.type == "cuda":
                    torch.cuda.set_rng_state(gpu_rng_state)
                elif accelerator.device.type == "xpu":
                    torch.xpu.set_rng_state(gpu_rng_state)
                elif accelerator.device.type == "mps":
                    torch.mps.set_rng_state(gpu_rng_state)
            random.setstate(python_rng_state)

        for epoch in range(epoch_to_start, num_train_epochs):
            accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}\n")
            current_epoch.value = epoch + 1

            self._metadata["ss_epoch"] = str(epoch + 1)

            accelerator.unwrap_model(network).on_epoch_start(text_encoder, unet)  # network.train() is called here

            # TRAINING
            skipped_dataloader = None
            if initial_step > 0:
                skipped_dataloader = accelerator.skip_first_batches(train_dataloader, initial_step - 1)
                initial_step = 1

            for step, batch in enumerate(skipped_dataloader or train_dataloader):
                current_step.value = global_step
                if initial_step > 0:
                    initial_step -= 1
                    continue

                with accelerator.accumulate(training_model):
                    on_step_start_for_network(text_encoder, unet)

                    # preprocess batch for each model
                    self.on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=True)

                    loss = self.process_batch(
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
                        train_text_encoder=train_text_encoder,
                        train_unet=train_unet,
                    )

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        self.all_reduce_network(accelerator, network)  # sync DDP grad manually
                        if args.max_grad_norm != 0.0:
                            params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                        if hasattr(network, "update_grad_norms"):
                            network.update_grad_norms()
                        if hasattr(network, "update_norms"):
                            network.update_norms()

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if args.scale_weight_norms:
                    keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(network).apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device
                    )
                    mean_grad_norm = None
                    mean_combined_norm = None
                    max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                else:
                    if hasattr(network, "weight_norms"):
                        weight_norms = network.weight_norms()
                        mean_norm = weight_norms.mean().item() if weight_norms is not None else None
                        grad_norms = network.grad_norms()
                        mean_grad_norm = grad_norms.mean().item() if grad_norms is not None else None
                        combined_weight_norms = network.combined_weight_norms()
                        mean_combined_norm = combined_weight_norms.mean().item() if combined_weight_norms is not None else None
                        maximum_norm = weight_norms.max().item() if weight_norms is not None else None
                        keys_scaled = None
                        max_mean_logs = {}
                    else:
                        keys_scaled, mean_norm, maximum_norm = None, None, None
                        mean_grad_norm = None
                        mean_combined_norm = None
                        max_mean_logs = {}

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    optimizer_eval_fn()
                    self.sample_images(
                        accelerator, args, None, global_step, accelerator.device, vae, tokenizers, text_encoder, unet
                    )
                    progress_bar.unpause()

                    # 指定ステップごとにモデルを保存
                    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            ckpt_name = checkpoint_io.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                            self._save_model(
                                args=args,
                                accelerator=accelerator,
                                save_dtype=save_dtype,
                                ckpt_name=ckpt_name,
                                unwrapped_nw=accelerator.unwrap_model(network),
                                steps=global_step,
                                epoch_no=epoch,
                            )

                            if args.save_state:
                                checkpoint_io.save_and_remove_state_stepwise(args, accelerator, global_step)

                            remove_step_no = checkpoint_io.get_remove_step_no(args, global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = checkpoint_io.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                                self._remove_model(
                                    args=args,
                                    accelerator=accelerator,
                                    old_ckpt_name=remove_ckpt_name,
                                    unwrapped_nw=accelerator.unwrap_model(network),
                                )
                    optimizer_train_fn()

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if is_tracking:
                    logs = self.generate_step_logs(
                        args,
                        current_loss,
                        avr_loss,
                        lr_scheduler,
                        lr_descriptions,
                        optimizer,
                        keys_scaled,
                        mean_norm,
                        maximum_norm,
                        mean_grad_norm,
                        mean_combined_norm,
                    )
                    self.step_logging(accelerator, logs, global_step, epoch + 1)

                # VALIDATION PER STEP: global_step is already incremented
                # for example, if validate_every_n_steps=100, validate at step 100, 200, 300, ...
                should_validate_step = args.validate_every_n_steps is not None and global_step % args.validate_every_n_steps == 0
                if accelerator.sync_gradients and validation_steps > 0 and should_validate_step:
                    optimizer_eval_fn()
                    accelerator.unwrap_model(network).eval()
                    rng_states = switch_rng_state(args.validation_seed if args.validation_seed is not None else args.seed)

                    self._run_validation_loop(
                        mode="step",
                        accelerator=accelerator,
                        args=args,
                        network=network,
                        text_encoders=text_encoders,
                        unet=unet,
                        vae=vae,
                        noise_scheduler=noise_scheduler,
                        vae_dtype=vae_dtype,
                        weight_dtype=weight_dtype,
                        text_encoding_strategy=text_encoding_strategy,
                        tokenize_strategy=tokenize_strategy,
                        val_dataloader=val_dataloader,
                        validation_steps=validation_steps,
                        validation_timesteps=validation_timesteps,
                        validation_total_steps=validation_total_steps,
                        train_text_encoder=train_text_encoder,
                        train_unet=train_unet,
                        epoch=epoch,
                        global_step=global_step,
                        is_tracking=is_tracking,
                        loss_recorder=val_step_loss_recorder,
                        train_loss_recorder=loss_recorder,
                    )

                    restore_rng_state(rng_states)
                    args.min_timestep = original_args_min_timestep
                    args.max_timestep = original_args_max_timestep
                    optimizer_train_fn()
                    accelerator.unwrap_model(network).train()
                    progress_bar.unpause()

                if global_step >= args.max_train_steps:
                    break

            # EPOCH VALIDATION
            should_validate_epoch = (
                (epoch + 1) % args.validate_every_n_epochs == 0 if args.validate_every_n_epochs is not None else True
            )

            if should_validate_epoch and len(val_dataloader) > 0:
                optimizer_eval_fn()
                accelerator.unwrap_model(network).eval()
                rng_states = switch_rng_state(args.validation_seed if args.validation_seed is not None else args.seed)

                self._run_validation_loop(
                    mode="epoch",
                    accelerator=accelerator,
                    args=args,
                    network=network,
                    text_encoders=text_encoders,
                    unet=unet,
                    vae=vae,
                    noise_scheduler=noise_scheduler,
                    vae_dtype=vae_dtype,
                    weight_dtype=weight_dtype,
                    text_encoding_strategy=text_encoding_strategy,
                    tokenize_strategy=tokenize_strategy,
                    val_dataloader=val_dataloader,
                    validation_steps=validation_steps,
                    validation_timesteps=validation_timesteps,
                    validation_total_steps=validation_total_steps,
                    train_text_encoder=train_text_encoder,
                    train_unet=train_unet,
                    epoch=epoch,
                    global_step=global_step,
                    is_tracking=is_tracking,
                    loss_recorder=val_epoch_loss_recorder,
                    train_loss_recorder=loss_recorder,
                )

                restore_rng_state(rng_states)
                args.min_timestep = original_args_min_timestep
                args.max_timestep = original_args_max_timestep
                optimizer_train_fn()
                accelerator.unwrap_model(network).train()
                progress_bar.unpause()

            # END OF EPOCH
            if is_tracking:
                logs = {"loss/epoch_average": loss_recorder.moving_average}
                self.epoch_logging(accelerator, logs, global_step, epoch + 1)

            accelerator.wait_for_everyone()

            # 指定エポックごとにモデルを保存
            optimizer_eval_fn()
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = checkpoint_io.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                    self._save_model(
                        args=args,
                        accelerator=accelerator,
                        save_dtype=save_dtype,
                        ckpt_name=ckpt_name,
                        unwrapped_nw=accelerator.unwrap_model(network),
                        steps=global_step,
                        epoch_no=epoch + 1,
                    )

                    remove_epoch_no = checkpoint_io.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = checkpoint_io.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                        self._remove_model(
                            args=args,
                            accelerator=accelerator,
                            old_ckpt_name=remove_ckpt_name,
                            unwrapped_nw=accelerator.unwrap_model(network),
                        )

                    if args.save_state:
                        checkpoint_io.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

            self.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)
            progress_bar.unpause()
            optimizer_train_fn()

            # end of epoch

        # metadata["ss_epoch"] = str(num_train_epochs)
        self._metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            network = accelerator.unwrap_model(network)

        accelerator.end_training()
        optimizer_eval_fn()

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            checkpoint_io.save_state_on_train_end(args, accelerator)

        if is_main_process:
            ckpt_name = checkpoint_io.get_last_ckpt_name(args, "." + args.save_model_as)
            self._save_model(
                args=args,
                accelerator=accelerator,
                save_dtype=save_dtype,
                ckpt_name=ckpt_name,
                unwrapped_nw=network,
                steps=global_step,
                epoch_no=num_train_epochs,
                force_sync_upload=True,
            )

            logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    args_util.add_anima_model_arguments(parser)
    anima_model_spec.add_model_spec_arguments(parser)
    args_util.add_dataset_arguments(parser, True, True, True)
    args_util.add_training_arguments(parser, True)
    args_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    args_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)

    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="[EXPERIMENTAL] enable offloading of tensors to CPU during checkpointing for U-Net or DiT, if supported"
        " / 勾配チェックポイント時にテンソルをCPUにオフロードする（U-NetまたはDiTのみ、サポートされている場合）",
    )
    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない"
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=None,
        nargs="*",
        help="learning rate for Text Encoder, can be multiple / Text Encoderの学習率、複数指定可能",
    )
    parser.add_argument(
        "--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み"
    )
    parser.add_argument(
        "--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール"
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    parser.add_argument(
        "--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する"
    )
    parser.add_argument(
        "--network_train_text_encoder_only",
        action="store_true",
        help="only training Text Encoder part / Text Encoder関連部分のみ学習する",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--skip_until_initial_step",
        action="store_true",
        help="skip training until initial_step is reached / initial_stepに到達するまで学習をスキップする",
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=None,
        help="initial epoch number, 1 means first epoch (same as not specifying). NOTE: initial_epoch/step doesn't affect to lr scheduler. Which means lr scheduler will start from 0 without `--resume`."
        + " / 初期エポック数、1で最初のエポック（未指定時と同じ）。注意：initial_epoch/stepはlr schedulerに影響しないため、`--resume`しない場合はlr schedulerは0から始まる",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=None,
        help="initial step number including all epochs, 0 means first step (same as not specifying). overwrites initial_epoch."
        + " / 初期ステップ数、全エポックを含むステップ数、0で最初のステップ（未指定時と同じ）。initial_epochを上書きする",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=None,
        help="Validation seed for shuffling validation dataset, training `--seed` used otherwise / 検証データセットをシャッフルするための検証シード、それ以外の場合はトレーニング `--seed` を使用する",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.0,
        help="Split for validation images out of the training dataset / 学習画像から検証画像に分割する割合",
    )
    parser.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=None,
        help="Run validation on validation dataset every N steps. By default, validation will only occur every epoch if a validation dataset is available / 検証データセットの検証をNステップごとに実行します。デフォルトでは、検証データセットが利用可能な場合にのみ、検証はエポックごとに実行されます",
    )
    parser.add_argument(
        "--validate_every_n_epochs",
        type=int,
        default=None,
        help="Run validation dataset every N epochs. By default, validation will run every epoch if a validation dataset is available / 検証データセットをNエポックごとに実行します。デフォルトでは、検証データセットが利用可能な場合、検証はエポックごとに実行されます",
    )
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=None,
        help="Max number of validation dataset items processed. By default, validation will run the entire validation dataset / 処理される検証データセット項目の最大数。デフォルトでは、検証は検証データセット全体を実行します",
    )
    return parser
