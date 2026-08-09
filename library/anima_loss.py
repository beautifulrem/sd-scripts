"""Loss helpers used by Anima full, LoRA and ControlNet-LLLite training."""

import math
from typing import Optional

import torch


def apply_masked_loss(loss: torch.Tensor, batch: dict) -> torch.Tensor:
    if "conditioning_images" in batch:
        mask = batch["conditioning_images"].to(dtype=loss.dtype)[:, 0].unsqueeze(1)
        mask = mask / 2 + 0.5
    elif batch.get("alpha_masks") is not None:
        mask = batch["alpha_masks"].to(dtype=loss.dtype).unsqueeze(1)
    else:
        return loss
    mask = torch.nn.functional.interpolate(mask, size=loss.shape[2:], mode="area")
    return loss * mask


def get_huber_threshold_if_needed(args, timesteps: torch.Tensor, noise_scheduler) -> Optional[torch.Tensor]:
    if args.loss_type not in ("huber", "smooth_l1"):
        return None

    batch_size = timesteps.shape[0]
    if args.huber_schedule == "exponential":
        num_timesteps = noise_scheduler.config.num_train_timesteps
        alpha = -math.log(args.huber_c) / num_timesteps
        return torch.exp(-alpha * timesteps) * args.huber_scale
    if args.huber_schedule == "constant":
        return torch.full((batch_size,), args.huber_c * args.huber_scale, device=timesteps.device)
    raise ValueError("Anima supports only 'constant' and 'exponential' Huber schedules")


def conditional_loss(
    model_prediction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    reduction: str,
    huber_c: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if loss_type == "l2":
        return torch.nn.functional.mse_loss(model_prediction, target, reduction=reduction)
    if loss_type == "l1":
        return torch.nn.functional.l1_loss(model_prediction, target, reduction=reduction)
    if loss_type not in ("huber", "smooth_l1"):
        raise ValueError(f"Unsupported Anima loss type: {loss_type}")
    if huber_c is None:
        raise ValueError(f"{loss_type} requires a Huber threshold")

    huber_c = huber_c.view(-1, *([1] * (model_prediction.ndim - 1)))
    difference = model_prediction - target
    if loss_type == "huber":
        loss = 2 * huber_c * (torch.sqrt(difference**2 + huber_c**2) - huber_c)
    else:
        loss = 2 * (torch.sqrt(difference**2 + huber_c**2) - huber_c)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
