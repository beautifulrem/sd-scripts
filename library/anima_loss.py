"""Loss helpers used by Anima full, LoRA and ControlNet-LLLite training."""

import math
from typing import Optional

import torch


def apply_masked_loss(
    loss: torch.Tensor,
    batch: dict,
    *,
    conditioning_image_is_mask: bool = True,
) -> torch.Tensor:
    if batch.get("alpha_masks") is not None:
        mask = batch["alpha_masks"].to(dtype=loss.dtype).unsqueeze(1)
    elif conditioning_image_is_mask and "conditioning_images" in batch:
        mask = batch["conditioning_images"].to(dtype=loss.dtype)[:, 0].unsqueeze(1)
        mask = mask / 2 + 0.5
    else:
        return loss
    mask = torch.nn.functional.interpolate(mask, size=loss.shape[2:], mode="area")
    return loss * mask


def get_huber_threshold_if_needed(args, timesteps: torch.Tensor, noise_scheduler) -> Optional[torch.Tensor]:
    if args.loss_type not in ("huber", "smooth_l1"):
        return None

    batch_size = timesteps.shape[0]
    if args.huber_schedule == "exponential":
        # Training passes scheduler timesteps in [0, num_train_timesteps].
        # Interpolate from huber_scale at t=0 to huber_c*huber_scale at t=N.
        normalized_timesteps = timesteps / noise_scheduler.config.num_train_timesteps
        return torch.exp(math.log(args.huber_c) * normalized_timesteps) * args.huber_scale
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


def reduce_weighted_loss(
    elementwise_loss: torch.Tensor,
    timestep_weighting: Optional[torch.Tensor],
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    """Return one loss per sample without cross-batch broadcasting.

    Timestep weights have latent-shaped singleton dimensions, whereas dataset
    weights are one-dimensional.  Applying both only after spatial reduction
    can accidentally form a B-by-B outer product.
    """

    if timestep_weighting is not None:
        elementwise_loss = elementwise_loss * timestep_weighting
    per_sample = elementwise_loss.mean(dim=tuple(range(1, elementwise_loss.ndim)))
    return per_sample * sample_weights.to(device=per_sample.device, dtype=per_sample.dtype)
