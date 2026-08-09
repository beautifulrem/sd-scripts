"""Anima-specific auxiliary training objectives.

These helpers are deliberately independent of SDXL/Flux trainers. They operate
on Anima's B,T,H,W,D block layout and rectified-flow parameterization.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _infer_grid(token_count: int, target_aspect: float) -> tuple[int, int]:
    """Choose the factor pair whose H/W is closest to the latent aspect."""
    candidates = []
    for height in range(1, int(math.sqrt(token_count)) + 1):
        if token_count % height == 0:
            width = token_count // height
            candidates.extend(((height, width), (width, height)))
    if not candidates:
        raise ValueError(f"cannot factor REPA token count {token_count}")
    return min(candidates, key=lambda hw: abs(hw[0] / hw[1] - target_aspect))


def gaussian_blur_grid(grid: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return grid
    radius = max(1, math.ceil(3.0 * sigma))
    radius = min(radius, min(grid.shape[-2:]) - 1)
    if radius <= 0:
        return grid
    coords = torch.arange(-radius, radius + 1, device=grid.device, dtype=torch.float32)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).to(grid.dtype)
    channels = grid.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    grid = F.pad(grid, (radius, radius, 0, 0), mode="reflect")
    grid = F.conv2d(grid, horizontal, groups=channels)
    grid = F.pad(grid, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(grid, vertical, groups=channels)


def dog_standardize(tokens: torch.Tensor, grid_hw: tuple[int, int], sigma_divisor: float = 16.0) -> torch.Tensor:
    """Band-pass cached visual tokens using a Difference-of-Gaussians target."""
    height, width = grid_hw
    batch, count, channels = tokens.shape
    if count != height * width:
        raise ValueError("REPA token grid does not match token count")
    grid = tokens.transpose(1, 2).reshape(batch, channels, height, width).float()
    sigma = min(height, width) / sigma_divisor
    high = grid - gaussian_blur_grid(grid, sigma)
    high = high / high.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return high.flatten(2).transpose(1, 2)


def relational_repa_loss(
    captured: torch.Tensor,
    vision_features: torch.Tensor,
    latent_hw: tuple[int, int],
    *,
    patch_size: int = 2,
    has_cls_token: bool = True,
    use_dog: bool = True,
    dog_sigma_divisor: float = 16.0,
    max_tokens: int | None = 512,
) -> torch.Tensor:
    """Relational REPA Gram alignment without a dimension-matching head."""
    batch, hidden = captured.shape[0], captured.shape[-1]
    dit_tokens = captured.reshape(batch, -1, hidden)
    latent_h, latent_w = latent_hw
    dit_h, dit_w = latent_h // patch_size, latent_w // patch_size
    if dit_tokens.shape[1] != dit_h * dit_w:
        raise ValueError(
            f"captured {dit_tokens.shape[1]} tokens but Anima latent grid is {dit_h}x{dit_w}"
        )

    target = vision_features.float()
    if has_cls_token:
        target = target[:, 1:]
    target_h, target_w = _infer_grid(target.shape[1], dit_h / dit_w)
    dit_grid = dit_tokens.reshape(batch, dit_h, dit_w, hidden).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(dit_grid.float(), (target_h, target_w)).flatten(2).transpose(1, 2)
    if use_dog:
        target = dog_standardize(target, (target_h, target_w), dog_sigma_divisor)

    if max_tokens is not None and target.shape[1] > max_tokens:
        indices = torch.linspace(0, target.shape[1] - 1, max_tokens, device=target.device).round().long()
        target = target.index_select(1, indices)
        pooled = pooled.index_select(1, indices)

    dit_norm = F.normalize(pooled, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    dit_gram = torch.bmm(dit_norm, dit_norm.transpose(1, 2))
    target_gram = torch.bmm(target_norm, target_norm.transpose(1, 2))
    return F.mse_loss(dit_gram, target_gram)


def transported_self_flow_input(
    noisy_input: torch.Tensor,
    velocity: torch.Tensor,
    sigmas: torch.Tensor,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transport x_sigma along a stop-gradient velocity to a lower sigma."""
    sigma = sigmas.reshape(-1, *([1] * (noisy_input.ndim - 1))).to(noisy_input)
    next_sigmas = (sigmas - delta).clamp_min(0.0)
    next_sigma = next_sigmas.reshape(-1, *([1] * (noisy_input.ndim - 1))).to(noisy_input)
    transported = noisy_input + (next_sigma - sigma) * velocity.detach()
    return transported, next_sigmas
