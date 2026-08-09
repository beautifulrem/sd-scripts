"""Small Euler flow schedule used by the standalone Anima inference script."""

from typing import Tuple

import torch


def get_timesteps_sigmas(sampling_steps: int, shift: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    sigmas = torch.linspace(1, 0, sampling_steps + 1)
    sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)
    sigmas = sigmas.to(torch.float32)
    timesteps = (sigmas[:-1] * 1000).to(dtype=torch.float32, device=device)
    return timesteps, sigmas


def step(latents: torch.Tensor, noise_pred: torch.Tensor, sigmas: torch.Tensor, step_index: int) -> torch.Tensor:
    return latents.float() - (sigmas[step_index] - sigmas[step_index + 1]) * noise_pred.float()
