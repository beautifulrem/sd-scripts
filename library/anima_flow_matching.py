"""Flow-matching timestep utilities used by Anima training.

This module intentionally contains only the scheduler state and sampling logic
that the Anima trainers consume.  It replaces the previous dependency on the
much larger FLUX and SD3 training utility modules.
"""

import logging
import math
from types import SimpleNamespace
from typing import Callable, Optional, Tuple

import torch


logger = logging.getLogger(__name__)


class AnimaFlowMatchScheduler:
    """Minimal training-time flow-matching schedule for Anima.

    The trainers only require ``config.num_train_timesteps``, ``timesteps`` and
    ``sigmas``.  Inference uses :mod:`library.anima_inference_scheduler`.
    """

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0):
        self.config = SimpleNamespace(num_train_timesteps=num_train_timesteps, shift=shift)
        timesteps = torch.arange(num_train_timesteps, 0, -1, dtype=torch.float32)
        sigmas = timesteps / num_train_timesteps
        self.sigmas = (shift * sigmas / (1 + (shift - 1) * sigmas)).cpu()
        self.timesteps = self.sigmas * num_train_timesteps

    def __len__(self) -> int:
        return self.config.num_train_timesteps


def _time_shift(mu: float, sigma: float, timestep: torch.Tensor) -> torch.Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / timestep - 1) ** sigma)


def _linear_function(
    x1: float = 256,
    y1: float = 0.5,
    x2: float = 4096,
    y2: float = 1.15,
) -> Callable[[float], float]:
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return lambda value: slope * value + intercept


def _get_sigmas(noise_scheduler, timesteps, device, dtype=torch.float32) -> torch.Tensor:
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]
    return sigmas[step_indices].flatten()


def _compute_timestep_density(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: Optional[float] = None,
    logit_std: Optional[float] = None,
    mode_scale: Optional[float] = None,
) -> torch.Tensor:
    if weighting_scheme == "logit_normal":
        return torch.sigmoid(torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu"))
    if weighting_scheme == "mode":
        values = torch.rand(size=(batch_size,), device="cpu")
        return 1 - values - mode_scale * (torch.cos(math.pi * values / 2) ** 2 - 1 + values)
    return torch.rand(size=(batch_size,), device="cpu")


def get_noisy_model_input_and_timesteps(
    args,
    noise_scheduler,
    latents: torch.Tensor,
    noise: torch.Tensor,
    device,
    dtype,
    timestep_sampling_offset: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample Anima flow timesteps and mix latent input with noise."""

    batch_size, height, width = latents.shape[0], latents.shape[-2], latents.shape[-1]
    assert batch_size > 0, "Batch size not large enough"
    num_timesteps = noise_scheduler.config.num_train_timesteps

    if args.timestep_sampling in ("uniform", "sigmoid"):
        if args.timestep_sampling == "sigmoid":
            values = torch.randn((batch_size,), device=device)
            if timestep_sampling_offset is not None:
                values = values + timestep_sampling_offset.to(device=device, dtype=values.dtype)
            sigmas = torch.sigmoid(args.sigmoid_scale * values)
        else:
            sigmas = torch.rand((batch_size,), device=device)
        timesteps = sigmas * num_timesteps
    elif args.timestep_sampling == "shift":
        sigmas = torch.randn(batch_size, device=device)
        if timestep_sampling_offset is not None:
            sigmas = sigmas + timestep_sampling_offset.to(device=device, dtype=sigmas.dtype)
        sigmas = torch.sigmoid(sigmas * args.sigmoid_scale)
        shift = args.discrete_flow_shift
        sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_timesteps
    elif args.timestep_sampling == "flux_shift":
        sigmas = torch.randn(batch_size, device=device)
        if timestep_sampling_offset is not None:
            sigmas = sigmas + timestep_sampling_offset.to(device=device, dtype=sigmas.dtype)
        sigmas = torch.sigmoid(sigmas * args.sigmoid_scale)
        # Anima uses the same resolution-aware shift formula for this sampling mode.
        mu = _linear_function(y1=0.5, y2=1.15)((height // 2) * (width // 2))
        sigmas = _time_shift(mu, 1.0, sigmas)
        timesteps = sigmas * num_timesteps
    else:
        density = _compute_timestep_density(
            args.weighting_scheme,
            batch_size,
            args.logit_mean,
            args.logit_std,
            args.mode_scale,
        )
        indices = (density * num_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=device)
        sigmas = _get_sigmas(noise_scheduler, timesteps, device, dtype=dtype)

    broadcast_shape = (-1,) + (1,) * (latents.ndim - 1)
    sigmas = sigmas.view(broadcast_shape)
    if args.ip_noise_gamma:
        input_perturbation = torch.randn_like(latents, device=latents.device, dtype=dtype)
        if args.ip_noise_gamma_random_strength:
            strength = torch.rand(1, device=latents.device, dtype=dtype) * args.ip_noise_gamma
        else:
            strength = args.ip_noise_gamma
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * (noise + strength * input_perturbation)
    else:
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

    return noisy_model_input.to(dtype), timesteps.to(dtype), sigmas


_SHIFT_AWARE_SAMPLING = ("sigma", "shift")
_OFFSET_AWARE_SAMPLING = ("sigmoid", "shift", "flux_shift")


def get_show_timesteps_offset(args) -> Tuple[Optional[float], str]:
    offset = getattr(args, "show_timesteps_offset", 0.0) or 0.0
    if offset == 0.0:
        return None, ""
    if args.timestep_sampling in _OFFSET_AWARE_SAMPLING:
        return offset, f", timestep_sampling_offset={offset}"
    return None, (
        f", timestep_sampling_offset={offset} (IGNORED for timestep_sampling='{args.timestep_sampling}'; "
        "only 'sigmoid', 'shift' and 'flux_shift' use it)"
    )


def get_timestep_sampling_info(args) -> str:
    sampling = args.timestep_sampling
    parts = [f"timestep_sampling={sampling}"]
    if sampling in _SHIFT_AWARE_SAMPLING:
        parts.append(f"discrete_flow_shift={args.discrete_flow_shift} (applied)")
    else:
        parts.append(
            f"discrete_flow_shift={args.discrete_flow_shift} (IGNORED for timestep_sampling='{sampling}'; "
            "only 'sigma' and 'shift' use it)"
        )
    if sampling in ("sigmoid", "shift", "flux_shift"):
        parts.append(f"sigmoid_scale={args.sigmoid_scale}")
    if sampling == "sigma":
        parts.append(f"weighting_scheme={args.weighting_scheme}")
    return ", ".join(parts)


def log_timestep_sampling_info(args) -> None:
    logger.info("Timestep sampling: %s", get_timestep_sampling_info(args))


def parse_show_timesteps_latent_size(args, vae_compression: int = 8) -> Tuple[int, int]:
    resolution = getattr(args, "show_timesteps_resolution", None) or "1024"
    values = [int(value.strip()) for value in str(resolution).split(",") if value.strip()]
    if len(values) == 1:
        height = width = values[0]
    elif len(values) == 2:
        height, width = values
    else:
        raise ValueError(f"--show_timesteps_resolution must be 'H' or 'H,W', got: {resolution}")
    return height // vae_compression, width // vae_compression
