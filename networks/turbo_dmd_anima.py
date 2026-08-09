"""Dual-branch LoRA used by Anima DP-DMD turbo distillation."""

from __future__ import annotations

import functools
import os

import torch

from networks import lora_anima


class TurboDMDLoRAModule(lora_anima.LoRAModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.lora_down, torch.nn.Linear):
            raise TypeError("Turbo DMD supports Linear Anima targets only")
        self.critic_down = torch.nn.Linear(self.lora_down.in_features, self.lora_dim, bias=False)
        self.critic_up = torch.nn.Linear(self.lora_dim, self.lora_up.out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.critic_down.weight, a=5**0.5)
        torch.nn.init.zeros_(self.critic_up.weight)
        self.adapter_mode = "generator"

    def forward(self, x):
        base = self.org_forward(x)
        if self.adapter_mode == "base" or not self.enabled:
            return base
        return base + self.forward_delta(x)

    def forward_delta(self, x):
        if self.adapter_mode == "generator":
            return super().forward_delta(x)
        if self.adapter_mode != "critic":
            raise RuntimeError(f"unknown Turbo DMD adapter mode {self.adapter_mode!r}")
        rank = self._apply_timestep_mask(self.critic_down(x))
        return self.critic_up(rank) * self.multiplier * self.scale

    def capture_recompute_context(self):
        return super().capture_recompute_context(), self.adapter_mode

    def restore_recompute_context(self, state) -> None:
        if state is None:
            super().restore_recompute_context(None)
            self.adapter_mode = "generator"
            return
        base_state, self.adapter_mode = state
        super().restore_recompute_context(base_state)


class TurboDMDNetwork(lora_anima.LoRANetwork):
    def __init__(self, text_encoders, unet, **kwargs):
        super().__init__(
            text_encoders,
            unet,
            module_class=functools.partial(TurboDMDLoRAModule),
            **kwargs,
        )

    def set_adapter_mode(self, mode: str) -> None:
        if mode not in ("generator", "critic", "base"):
            raise ValueError("Turbo DMD mode must be generator, critic, or base")
        for module in self.text_encoder_loras + self.unet_loras:
            module.adapter_mode = mode

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lr, unet_lr, default_lr):
        groups, descriptions = super().prepare_optimizer_params_with_multiple_te_lrs(
            text_encoder_lr, unet_lr, default_lr
        )
        return groups, [f"turbo {description}" for description in descriptions]

    def save_weights(self, file, dtype, metadata):
        # Primary artifact: ordinary generator LoRA, fully compatible with the
        # standard Anima inference ecosystem.
        full = self.state_dict()
        critic = {key: value for key, value in full.items() if ".critic_" in key}
        generator = {key: value for key, value in full.items() if ".critic_" not in key}
        generator = {key: value.detach().cpu().to(dtype or value.dtype) for key, value in generator.items()}
        critic = {key: value.detach().cpu().to(dtype or value.dtype) for key, value in critic.items()}
        if file.endswith(".safetensors"):
            from library import anima_model_io

            generator_metadata = dict(metadata or {})
            anima_model_io.save_safetensors_with_hashes(generator, file, generator_metadata)
            stem, ext = os.path.splitext(file)
            anima_model_io.save_safetensors_with_hashes(
                critic,
                f"{stem}_dmd_critic{ext}",
                {"ss_adapter": "anima_dp_dmd_critic"},
            )
        else:
            torch.save(generator, file)
            stem, ext = os.path.splitext(file)
            torch.save(critic, f"{stem}_dmd_critic{ext or '.pt'}")

    @staticmethod
    def get_checkpoint_sidecar_paths(file):
        stem, ext = os.path.splitext(file)
        return [f"{stem}_dmd_critic{ext or '.pt'}"]

    def load_weights(self, file):
        if file.endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(file)
            stem, ext = os.path.splitext(file)
            critic_path = f"{stem}_dmd_critic{ext}"
            if not os.path.isfile(critic_path):
                raise FileNotFoundError(f"DP-DMD critic sidecar missing: {critic_path}")
            state.update(load_file(critic_path))
        else:
            state = torch.load(file, map_location="cpu")
            stem, ext = os.path.splitext(file)
            critic_path = f"{stem}_dmd_critic{ext or '.pt'}"
            if not os.path.isfile(critic_path):
                raise FileNotFoundError(f"DP-DMD critic sidecar missing: {critic_path}")
            state.update(torch.load(critic_path, map_location="cpu"))
        return self.load_state_dict(state, strict=False)


def _as_bool(value):
    return str(value).lower() in ("1", "true", "yes", "on")


def create_network(multiplier, network_dim, network_alpha, vae, text_encoders, unet, neuron_dropout=None, **kwargs):
    exclude = [r".*(_modulation|_norm|_embedder|final_layer).*"]
    return TurboDMDNetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=int(network_dim or 32),
        alpha=float(network_alpha or network_dim or 32),
        dropout=neuron_dropout,
        exclude_patterns=exclude,
        down_init=kwargs.get("down_init", "weight_svd"),
        use_timestep_mask=_as_bool(kwargs.get("use_timestep_mask", False)),
        min_rank=int(kwargs.get("min_rank", 1)),
        alpha_rank_scale=float(kwargs.get("alpha_rank_scale", 1.0)),
    )


def create_network_from_weights(multiplier, file, ae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    if for_inference:
        raise ValueError(
            "Turbo generator checkpoints are standard LoRA files; load or merge them with networks.lora_anima"
        )
    if weights_sd is None:
        if file.endswith(".safetensors"):
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
    stem, ext = os.path.splitext(file)
    critic_path = f"{stem}_dmd_critic{ext or '.pt'}"
    if not os.path.isfile(critic_path):
        raise FileNotFoundError(f"DP-DMD critic sidecar missing: {critic_path}")
    if file.endswith(".safetensors"):
        from safetensors.torch import load_file

        weights_sd.update(load_file(critic_path))
    else:
        weights_sd.update(torch.load(critic_path, map_location="cpu"))

    dims, alphas = {}, {}
    for key, value in weights_sd.items():
        prefix = key.split(".")[0]
        if key.endswith(".lora_down.weight"):
            dims[prefix] = value.shape[0]
        elif key.endswith(".alpha"):
            alphas[prefix] = value
    network = TurboDMDNetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=dims,
        modules_alpha=alphas,
    )
    return network, weights_sd
