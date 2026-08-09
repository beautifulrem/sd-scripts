"""SoftREPA-style timestep- and layer-conditioned cross-attention tokens for Anima."""

from __future__ import annotations

import os
import weakref

import torch


class SoftTokenNetwork(torch.nn.Module):
    def is_mergeable(self):
        return False

    def __init__(
        self,
        unet,
        *,
        num_tokens: int = 4,
        num_timestep_bins: int = 8,
        init_std: float = 0.02,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        if num_tokens <= 0 or num_timestep_bins <= 0:
            raise ValueError("num_tokens and num_timestep_bins must be positive")
        self.num_layers = len(unet.blocks)
        self.num_tokens = int(num_tokens)
        self.num_timestep_bins = int(num_timestep_bins)
        self.context_dim = int(unet.blocks[0].cross_attn.context_dim)
        self.multiplier = float(multiplier)
        self.soft_tokens = torch.nn.Parameter(
            torch.empty(self.num_layers, self.num_timestep_bins, self.num_tokens, self.context_dim)
        )
        torch.nn.init.normal_(self.soft_tokens, std=init_std)
        self.register_buffer("_step_sigmas", torch.empty(0), persistent=False)
        self._hooks = []

    def set_step_sigmas(self, sigmas: torch.Tensor) -> None:
        self._step_sigmas = sigmas.detach().flatten()

    def clear_step_sigmas(self) -> None:
        self._step_sigmas = torch.empty(0, device=self.soft_tokens.device)

    def _tokens_for(self, layer: int, batch: int, device, dtype) -> torch.Tensor:
        if self._step_sigmas.numel() == 0:
            bins = torch.zeros(batch, dtype=torch.long, device=device)
        else:
            if self._step_sigmas.numel() not in (1, batch):
                raise RuntimeError("Soft Tokens sigma batch does not match Anima batch")
            sigma = self._step_sigmas.to(device=device).expand(batch).clamp(0, 1)
            bins = torch.clamp((sigma * self.num_timestep_bins).long(), max=self.num_timestep_bins - 1)
        return self.soft_tokens[layer, bins].to(device=device, dtype=dtype) * self.multiplier

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if not apply_unet:
            raise ValueError("Soft Tokens require Anima DiT training")
        unet._anima_adapter_network = weakref.ref(self)
        for layer, block in enumerate(unet.blocks):
            def inject(_module, args, kwargs, layer=layer):
                if len(args) < 3:
                    raise RuntimeError("unexpected Anima Block call signature for Soft Tokens")
                context = args[2]
                tokens = self._tokens_for(layer, context.shape[0], context.device, context.dtype)
                updated = list(args)
                updated[2] = torch.cat((context, tokens), dim=1)
                return tuple(updated), kwargs

            self._hooks.append(block.register_forward_pre_hook(inject, with_kwargs=True))

    def prepare_network(self, args):
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def enable_gradient_checkpointing(self):
        pass

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        lr = unet_lr if unet_lr is not None else default_lr
        return [{"params": [self.soft_tokens], "lr": lr}]

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lrs, unet_lr, default_lr):
        return self.prepare_optimizer_params(None, unet_lr, default_lr), ["soft tokens"]

    def get_trainable_params(self):
        return self.parameters()

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(file)
        else:
            state = torch.load(file, map_location="cpu")
        return self.load_state_dict(state, strict=False)

    def save_weights(self, file, dtype, metadata):
        state = {key: value.detach().cpu().to(dtype or value.dtype) for key, value in self.state_dict().items()}
        if os.path.splitext(file)[1] == ".safetensors":
            from library import anima_model_io

            anima_model_io.save_safetensors_with_hashes(state, file, metadata)
        else:
            torch.save(state, file)


def create_network(
    multiplier: float,
    network_dim: int | None,
    network_alpha: float | None,
    vae,
    text_encoders: list,
    unet,
    neuron_dropout=None,
    **kwargs,
):
    return SoftTokenNetwork(
        unet,
        num_tokens=int(kwargs.get("num_tokens", network_dim or 4)),
        num_timestep_bins=int(kwargs.get("num_timestep_bins", 8)),
        init_std=float(kwargs.get("init_std", 0.02)),
        multiplier=multiplier,
    )


def create_network_from_weights(multiplier, file, ae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
    bank = weights_sd["soft_tokens"]
    network = SoftTokenNetwork(
        unet,
        num_tokens=bank.shape[2],
        num_timestep_bins=bank.shape[1],
        multiplier=multiplier,
    )
    return network, weights_sd
