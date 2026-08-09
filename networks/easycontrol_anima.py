"""EasyControl-style extended self-attention conditioning for Anima."""

from __future__ import annotations

import contextlib
import weakref

import torch

from library import anima_models, attention


class EasyControlNetwork(torch.nn.Module):
    def __init__(self, unet, *, rank=16, condition_channels=16, multiplier=1.0):
        super().__init__()
        self.model_dim = int(unet.model_channels)
        self.rank = int(rank)
        self.multiplier = float(multiplier)
        self.condition_patch = torch.nn.Conv2d(
            int(condition_channels), self.model_dim, kernel_size=unet.patch_spatial, stride=unet.patch_spatial
        )
        self.cond_down = torch.nn.ModuleList(
            [torch.nn.Linear(self.model_dim, self.rank, bias=False) for _ in unet.blocks]
        )
        self.cond_up = torch.nn.ModuleList(
            [torch.nn.Linear(self.rank, self.model_dim, bias=False) for _ in unet.blocks]
        )
        for down, up in zip(self.cond_down, self.cond_up):
            torch.nn.init.kaiming_uniform_(down.weight, a=5**0.5)
            torch.nn.init.zeros_(up.weight)
        self.gates = torch.nn.Parameter(torch.zeros(len(unet.blocks)))
        self._condition_latents = None
        self._original_forwards = []

    def set_condition_latents(self, latents: torch.Tensor) -> None:
        self._condition_latents = latents

    def clear_condition_latents(self) -> None:
        self._condition_latents = None

    def capture_recompute_context(self):
        return self._condition_latents

    @contextlib.contextmanager
    def use_recompute_context(self, condition_latents):
        previous = self._condition_latents
        self._condition_latents = condition_latents
        try:
            yield
        finally:
            self._condition_latents = previous

    def _condition_tokens(self, block_index: int, x: torch.Tensor) -> torch.Tensor:
        if self._condition_latents is None:
            return torch.zeros_like(x)
        latents = self._condition_latents
        if latents.ndim == 5:
            latents = latents.squeeze(2)
        grid = self.condition_patch(latents.to(device=x.device, dtype=x.dtype))
        tokens = grid.flatten(2).transpose(1, 2)
        if tokens.shape[1] != x.shape[1]:
            side = int(tokens.shape[1] ** 0.5)
            target_side = int(x.shape[1] ** 0.5)
            if side * side != tokens.shape[1] or target_side * target_side != x.shape[1]:
                raise RuntimeError("EasyControl condition and target token grids do not match")
            grid = grid.reshape(grid.shape[0], grid.shape[1], side, side)
            grid = torch.nn.functional.adaptive_avg_pool2d(grid, (target_side, target_side))
            tokens = grid.flatten(2).transpose(1, 2)
        return tokens + self.cond_up[block_index](self.cond_down[block_index](tokens))

    @staticmethod
    def _extended_attention(attn, x, cond, attn_params, rope_emb):
        batch, seq, _ = x.shape
        context = torch.cat((x, cond), dim=1)
        q = attn.q_proj(x).view(batch, seq, attn.n_heads, attn.head_dim)
        k = attn.k_proj(context).view(batch, context.shape[1], attn.n_heads, attn.head_dim)
        v = attn.v_proj(context).view(batch, context.shape[1], attn.n_heads, attn.head_dim)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        v = attn.v_norm(v)
        if rope_emb is not None:
            q = anima_models.apply_rotary_pos_emb(q, rope_emb, tensor_format="bshd")
            k_target, k_cond = k[:, :seq], k[:, seq:]
            k = torch.cat(
                (
                    anima_models.apply_rotary_pos_emb(k_target, rope_emb, tensor_format="bshd"),
                    anima_models.apply_rotary_pos_emb(k_cond, rope_emb, tensor_format="bshd"),
                ),
                dim=1,
            )
        result = attention.attention([q, k, v], attn_params=attn_params)
        return attn.output_dropout(attn.output_proj(result))

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if not apply_unet:
            raise ValueError("EasyControl requires Anima DiT training")
        provider_ref = weakref.ref(self)
        unet._anima_adapter_network = provider_ref
        for block_index, block in enumerate(unet.blocks):
            block._anima_adapter_context_provider = provider_ref
            attn = block.self_attn
            original = attn.forward
            self._original_forwards.append(original)

            def extended(x, attn_params, context=None, rope_emb=None, block_index=block_index, original=original, attn=attn):
                base = original(x, attn_params, context=context, rope_emb=rope_emb)
                if self._condition_latents is None:
                    return base
                cond = self._condition_tokens(block_index, x)
                controlled = self._extended_attention(attn, x, cond, attn_params, rope_emb)
                gate = torch.tanh(self.gates[block_index]) * self.multiplier
                return base + gate * (controlled - base)

            attn.forward = extended

    def prepare_network(self, args):
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def enable_gradient_checkpointing(self):
        pass

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lr, unet_lr, default_lr):
        lr = unet_lr if unet_lr is not None else default_lr
        return [{"params": self.parameters(), "lr": lr}], ["easycontrol"]

    def get_trainable_params(self):
        return self.parameters()

    def load_weights(self, file):
        if file.endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(file)
        else:
            state = torch.load(file, map_location="cpu")
        return self.load_state_dict(state, strict=False)

    def save_weights(self, file, dtype, metadata):
        state = {key: value.detach().cpu().to(dtype or value.dtype) for key, value in self.state_dict().items()}
        if file.endswith(".safetensors"):
            from library import anima_model_io

            anima_model_io.save_safetensors_with_hashes(state, file, metadata)
        else:
            torch.save(state, file)


def create_network(multiplier, network_dim, network_alpha, vae, text_encoders, unet, neuron_dropout=None, **kwargs):
    return EasyControlNetwork(
        unet,
        rank=int(network_dim or kwargs.get("rank", 16)),
        condition_channels=int(kwargs.get("condition_channels", 16)),
        multiplier=multiplier,
    )


def create_network_from_weights(multiplier, file, ae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if file.endswith(".safetensors"):
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
    rank = weights_sd["cond_down.0.weight"].shape[0]
    channels = weights_sd["condition_patch.weight"].shape[1]
    network = EasyControlNetwork(unet, rank=rank, condition_channels=channels, multiplier=multiplier)
    return network, weights_sd
