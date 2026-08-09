"""Routed multi-expert HydraLoRA for Anima.

The full checkpoint retains routers and experts. ``export_mode=standard_mean``
produces an explicitly lossy, router-averaged standard LoRA for ordinary loaders.
"""

from __future__ import annotations

import ast
import functools

import torch

from networks import lora_anima


class HydraLoRAModule(lora_anima.LoRAModule):
    def __init__(self, *args, num_experts=4, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.lora_down, torch.nn.Linear):
            raise TypeError("HydraLoRA currently supports Linear targets only")
        self.num_experts = int(num_experts)
        if self.num_experts < 2:
            raise ValueError("HydraLoRA requires at least two experts")
        self.expert_ups = torch.nn.ModuleList(
            [torch.nn.Linear(self.lora_dim, self.lora_up.out_features, bias=False) for _ in range(self.num_experts)]
        )
        # Break expert symmetry while keeping the uniform routed residual at
        # exactly zero. This gives the router a useful gradient on step one
        # without changing the adapter's zero-delta initialization.
        expert_weights = torch.empty(
            self.num_experts,
            *self.expert_ups[0].weight.shape,
            device=self.expert_ups[0].weight.device,
            dtype=self.expert_ups[0].weight.dtype,
        )
        torch.nn.init.normal_(expert_weights, std=1e-3)
        expert_weights.sub_(expert_weights.mean(dim=0, keepdim=True))
        with torch.no_grad():
            for expert, weight in zip(self.expert_ups, expert_weights):
                expert.weight.copy_(weight)
        self.router = torch.nn.Linear(self.lora_down.in_features, self.num_experts, bias=True)
        torch.nn.init.zeros_(self.router.weight)
        torch.nn.init.zeros_(self.router.bias)
        self._last_router_probs = None

    def forward_delta(self, x):
        shared = super().forward_delta(x)
        x_lora = x * self.inv_scale.to(device=x.device, dtype=x.dtype) if self._has_channel_scale else x
        rank = self._apply_timestep_mask(self.lora_down(x_lora))
        pooled = x if x.ndim == 2 else x.mean(dim=tuple(range(1, x.ndim - 1)))
        probs = torch.softmax(self.router(pooled.float()).to(x.dtype), dim=-1)
        self._last_router_probs = probs
        expert_values = torch.stack([expert(rank) for expert in self.expert_ups], dim=1)
        gate_shape = (x.shape[0], self.num_experts) + (1,) * (expert_values.ndim - 2)
        routed = (expert_values * probs.reshape(gate_shape)).sum(dim=1)
        return shared + routed * self.multiplier * self.scale

    def auxiliary_loss(self, balance_weight: float, orthogonal_weight: float) -> torch.Tensor:
        loss = self.lora_down.weight.new_zeros(())
        if balance_weight > 0 and self._last_router_probs is not None:
            usage = self._last_router_probs.float().mean(dim=0)
            loss = loss + balance_weight * self.num_experts * ((usage - 1.0 / self.num_experts) ** 2).sum()
        if orthogonal_weight > 0:
            flat = torch.stack([expert.weight.flatten() for expert in self.expert_ups]).float()
            flat = torch.nn.functional.normalize(flat, dim=1)
            gram = flat @ flat.T
            eye = torch.eye(self.num_experts, device=gram.device)
            loss = loss + orthogonal_weight * ((gram - eye) ** 2).mean()
        return loss


class HydraLoRANetwork(lora_anima.LoRANetwork):
    def is_mergeable(self):
        return False

    def __init__(
        self,
        text_encoders,
        unet,
        *,
        num_experts=4,
        balance_weight=0.01,
        orthogonal_weight=0.01,
        export_mode="full",
        **kwargs,
    ):
        self.num_experts = int(num_experts)
        self.balance_weight = float(balance_weight)
        self.orthogonal_weight = float(orthogonal_weight)
        self.export_mode = str(export_mode)
        module_factory = functools.partial(HydraLoRAModule, num_experts=self.num_experts)
        super().__init__(text_encoders, unet, module_class=module_factory, **kwargs)

    def get_auxiliary_loss(self):
        losses = [
            module.auxiliary_loss(self.balance_weight, self.orthogonal_weight)
            for module in self.unet_loras
            if isinstance(module, HydraLoRAModule)
        ]
        return torch.stack(losses).sum() if losses else self.soft_zero()

    def soft_zero(self):
        return next(self.parameters()).new_zeros(())

    def save_weights(self, file, dtype, metadata):
        if self.export_mode == "full":
            return super().save_weights(file, dtype, metadata)
        if self.export_mode != "standard_mean":
            raise ValueError("Hydra export_mode must be 'full' or 'standard_mean'")
        state = {key: value.detach().clone() for key, value in self.state_dict().items()}
        for name, module in self.named_children():
            if not isinstance(module, HydraLoRAModule):
                continue
            up_key = f"{name}.lora_up.weight"
            mean_expert = torch.stack([expert.weight.detach() for expert in module.expert_ups]).mean(dim=0)
            state[up_key] = state[up_key] + mean_expert.to(state[up_key])
            for key in list(state):
                if key.startswith((f"{name}.expert_ups.", f"{name}.router.")):
                    state.pop(key)
        # Reuse the standard writer without mutating live parameters.
        state = {key: value.cpu().to(dtype or value.dtype) for key, value in state.items()}
        if file.endswith(".safetensors"):
            from library import anima_model_io

            md = dict(metadata or {})
            md["ss_hydra_export"] = "standard_mean_lossy"
            anima_model_io.save_safetensors_with_hashes(state, file, md)
        else:
            torch.save(state, file)


def _as_bool(value) -> bool:
    return str(value).lower() in ("1", "true", "yes", "on")


def create_network(multiplier, network_dim, network_alpha, vae, text_encoders, unet, neuron_dropout=None, **kwargs):
    network_dim = int(network_dim or 4)
    network_alpha = float(network_alpha or 1.0)
    exclude = kwargs.get("exclude_patterns")
    exclude = ast.literal_eval(exclude) if exclude else []
    if not isinstance(exclude, list):
        exclude = [exclude]
    exclude.append(r".*(_modulation|_norm|_embedder|final_layer).*")
    include = kwargs.get("include_patterns")
    include = ast.literal_eval(include) if include else None
    if include is not None and not isinstance(include, list):
        include = [include]
    train_adaln = _as_bool(kwargs.get("train_adaln", False))
    reg_dims = None
    reg_alphas = None
    reg_lrs = None
    if train_adaln:
        pattern = r".*adaln_modulation_.*"
        include = list(include or []) + [pattern]
        reg_dims = {pattern: int(kwargs.get("adaln_rank", min(16, network_dim)))}
        if kwargs.get("adaln_alpha") is not None:
            reg_alphas = {pattern: float(kwargs["adaln_alpha"])}
        if kwargs.get("adaln_lr") is not None:
            reg_lrs = {pattern: float(kwargs["adaln_lr"])}
    return HydraLoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=float(kwargs["rank_dropout"]) if kwargs.get("rank_dropout") is not None else None,
        module_dropout=float(kwargs["module_dropout"]) if kwargs.get("module_dropout") is not None else None,
        exclude_patterns=exclude,
        include_patterns=include,
        reg_dims=reg_dims,
        reg_alphas=reg_alphas,
        reg_lrs=reg_lrs,
        down_init=kwargs.get("down_init", "kaiming"),
        use_timestep_mask=_as_bool(kwargs.get("use_timestep_mask", False)),
        min_rank=int(kwargs.get("min_rank", 1)),
        alpha_rank_scale=float(kwargs.get("alpha_rank_scale", 1.0)),
        num_experts=int(kwargs.get("num_experts", 4)),
        balance_weight=float(kwargs.get("balance_weight", 0.01)),
        orthogonal_weight=float(kwargs.get("orthogonal_weight", 0.01)),
        export_mode=kwargs.get("export_mode", "full"),
    )


def create_network_from_weights(multiplier, file, ae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if file.endswith(".safetensors"):
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
    expert_indices = [int(key.split(".expert_ups.")[1].split(".")[0]) for key in weights_sd if ".expert_ups." in key]
    if not expert_indices:
        raise ValueError("checkpoint has no Hydra expert weights; use networks.lora_anima for standard_mean exports")
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        prefix = key.split(".")[0]
        if key.endswith(".lora_down.weight"):
            modules_dim[prefix] = value.shape[0]
        elif key.endswith(".alpha"):
            modules_alpha[prefix] = value
    network = HydraLoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        num_experts=max(expert_indices) + 1,
        balance_weight=float(kwargs.get("balance_weight", 0.01)),
        orthogonal_weight=float(kwargs.get("orthogonal_weight", 0.01)),
        export_mode=kwargs.get("export_mode", "full"),
        **lora_anima.get_resume_network_kwargs(kwargs, weights_sd),
    )
    return network, weights_sd
