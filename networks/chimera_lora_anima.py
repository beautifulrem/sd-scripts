"""ChimeraHydra: additive content and frequency expert pools for Anima."""

from __future__ import annotations

import ast
import functools

import torch

from networks import hydra_lora_anima, lora_anima


class ChimeraLoRAModule(hydra_lora_anima.HydraLoRAModule):
    def __init__(self, *args, num_frequency_experts=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_frequency_experts = int(num_frequency_experts)
        if self.num_frequency_experts < 2:
            raise ValueError("Chimera requires at least two frequency experts")
        self.frequency_down = torch.nn.Linear(self.lora_down.in_features, self.lora_dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.frequency_down.weight, a=5**0.5)
        self.frequency_ups = torch.nn.ModuleList(
            [torch.nn.Linear(self.lora_dim, self.lora_up.out_features, bias=False) for _ in range(self.num_frequency_experts)]
        )
        frequency_weights = torch.empty(
            self.num_frequency_experts,
            *self.frequency_ups[0].weight.shape,
            device=self.frequency_ups[0].weight.device,
            dtype=self.frequency_ups[0].weight.dtype,
        )
        torch.nn.init.normal_(frequency_weights, std=1e-3)
        frequency_weights.sub_(frequency_weights.mean(dim=0, keepdim=True))
        with torch.no_grad():
            for expert, weight in zip(self.frequency_ups, frequency_weights):
                expert.weight.copy_(weight)
        # Router inputs: sigma and normalized high-frequency energy index.
        self.frequency_router = torch.nn.Linear(2, self.num_frequency_experts, bias=True)
        torch.nn.init.zeros_(self.frequency_router.weight)
        torch.nn.init.zeros_(self.frequency_router.bias)
        self.register_buffer("_chimera_sigma", torch.empty(0), persistent=False)
        self.register_buffer("_chimera_fei", torch.empty(0), persistent=False)
        self._last_frequency_probs = None

    def set_frequency_context(self, sigmas: torch.Tensor, fei: torch.Tensor) -> None:
        self._chimera_sigma = sigmas.detach().flatten()
        self._chimera_fei = fei.detach().flatten()

    def clear_frequency_context(self) -> None:
        self._chimera_sigma = torch.empty(0, device=self.lora_down.weight.device)
        self._chimera_fei = torch.empty(0, device=self.lora_down.weight.device)

    def capture_recompute_context(self):
        base_state = super().capture_recompute_context()
        if base_state is None and self._chimera_sigma.numel() == 0:
            return None
        return base_state, self._chimera_sigma, self._chimera_fei

    def restore_recompute_context(self, state) -> None:
        if state is None:
            super().restore_recompute_context(None)
            self.clear_frequency_context()
            return
        base_state, self._chimera_sigma, self._chimera_fei = state
        super().restore_recompute_context(base_state)

    def forward_delta(self, x):
        content = super().forward_delta(x)
        batch = x.shape[0]
        if self._chimera_sigma.numel() == 0:
            sigma = torch.zeros(batch, device=x.device)
            fei = torch.zeros(batch, device=x.device)
        else:
            sigma = self._chimera_sigma.to(x.device).expand(batch)
            fei = self._chimera_fei.to(x.device).expand(batch)
        route_features = torch.stack((sigma, fei), dim=-1)
        probs = torch.softmax(self.frequency_router(route_features.float()).to(x.dtype), dim=-1)
        self._last_frequency_probs = probs
        rank = self._apply_timestep_mask(self.frequency_down(x))
        values = torch.stack([expert(rank) for expert in self.frequency_ups], dim=1)
        gate_shape = (batch, self.num_frequency_experts) + (1,) * (values.ndim - 2)
        frequency = (values * probs.reshape(gate_shape)).sum(dim=1)
        return content + frequency * self.multiplier * self.scale

    def auxiliary_loss(self, balance_weight: float, orthogonal_weight: float) -> torch.Tensor:
        loss = super().auxiliary_loss(balance_weight, orthogonal_weight)
        if balance_weight > 0 and self._last_frequency_probs is not None:
            usage = self._last_frequency_probs.float().mean(dim=0)
            loss = loss + balance_weight * self.num_frequency_experts * (
                (usage - 1.0 / self.num_frequency_experts) ** 2
            ).sum()
        if orthogonal_weight > 0:
            content_basis = torch.nn.functional.normalize(self.lora_down.weight.float(), dim=1)
            frequency_basis = torch.nn.functional.normalize(self.frequency_down.weight.float(), dim=1)
            loss = loss + orthogonal_weight * (content_basis @ frequency_basis.T).pow(2).mean()
        return loss


class ChimeraLoRANetwork(hydra_lora_anima.HydraLoRANetwork):
    def is_mergeable(self):
        return False

    def __init__(self, text_encoders, unet, *, num_frequency_experts=2, **kwargs):
        self.num_frequency_experts = int(num_frequency_experts)
        num_experts = int(kwargs.pop("num_experts", 4))
        balance = float(kwargs.pop("balance_weight", 0.01))
        orthogonal = float(kwargs.pop("orthogonal_weight", 0.01))
        export_mode = str(kwargs.pop("export_mode", "full"))
        # Bypass HydraLoRANetwork's factory so both pool sizes reach modules.
        self.num_experts = num_experts
        self.balance_weight = balance
        self.orthogonal_weight = orthogonal
        self.export_mode = export_mode
        factory = functools.partial(
            ChimeraLoRAModule,
            num_experts=num_experts,
            num_frequency_experts=self.num_frequency_experts,
        )
        lora_anima.LoRANetwork.__init__(self, text_encoders, unet, module_class=factory, **kwargs)

    @staticmethod
    def frequency_energy(latents: torch.Tensor) -> torch.Tensor:
        x = latents.float()
        dh = x[..., 1:, :] - x[..., :-1, :]
        dw = x[..., :, 1:] - x[..., :, :-1]
        high = dh.square().mean(dim=tuple(range(1, dh.ndim))) + dw.square().mean(dim=tuple(range(1, dw.ndim)))
        total = x.square().mean(dim=tuple(range(1, x.ndim))).clamp_min(1e-8)
        return (high / total).clamp(0, 10) / 10

    def set_frequency_context(self, sigmas: torch.Tensor, latents: torch.Tensor) -> None:
        fei = self.frequency_energy(latents)
        for module in self.unet_loras:
            if isinstance(module, ChimeraLoRAModule):
                module.set_frequency_context(sigmas, fei)

    def clear_frequency_context(self) -> None:
        for module in self.unet_loras:
            if isinstance(module, ChimeraLoRAModule):
                module.clear_frequency_context()

    def save_weights(self, file, dtype, metadata):
        if self.export_mode == "full":
            return lora_anima.LoRANetwork.save_weights(self, file, dtype, metadata)
        if self.export_mode != "standard_mean":
            raise ValueError("Chimera export_mode must be 'full' or 'standard_mean'")
        state = {key: value.detach().clone() for key, value in self.state_dict().items()}
        for name, module in self.named_children():
            if not isinstance(module, ChimeraLoRAModule):
                continue
            content_up = state[f"{name}.lora_up.weight"] + torch.stack(
                [expert.weight.detach() for expert in module.expert_ups]
            ).mean(0)
            frequency_up = torch.stack([expert.weight.detach() for expert in module.frequency_ups]).mean(0)
            state[f"{name}.lora_down.weight"] = torch.cat(
                (state[f"{name}.lora_down.weight"], state[f"{name}.frequency_down.weight"]), dim=0
            )
            state[f"{name}.lora_up.weight"] = torch.cat((content_up, frequency_up), dim=1)
            state[f"{name}.alpha"] = state[f"{name}.alpha"] * 2
            for key in list(state):
                if key.startswith((f"{name}.expert_ups.", f"{name}.router.", f"{name}.frequency_")):
                    state.pop(key)
        state = {key: value.cpu().to(dtype or value.dtype) for key, value in state.items()}
        if file.endswith(".safetensors"):
            from library import anima_model_io

            md = dict(metadata or {})
            md["ss_chimera_export"] = "standard_mean_lossy_routed_rank_concat"
            anima_model_io.save_safetensors_with_hashes(state, file, md)
        else:
            torch.save(state, file)


def _as_bool(value):
    return str(value).lower() in ("1", "true", "yes", "on")


def create_network(multiplier, network_dim, network_alpha, vae, text_encoders, unet, neuron_dropout=None, **kwargs):
    rank = int(network_dim or 4)
    alpha = float(network_alpha or 1.0)
    exclude = kwargs.get("exclude_patterns")
    exclude = ast.literal_eval(exclude) if exclude else []
    if not isinstance(exclude, list):
        exclude = [exclude]
    exclude.append(r".*(_modulation|_norm|_embedder|final_layer).*")
    include = kwargs.get("include_patterns")
    include = ast.literal_eval(include) if include else None
    if include is not None and not isinstance(include, list):
        include = [include]
    return ChimeraLoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=rank,
        alpha=alpha,
        dropout=neuron_dropout,
        exclude_patterns=exclude,
        include_patterns=include,
        down_init=kwargs.get("down_init", "kaiming"),
        use_timestep_mask=_as_bool(kwargs.get("use_timestep_mask", False)),
        min_rank=int(kwargs.get("min_rank", 1)),
        alpha_rank_scale=float(kwargs.get("alpha_rank_scale", 1.0)),
        num_experts=int(kwargs.get("num_experts", 4)),
        num_frequency_experts=int(kwargs.get("num_frequency_experts", 2)),
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
    content_ids = [int(k.split(".expert_ups.")[1].split(".")[0]) for k in weights_sd if ".expert_ups." in k]
    frequency_ids = [int(k.split(".frequency_ups.")[1].split(".")[0]) for k in weights_sd if ".frequency_ups." in k]
    if not content_ids or not frequency_ids:
        raise ValueError("checkpoint is not a full Chimera checkpoint")
    dims, alphas = {}, {}
    for key, value in weights_sd.items():
        prefix = key.split(".")[0]
        if key.endswith(".lora_down.weight"):
            dims[prefix] = value.shape[0]
        elif key.endswith(".alpha"):
            alphas[prefix] = value
    network = ChimeraLoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=dims,
        modules_alpha=alphas,
        num_experts=max(content_ids) + 1,
        num_frequency_experts=max(frequency_ids) + 1,
        balance_weight=float(kwargs.get("balance_weight", 0.01)),
        orthogonal_weight=float(kwargs.get("orthogonal_weight", 0.01)),
        export_mode=kwargs.get("export_mode", "full"),
        **lora_anima.get_resume_network_kwargs(kwargs, weights_sd),
    )
    return network, weights_sd
