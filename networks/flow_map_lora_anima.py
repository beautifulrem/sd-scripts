"""AnyFlow-style arbitrary-interval flow-map LoRA for Anima."""

from __future__ import annotations

import weakref

import torch

from networks import lora_anima


class FlowMapLoRANetwork(lora_anima.LoRANetwork):
    def is_mergeable(self):
        return False

    def __init__(self, text_encoders, unet, *, interval_hidden_dim=128, **kwargs):
        super().__init__(text_encoders, unet, **kwargs)
        model_dim = int(unet.model_channels)
        self.interval_embedder = torch.nn.Sequential(
            torch.nn.Linear(2, int(interval_hidden_dim)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(interval_hidden_dim), model_dim),
        )
        torch.nn.init.zeros_(self.interval_embedder[-1].weight)
        torch.nn.init.zeros_(self.interval_embedder[-1].bias)
        self.register_buffer("_flow_t", torch.empty(0), persistent=False)
        self.register_buffer("_flow_r", torch.empty(0), persistent=False)
        self._flow_hooks = []

    def set_flow_interval(self, source_t: torch.Tensor, target_r: torch.Tensor) -> None:
        self._flow_t = source_t.detach().flatten()
        self._flow_r = target_r.detach().flatten()

    def clear_flow_interval(self) -> None:
        self._flow_t = torch.empty(0, device=self.interval_embedder[0].weight.device)
        self._flow_r = torch.empty(0, device=self.interval_embedder[0].weight.device)

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        super().apply_to(text_encoders, unet, apply_text_encoder, apply_unet)
        unet._anima_flow_map_network = weakref.ref(self)
        for block in unet.blocks:
            def inject_interval(_module, args, kwargs):
                if self._flow_t.numel() == 0:
                    return args, kwargs
                embedding = args[1]
                batch = embedding.shape[0]
                t = self._flow_t.to(embedding.device).expand(batch)
                r = self._flow_r.to(embedding.device).expand(batch)
                interval = self.interval_embedder(torch.stack((t, r), dim=-1).float()).to(embedding.dtype).unsqueeze(1)
                updated = list(args)
                updated[1] = embedding + interval
                return tuple(updated), kwargs

            self._flow_hooks.append(block.register_forward_pre_hook(inject_interval, with_kwargs=True))

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lr, unet_lr, default_lr):
        groups, descriptions = super().prepare_optimizer_params_with_multiple_te_lrs(
            text_encoder_lr, unet_lr, default_lr
        )
        lr = unet_lr if unet_lr is not None else default_lr
        groups.append({"params": self.interval_embedder.parameters(), "lr": lr})
        descriptions.append("flow interval embedder")
        return groups, descriptions


def create_network(multiplier, network_dim, network_alpha, vae, text_encoders, unet, neuron_dropout=None, **kwargs):
    return FlowMapLoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=int(network_dim or 32),
        alpha=float(network_alpha or network_dim or 32),
        dropout=neuron_dropout,
        exclude_patterns=[r".*(_modulation|_norm|_embedder|final_layer).*"],
        down_init=kwargs.get("down_init", "weight_svd"),
        interval_hidden_dim=int(kwargs.get("interval_hidden_dim", 128)),
    )


def create_network_from_weights(multiplier, file, ae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if file.endswith(".safetensors"):
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
    dims, alphas = {}, {}
    for key, value in weights_sd.items():
        prefix = key.split(".")[0]
        if key.endswith(".lora_down.weight"):
            dims[prefix] = value.shape[0]
        elif key.endswith(".alpha"):
            alphas[prefix] = value
    hidden = weights_sd["interval_embedder.0.weight"].shape[0]
    network = FlowMapLoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=dims,
        modules_alpha=alphas,
        interval_hidden_dim=hidden,
        **lora_anima.get_resume_network_kwargs(kwargs, weights_sd),
    )
    return network, weights_sd
