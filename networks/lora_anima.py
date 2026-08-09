# LoRA network module for Anima
import ast
import contextlib
import math
import os
import re
import weakref
from typing import Dict, List, Optional, Tuple, Type, Union
import torch
from library.utils import setup_logging

import logging

setup_logging()
logger = logging.getLogger(__name__)


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        down_init="kaiming",
        use_timestep_mask=False,
        min_rank=1,
        alpha_rank_scale=1.0,
        channel_scale=None,
    ):
        """
        if alpha == 0 or None, alpha is rank (no scaling).
        """
        super().__init__()
        self.lora_name = lora_name

        for name, value, upper_inclusive in (
            ("dropout", dropout, True),
            ("rank_dropout", rank_dropout, False),
            ("module_dropout", module_dropout, True),
        ):
            if value is None:
                continue
            value = float(value)
            valid = 0.0 <= value <= 1.0 if upper_inclusive else 0.0 <= value < 1.0
            if not valid:
                interval = "[0, 1]" if upper_inclusive else "[0, 1)"
                raise ValueError(f"{name} must be in {interval}, got {value}")

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if down_init == "kaiming" or isinstance(self.lora_down, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        elif down_init == "weight_svd":
            self._init_down_weight_svd(org_module)
        else:
            raise ValueError(f"Unsupported LoRA down_init: {down_init!r}")
        torch.nn.init.zeros_(self.lora_up.weight)

        self._has_channel_scale = False
        if channel_scale is not None:
            if not isinstance(self.lora_down, torch.nn.Linear):
                raise ValueError("channel scaling supports Linear LoRA modules only")
            channel_scale = channel_scale.detach().float().clamp_min(1e-6)
            if channel_scale.ndim != 1 or channel_scale.shape[0] != self.lora_down.in_features:
                raise ValueError(
                    f"channel scale shape {tuple(channel_scale.shape)} does not match "
                    f"{self.lora_down.in_features} input channels"
                )
            channel_scale = channel_scale / channel_scale.mean().clamp_min(1e-12)
            with torch.no_grad():
                self.lora_down.weight.mul_(channel_scale.to(self.lora_down.weight).unsqueeze(0))
            self.register_buffer("inv_scale", (1.0 / channel_scale).contiguous(), persistent=True)
            self._has_channel_scale = True

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.enabled = True
        self.use_timestep_mask = bool(use_timestep_mask)
        self.min_rank = int(min_rank)
        self.alpha_rank_scale = float(alpha_rank_scale)
        if not 1 <= self.min_rank <= self.lora_dim:
            raise ValueError(f"min_rank must be in [1, {self.lora_dim}], got {self.min_rank}")
        if self.alpha_rank_scale <= 0:
            raise ValueError(f"alpha_rank_scale must be positive, got {self.alpha_rank_scale}")
        self.register_buffer("_timestep_rank_mask", torch.empty(0), persistent=False)

    def _init_down_weight_svd(self, org_module: torch.nn.Module) -> None:
        """Initialize a Linear LoRA input basis from the frozen weight's top right singular vectors.

        The up projection remains zero, so the adapter's initial delta is exactly zero and
        the checkpoint stays a conventional LoRA checkpoint.
        """

        if not isinstance(org_module, torch.nn.Linear):
            raise TypeError("weight_svd initialization currently supports Linear layers only")

        weight = org_module.weight.detach().float()
        max_rank = min(weight.shape)
        if self.lora_dim > max_rank:
            raise ValueError(
                f"LoRA rank {self.lora_dim} exceeds the source weight rank bound {max_rank} for {self.lora_name}"
            )

        # A low-rank randomized SVD avoids allocating the full singular-vector matrices of
        # Anima's wide 2K/6K Linear layers. A small oversampling margin is sufficient for
        # an initialization basis; both LoRA factors remain trainable afterwards.
        q = min(self.lora_dim + 6, max_rank)
        _, _, right_vectors = torch.svd_lowrank(weight, q=q, niter=2)
        basis = right_vectors[:, : self.lora_dim].T / math.sqrt(3.0)
        self.lora_down.weight.data.copy_(basis.to(self.lora_down.weight))

    def set_timestep_mask(self, sigmas: torch.Tensor) -> None:
        if not self.use_timestep_mask:
            return
        sigma = sigmas.detach().float().flatten().clamp_(0.0, 1.0)
        fraction = (1.0 - sigma).pow(self.alpha_rank_scale)
        active_rank = self.min_rank + torch.floor(fraction * (self.lora_dim - self.min_rank)).to(torch.long)
        active_rank = active_rank.clamp_(self.min_rank, self.lora_dim)
        columns = torch.arange(self.lora_dim, device=sigma.device).unsqueeze(0)
        self._timestep_rank_mask = (columns < active_rank.unsqueeze(1)).to(dtype=torch.float32)

    def clear_timestep_mask(self) -> None:
        self._timestep_rank_mask = torch.empty(0, device=self._timestep_rank_mask.device)

    def capture_recompute_context(self):
        """Snapshot external adapter state used inside checkpointed blocks."""
        if self._timestep_rank_mask.numel() == 0 and self.enabled:
            return None
        return self._timestep_rank_mask, self.enabled

    def restore_recompute_context(self, state) -> None:
        if state is None:
            self.clear_timestep_mask()
            self.enabled = True
            return
        self._timestep_rank_mask, self.enabled = state

    def _apply_timestep_mask(self, lx: torch.Tensor) -> torch.Tensor:
        if not self.use_timestep_mask or self._timestep_rank_mask.numel() == 0:
            return lx
        if self._timestep_rank_mask.shape[0] not in (1, lx.shape[0]):
            raise RuntimeError(
                f"T-LoRA mask batch {self._timestep_rank_mask.shape[0]} does not match activation batch {lx.shape[0]}"
            )
        mask = self._timestep_rank_mask.to(device=lx.device, dtype=lx.dtype)
        if isinstance(self.lora_down, torch.nn.Conv2d):
            mask = mask[:, :, None, None]
        else:
            for _ in range(lx.ndim - 2):
                mask = mask.unsqueeze(1)
        return lx * mask

    def apply_to(self):
        self.org_forward = self.org_module.forward
        # Projection fusion evaluates this adapter residual separately while
        # keeping the frozen base Q/K/V projections in one GEMM. A weakref avoids
        # registering this module a second time below the original Linear.
        self.org_module._anima_lora_ref = weakref.ref(self)
        self.org_module.forward = self.forward

        del self.org_module

    def forward(self, x):
        org_forwarded = self.org_forward(x)
        if not self.enabled:
            return org_forwarded
        return org_forwarded + self.forward_delta(x)

    def forward_delta(self, x):
        """Return only the LoRA residual, preserving all training dropouts."""

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                if isinstance(self.lora_up, torch.nn.Linear):
                    return torch.zeros((*x.shape[:-1], self.lora_up.out_features), device=x.device, dtype=x.dtype)
                return self.lora_up(self.lora_down(x)) * 0

        x_lora = x * self.inv_scale.to(device=x.device, dtype=x.dtype) if self._has_channel_scale else x
        lx = self.lora_down(x_lora)
        lx = self._apply_timestep_mask(lx)

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if isinstance(self.lora_down, torch.nn.Conv2d):
                # Conv2d: lora_dim is at dim 1 → [B, dim, 1, 1]
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            else:
                # Linear: lora_dim is at last dim → [B, 1, ..., 1, dim]
                for _ in range(len(lx.size()) - 2):
                    mask = mask.unsqueeze(1)
            lx = lx * mask

            # scaling for rank dropout: treat as if the rank is changed
            # maskから計算することも考えられるが、augmentation的な効果を期待してrank_dropoutを用いる
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        return lx * self.multiplier * scale

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class LoRAInfModule(LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        # no dropout for inference
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha)

        self.org_module_ref = [org_module]  # 後から参照できるように
        self.enabled = True
        self.network: LoRANetwork = None

    def set_network(self, network):
        self.network = network

    # freezeしてマージする
    def merge_to(self, sd, dtype, device):
        # extract weight from org_module
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        # get up/down weight
        down_weight = sd["lora_down.weight"].to(torch.float).to(device)
        up_weight = sd["lora_up.weight"].to(torch.float).to(device)

        # weight
        weight = weight.to(torch.float).to(device)  # calc in float

        # merge weight
        if len(weight.size()) == 2:
            # linear
            weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                weight
                + self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * self.scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            # logger.info(conved.size(), weight.size(), module.stride, module.padding)
            weight = weight + self.multiplier * conved * self.scale

        # set weight to org_module
        org_sd["weight"] = weight.to(dtype)
        self.org_module.load_state_dict(org_sd)

    # 復元できるマージのため、このモジュールのweightを返す
    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        # get up/down weight from module
        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        # pre-calculated weight
        if len(down_weight.size()) == 2:
            # linear
            weight = self.multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * self.scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            weight = self.multiplier * conved * self.scale

        return weight

    def default_forward(self, x):
        # logger.info(f"default_forward {self.lora_name} {x.size()}")
        lx = self.lora_down(x)
        lx = self.lora_up(lx)
        return self.org_forward(x) + lx * self.multiplier * self.scale

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    down_init = kwargs.get("down_init", "kaiming")
    if down_init not in ("kaiming", "weight_svd"):
        raise ValueError("down_init must be either 'kaiming' or 'weight_svd'")

    use_timestep_mask = str(kwargs.get("use_timestep_mask", "false")).lower() == "true"
    min_rank = int(kwargs.get("min_rank", 1))
    alpha_rank_scale = float(kwargs.get("alpha_rank_scale", 1.0))
    if not use_timestep_mask and ("min_rank" in kwargs or "alpha_rank_scale" in kwargs):
        raise ValueError("min_rank/alpha_rank_scale require use_timestep_mask=true")

    channel_scaling_alpha = float(kwargs.get("channel_scaling_alpha", 0.0) or 0.0)
    channel_scaling_stats = kwargs.get("channel_scaling_stats", None)
    if not 0.0 <= channel_scaling_alpha <= 1.0:
        raise ValueError("channel_scaling_alpha must be in [0, 1]")
    channel_scales = None
    if channel_scaling_alpha > 0:
        if not channel_scaling_stats:
            raise ValueError("channel_scaling_alpha > 0 requires channel_scaling_stats=<safetensors path>")
        from safetensors.torch import load_file

        raw_channel_stats = load_file(str(channel_scaling_stats))
        channel_scales = {
            name: values.float().clamp_min(1e-6).pow(channel_scaling_alpha)
            for name, values in raw_channel_stats.items()
        }
        logger.info(
            f"Loaded channel calibration for {len(channel_scales)} modules from {channel_scaling_stats} "
            f"(alpha={channel_scaling_alpha})"
        )

    # train LLM adapter
    train_llm_adapter = kwargs.get("train_llm_adapter", "false")
    if train_llm_adapter is not None:
        train_llm_adapter = True if train_llm_adapter.lower() == "true" else False

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)
        if not isinstance(exclude_patterns, list):
            exclude_patterns = [exclude_patterns]

    # add default exclude patterns
    exclude_patterns.append(r".*(_modulation|_norm|_embedder|final_layer).*")

    # regular expression for module selection: exclude and include
    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None:
        include_patterns = ast.literal_eval(include_patterns)
        if not isinstance(include_patterns, list):
            include_patterns = [include_patterns]

    train_adaln = str(kwargs.get("train_adaln", "false")).lower() == "true"
    adaln_rank = kwargs.get("adaln_rank", None)
    adaln_alpha = kwargs.get("adaln_alpha", None)
    adaln_lr = kwargs.get("adaln_lr", None)
    if not train_adaln and any(value is not None for value in (adaln_rank, adaln_alpha, adaln_lr)):
        raise ValueError("adaln_rank/adaln_alpha/adaln_lr require train_adaln=true")

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # verbose
    verbose = kwargs.get("verbose", "false")
    if verbose is not None:
        verbose = True if verbose.lower() == "true" else False

    # regex-specific learning rates / dimensions
    def parse_kv_pairs(kv_pair_str: str, is_int: bool) -> Dict[str, float]:
        """
        Parse a string of key-value pairs separated by commas.
        """
        pairs = {}
        for pair in kv_pair_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                logger.warning(f"Invalid format: {pair}, expected 'key=value'")
                continue
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            try:
                pairs[key] = int(value) if is_int else float(value)
            except ValueError:
                logger.warning(f"Invalid value for {key}: {value}")
        return pairs

    network_reg_lrs = kwargs.get("network_reg_lrs", None)
    if network_reg_lrs is not None:
        reg_lrs = parse_kv_pairs(network_reg_lrs, is_int=False)
    else:
        reg_lrs = None

    network_reg_dims = kwargs.get("network_reg_dims", None)
    if network_reg_dims is not None:
        reg_dims = parse_kv_pairs(network_reg_dims, is_int=True)
    else:
        reg_dims = None

    network_reg_alphas = kwargs.get("network_reg_alphas", None)
    if network_reg_alphas is not None:
        reg_alphas = parse_kv_pairs(network_reg_alphas, is_int=False)
    else:
        reg_alphas = None

    if train_adaln:
        adaln_pattern = r".*adaln_modulation_.*"
        include_patterns = list(include_patterns or [])
        if adaln_pattern not in include_patterns:
            include_patterns.append(adaln_pattern)
        reg_dims = dict(reg_dims or {})
        reg_alphas = dict(reg_alphas or {})
        reg_lrs = dict(reg_lrs or {})
        # Explicit user regex entries retain precedence because module selection uses
        # insertion order and these Anima-specific defaults are appended last.
        reg_dims.setdefault(adaln_pattern, int(adaln_rank) if adaln_rank is not None else min(16, network_dim))
        if adaln_alpha is not None:
            reg_alphas.setdefault(adaln_pattern, float(adaln_alpha))
        if adaln_lr is not None:
            reg_lrs.setdefault(adaln_pattern, float(adaln_lr))

    network = LoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        train_llm_adapter=train_llm_adapter,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        reg_dims=reg_dims,
        reg_alphas=reg_alphas,
        reg_lrs=reg_lrs,
        down_init=down_init,
        use_timestep_mask=use_timestep_mask,
        min_rank=min_rank,
        alpha_rank_scale=alpha_rank_scale,
        channel_scales=channel_scales,
        verbose=verbose,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio", None)
    loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    loraplus_unet_lr_ratio = float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio is not None else None
    loraplus_text_encoder_lr_ratio = float(loraplus_text_encoder_lr_ratio) if loraplus_text_encoder_lr_ratio is not None else None
    if loraplus_lr_ratio is not None or loraplus_unet_lr_ratio is not None or loraplus_text_encoder_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio)

    return network


def create_network_from_weights(multiplier, file, ae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    modules_dim = {}
    modules_alpha = {}
    train_llm_adapter = False
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim

        if "llm_adapter" in lora_name:
            train_llm_adapter = True

    module_class = LoRAInfModule if for_inference else LoRAModule
    dropout = kwargs.get("dropout", kwargs.get("neuron_dropout", None))
    dropout = float(dropout) if dropout is not None else None
    rank_dropout = kwargs.get("rank_dropout", None)
    rank_dropout = float(rank_dropout) if rank_dropout is not None else None
    module_dropout = kwargs.get("module_dropout", None)
    module_dropout = float(module_dropout) if module_dropout is not None else None
    use_timestep_mask = str(kwargs.get("use_timestep_mask", "false")).lower() == "true"
    min_rank = int(kwargs.get("min_rank", 1))
    alpha_rank_scale = float(kwargs.get("alpha_rank_scale", 1.0))
    if not use_timestep_mask and ("min_rank" in kwargs or "alpha_rank_scale" in kwargs):
        raise ValueError("min_rank/alpha_rank_scale require use_timestep_mask=true")

    def parse_float_pairs(value):
        if value is None:
            return None
        pairs = {}
        for pair in str(value).split(","):
            if pair.strip():
                pattern, lr = pair.split("=", 1)
                pairs[pattern.strip()] = float(lr)
        return pairs

    channel_scales = {}
    for key, value in weights_sd.items():
        if key.endswith(".inv_scale"):
            channel_scales[key.rsplit(".", 1)[0]] = value.detach().float().reciprocal()

    network = LoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        train_llm_adapter=train_llm_adapter,
        dropout=dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        reg_lrs=parse_float_pairs(kwargs.get("network_reg_lrs", None)),
        use_timestep_mask=use_timestep_mask,
        min_rank=min_rank,
        alpha_rank_scale=alpha_rank_scale,
        channel_scales=channel_scales or None,
    )
    ratios = (
        kwargs.get("loraplus_lr_ratio", None),
        kwargs.get("loraplus_unet_lr_ratio", None),
        kwargs.get("loraplus_text_encoder_lr_ratio", None),
    )
    ratios = tuple(float(value) if value is not None else None for value in ratios)
    if any(value is not None for value in ratios):
        network.set_loraplus_lr_ratio(*ratios)
    return network, weights_sd


def get_resume_network_kwargs(kwargs, weights_sd) -> dict:
    """Behavioral options that cannot be inferred from ordinary LoRA tensors."""
    dropout = kwargs.get("dropout", kwargs.get("neuron_dropout", None))
    network_kwargs = {
        "dropout": float(dropout) if dropout is not None else None,
        "rank_dropout": float(kwargs["rank_dropout"]) if kwargs.get("rank_dropout") is not None else None,
        "module_dropout": float(kwargs["module_dropout"]) if kwargs.get("module_dropout") is not None else None,
        "use_timestep_mask": str(kwargs.get("use_timestep_mask", "false")).lower() == "true",
        "min_rank": int(kwargs.get("min_rank", 1)),
        "alpha_rank_scale": float(kwargs.get("alpha_rank_scale", 1.0)),
    }
    if not network_kwargs["use_timestep_mask"] and ("min_rank" in kwargs or "alpha_rank_scale" in kwargs):
        raise ValueError("min_rank/alpha_rank_scale require use_timestep_mask=true")
    reg_lrs = kwargs.get("network_reg_lrs")
    if reg_lrs:
        network_kwargs["reg_lrs"] = {
            pattern.strip(): float(value) for pattern, value in (pair.split("=", 1) for pair in str(reg_lrs).split(","))
        }
    channel_scales = {
        key.rsplit(".", 1)[0]: value.detach().float().reciprocal()
        for key, value in weights_sd.items()
        if key.endswith(".inv_scale")
    }
    if channel_scales:
        network_kwargs["channel_scales"] = channel_scales
    return network_kwargs


class LoRANetwork(torch.nn.Module):
    # Target modules: DiT blocks, embedders, final layer. embedders and final layer are excluded by default.
    ANIMA_TARGET_REPLACE_MODULE = ["Block", "PatchEmbed", "TimestepEmbedding", "FinalLayer"]
    # Target modules: LLM Adapter blocks
    ANIMA_ADAPTER_TARGET_REPLACE_MODULE = ["LLMAdapterTransformerBlock"]
    # Target modules for text encoder (Qwen3)
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["Qwen3Attention", "Qwen3MLP", "Qwen3SdpaAttention", "Qwen3FlashAttention2"]

    LORA_PREFIX_ANIMA = "lora_unet"  # ComfyUI compatible
    LORA_PREFIX_TEXT_ENCODER = "lora_te"  # Qwen3

    def __init__(
        self,
        text_encoders: list,
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        module_class: Type[object] = LoRAModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        train_llm_adapter: bool = False,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        reg_dims: Optional[Dict[str, int]] = None,
        reg_alphas: Optional[Dict[str, float]] = None,
        reg_lrs: Optional[Dict[str, float]] = None,
        down_init: str = "kaiming",
        use_timestep_mask: bool = False,
        min_rank: int = 1,
        alpha_rank_scale: float = 1.0,
        channel_scales: Optional[Dict[str, torch.Tensor]] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.train_llm_adapter = train_llm_adapter
        self.reg_dims = reg_dims
        self.reg_alphas = reg_alphas
        self.reg_lrs = reg_lrs
        self.down_init = down_init
        self.use_timestep_mask = use_timestep_mask
        self.min_rank = min_rank
        self.alpha_rank_scale = alpha_rank_scale
        self.channel_scales = channel_scales

        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if modules_dim is not None:
            logger.info("create LoRA network from weights")
        else:
            logger.info(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )

        # compile regular expression if specified
        def str_to_re_patterns(patterns: Optional[List[str]]) -> List[re.Pattern]:
            re_patterns = []
            if patterns is not None:
                for pattern in patterns:
                    try:
                        re_pattern = re.compile(pattern)
                    except re.error as e:
                        logger.error(f"Invalid pattern '{pattern}': {e}")
                        continue
                    re_patterns.append(re_pattern)
            return re_patterns

        exclude_re_patterns = str_to_re_patterns(exclude_patterns)
        include_re_patterns = str_to_re_patterns(include_patterns)

        # create module instances
        def create_modules(
            is_unet: bool,
            text_encoder_idx: Optional[int],
            root_module: torch.nn.Module,
            target_replace_modules: List[str],
            default_dim: Optional[int] = None,
        ) -> Tuple[List[LoRAModule], List[str]]:
            prefix = self.LORA_PREFIX_ANIMA if is_unet else self.LORA_PREFIX_TEXT_ENCODER

            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if target_replace_modules is None or module.__class__.__name__ in target_replace_modules:
                    if target_replace_modules is None:
                        module = root_module

                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            original_name = (name + "." if name else "") + child_name
                            lora_name = f"{prefix}.{original_name}".replace(".", "_")

                            # exclude/include filter (fullmatch: pattern must match the entire original_name)
                            excluded = any(pattern.fullmatch(original_name) for pattern in exclude_re_patterns)
                            included = any(pattern.fullmatch(original_name) for pattern in include_re_patterns)
                            if excluded and not included:
                                if verbose:
                                    logger.info(f"exclude: {original_name}")
                                continue

                            dim = None
                            alpha_val = None

                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    # Some standard LoRA exporters omit alpha;
                                    # in that format alpha defaults to rank.
                                    alpha_val = modules_alpha.get(lora_name, dim)
                            else:
                                if self.reg_dims is not None:
                                    for reg, d in self.reg_dims.items():
                                        if re.fullmatch(reg, original_name):
                                            dim = d
                                            alpha_val = self.alpha
                                            logger.info(f"Module {original_name} matched with regex '{reg}' -> dim: {dim}")
                                            break
                                # fallback to default dim if not matched by reg_dims or reg_dims is not specified
                                if dim is None:
                                    if is_linear or is_conv2d_1x1:
                                        dim = default_dim if default_dim is not None else self.lora_dim
                                        alpha_val = self.alpha

                                if self.reg_alphas is not None:
                                    for reg, configured_alpha in self.reg_alphas.items():
                                        if re.fullmatch(reg, original_name):
                                            alpha_val = configured_alpha
                                            logger.info(
                                                f"Module {original_name} matched with regex '{reg}' -> alpha: {alpha_val}"
                                            )
                                            break

                            if dim is None or dim == 0:
                                if is_linear or is_conv2d_1x1:
                                    skipped.append(lora_name)
                                continue

                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha_val,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                                down_init=self.down_init,
                                use_timestep_mask=self.use_timestep_mask,
                                min_rank=min(self.min_rank, dim),
                                alpha_rank_scale=self.alpha_rank_scale,
                                channel_scale=(
                                    None
                                    if not is_unet or self.channel_scales is None
                                    else self.channel_scales.get(lora_name, self.channel_scales.get(original_name))
                                ),
                            )
                            lora.original_name = original_name
                            loras.append(lora)

                    if target_replace_modules is None:
                        break
            return loras, skipped

        # Create LoRA for text encoders (Qwen3 - typically not trained for Anima)
        self.text_encoder_loras: List[Union[LoRAModule, LoRAInfModule]] = []
        skipped_te = []
        if text_encoders is not None:
            for i, text_encoder in enumerate(text_encoders):
                if text_encoder is None:
                    continue
                logger.info(f"create LoRA for Text Encoder {i+1}:")
                te_loras, te_skipped = create_modules(False, i, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
                logger.info(f"create LoRA for Text Encoder {i+1}: {len(te_loras)} modules.")
                self.text_encoder_loras.extend(te_loras)
                skipped_te += te_skipped

        # Create LoRA for DiT blocks
        target_modules = list(LoRANetwork.ANIMA_TARGET_REPLACE_MODULE)
        if train_llm_adapter:
            target_modules.extend(LoRANetwork.ANIMA_ADAPTER_TARGET_REPLACE_MODULE)

        self.unet_loras: List[Union[LoRAModule, LoRAInfModule]]
        self.unet_loras, skipped_un = create_modules(True, None, unet, target_modules)

        logger.info(f"create LoRA for Anima DiT: {len(self.unet_loras)} modules.")
        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:60} {lora.lora_dim}, {lora.alpha}")

        skipped = skipped_te + skipped_un
        if verbose and len(skipped) > 0:
            logger.warning(f"dim (rank) is 0, {len(skipped)} LoRA modules are skipped:")
            for name in skipped:
                logger.info(f"\t{name}")

        # assertion: no duplicate names
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def set_timestep_mask(self, sigmas: torch.Tensor) -> None:
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.set_timestep_mask(sigmas)

    def clear_timestep_mask(self) -> None:
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.clear_timestep_mask()

    def capture_recompute_context(self):
        states = tuple(lora.capture_recompute_context() for lora in self.text_encoder_loras + self.unet_loras)
        return None if all(state is None for state in states) else states

    def restore_recompute_context(self, states) -> None:
        loras = self.text_encoder_loras + self.unet_loras
        if states is None:
            states = (None,) * len(loras)
        if len(states) != len(loras):
            raise RuntimeError("Anima adapter checkpoint context does not match the active LoRA modules")
        for lora, state in zip(loras, states):
            lora.restore_recompute_context(state)

    @contextlib.contextmanager
    def use_recompute_context(self, states):
        previous = self.capture_recompute_context()
        self.restore_recompute_context(states)
        try:
            yield
        finally:
            self.restore_recompute_context(previous)

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        # Standard files have inv_scale baked into lora_down. Re-absorb the
        # calibration when resuming an internally channel-scaled run.
        for lora in self.text_encoder_loras + self.unet_loras:
            if getattr(lora, "_has_channel_scale", False) and f"{lora.lora_name}.inv_scale" not in weights_sd:
                with torch.no_grad():
                    lora.lora_down.weight.div_(lora.inv_scale.to(lora.lora_down.weight).unsqueeze(0))
        return info

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(f"enable LoRA for text encoder: {len(self.text_encoder_loras)} modules")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"enable LoRA for DiT: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        if apply_unet and hasattr(unet, "blocks"):
            provider_ref = weakref.ref(self)
            unet._anima_adapter_network = provider_ref
            for block in unet.blocks:
                block._anima_adapter_context_provider = provider_ref

    def is_mergeable(self):
        return True

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(LoRANetwork.LORA_PREFIX_ANIMA):
                apply_unet = True

        if apply_text_encoder:
            logger.info("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable LoRA for DiT")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
            lora.merge_to(sd_for_lora, dtype, device)

        logger.info("weights are merged")

    def set_loraplus_lr_ratio(self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio

        logger.info(f"LoRA+ UNet LR Ratio: {self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio}")
        logger.info(f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}")

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lr, unet_lr, default_lr):
        if text_encoder_lr is None or (isinstance(text_encoder_lr, list) and len(text_encoder_lr) == 0):
            text_encoder_lr = [default_lr]
        elif isinstance(text_encoder_lr, float) or isinstance(text_encoder_lr, int):
            text_encoder_lr = [float(text_encoder_lr)]
        elif len(text_encoder_lr) == 1:
            pass  # already a list with one element

        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}}
            reg_groups = {}
            reg_lrs_list = list(self.reg_lrs.items()) if self.reg_lrs is not None else []

            for lora in loras:
                matched_reg_lr = None
                for i, (regex_str, reg_lr) in enumerate(reg_lrs_list):
                    if re.fullmatch(regex_str, lora.original_name):
                        matched_reg_lr = (i, reg_lr)
                        logger.info(f"Module {lora.original_name} matched regex '{regex_str}' -> LR {reg_lr}")
                        break

                for name, param in lora.named_parameters():
                    if matched_reg_lr is not None:
                        reg_idx, reg_lr = matched_reg_lr
                        group_key = f"reg_lr_{reg_idx}"
                        if group_key not in reg_groups:
                            reg_groups[group_key] = {"lora": {}, "plus": {}, "lr": reg_lr}
                        if loraplus_ratio is not None and "lora_up" in name:
                            reg_groups[group_key]["plus"][f"{lora.lora_name}.{name}"] = param
                        else:
                            reg_groups[group_key]["lora"][f"{lora.lora_name}.{name}"] = param
                        continue

                    if loraplus_ratio is not None and "lora_up" in name:
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            params = []
            descriptions = []
            for group_key, group in reg_groups.items():
                reg_lr = group["lr"]
                for key in ("lora", "plus"):
                    param_data = {"params": group[key].values()}
                    if len(param_data["params"]) == 0:
                        continue
                    if key == "plus":
                        param_data["lr"] = reg_lr * loraplus_ratio if loraplus_ratio is not None else reg_lr
                    else:
                        param_data["lr"] = reg_lr
                    if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                        logger.info("NO LR skipping!")
                        continue
                    params.append(param_data)
                    desc = f"reg_lr_{group_key.split('_')[-1]}"
                    descriptions.append(desc + (" plus" if key == "plus" else ""))

            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}
                if len(param_data["params"]) == 0:
                    continue
                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * loraplus_ratio
                    else:
                        param_data["lr"] = lr
                if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                    logger.info("NO LR skipping!")
                    continue
                params.append(param_data)
                descriptions.append("plus" if key == "plus" else "")
            return params, descriptions

        if self.text_encoder_loras:
            loraplus_ratio = self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio
            te1_loras = [lora for lora in self.text_encoder_loras if lora.lora_name.startswith(self.LORA_PREFIX_TEXT_ENCODER)]
            if len(te1_loras) > 0:
                logger.info(f"Text Encoder 1 (Qwen3): {len(te1_loras)} modules, LR {text_encoder_lr[0]}")
                params, descriptions = assemble_params(te1_loras, text_encoder_lr[0], loraplus_ratio)
                all_params.extend(params)
                lr_descriptions.extend(["textencoder 1" + (" " + d if d else "") for d in descriptions])

        if self.unet_loras:
            params, descriptions = assemble_params(
                self.unet_loras,
                unet_lr if unet_lr is not None else default_lr,
                self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(["unet" + (" " + d if d else "") for d in descriptions])

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        pass  # not supported

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = {key: value.detach().clone() for key, value in self.state_dict().items()}
        # Bake the SmoothQuant reparameterization into ordinary lora_down
        # weights and omit calibration buffers. The file remains a standard
        # Anima LoRA for ComfyUI and existing loaders.
        for key in list(state_dict):
            if not key.endswith(".inv_scale"):
                continue
            prefix = key[: -len(".inv_scale")]
            down_key = f"{prefix}.lora_down.weight"
            inv_scale = state_dict.pop(key)
            if down_key in state_dict:
                state_dict[down_key].mul_(inv_scale.to(state_dict[down_key]).unsqueeze(0))

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            import library.anima_model_io as model_io

            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = model_io.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def backup_weights(self):
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not org_module._lora_restored:
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True

    def pre_calculation(self):
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            sd = org_module.state_dict()

            org_weight = sd["weight"]
            lora_weight = lora.get_weight().to(org_weight.device, dtype=org_weight.dtype)
            sd["weight"] = org_weight + lora_weight
            assert sd["weight"].shape == org_weight.shape
            org_module.load_state_dict(sd)

            org_module._lora_restored = False
            lora.enabled = False

    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)
