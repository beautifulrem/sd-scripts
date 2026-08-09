# Anima model loading/saving utilities

import math
import os
import re
from typing import Dict, List, Optional, Union
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from accelerate import init_empty_weights

from library.fp8_optimization_utils import apply_fp8_monkey_patch
from library.lora_utils import load_safetensors_with_lora_and_fp8
from library import anima_models
from library.safetensors_utils import WeightTransformHooks
from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# Original Anima high-precision keys. Kept for reference, but not used currently.
# # Keys that should stay in high precision (float32/bfloat16, not quantized)
# KEEP_IN_HIGH_PRECISION = ["x_embedder", "t_embedder", "t_embedding_norm", "final_layer"]


FP8_OPTIMIZATION_TARGET_KEYS = ["blocks", ""]
# ".embed." excludes Embedding in LLMAdapter
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["_embedder", "norm", "adaln", "final_layer", ".embed."]


def _normalize_anima_key(key: str) -> str:
    for prefix in ("net.", "model.diffusion_model."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _infer_anima_dit_config_from_shapes(shapes: Dict[str, tuple[int, ...]]) -> dict:
    """Infer architecture-defining Anima options from checkpoint tensor shapes."""
    shapes = {_normalize_anima_key(key): tuple(shape) for key, shape in shapes.items()}
    required = (
        "x_embedder.proj.1.weight",
        "blocks.0.self_attn.q_norm.weight",
        "blocks.0.cross_attn.k_proj.weight",
        "blocks.0.mlp.layer1.weight",
        "final_layer.linear.weight",
    )
    missing = [key for key in required if key not in shapes]
    if missing:
        raise ValueError(f"cannot infer Anima DiT architecture; checkpoint is missing {missing}")

    model_channels, patch_input_features = shapes["x_embedder.proj.1.weight"]
    head_dim = shapes["blocks.0.self_attn.q_norm.weight"][0]
    if model_channels % head_dim:
        raise ValueError(f"Anima model width {model_channels} is not divisible by attention head width {head_dim}")
    block_indices = {
        int(match.group(1))
        for key in shapes
        if (match := re.match(r"blocks\.(\d+)\.", key)) is not None
    }
    if not block_indices or block_indices != set(range(max(block_indices) + 1)):
        raise ValueError("Anima checkpoint contains a non-contiguous transformer block set")

    # Anima image checkpoints have 16 output latent channels and one temporal
    # patch.  The final projection therefore uniquely determines spatial patch.
    out_channels = anima_models.Anima.LATENT_CHANNELS
    final_features = shapes["final_layer.linear.weight"][0]
    spatial_square = final_features // out_channels
    patch_spatial = math.isqrt(spatial_square)
    if final_features != out_channels * patch_spatial**2:
        raise ValueError(f"unsupported Anima final projection shape {shapes['final_layer.linear.weight']}")
    patched_input_channels = patch_input_features // patch_spatial**2
    if patch_input_features != patched_input_channels * patch_spatial**2 or patched_input_channels not in (16, 17):
        raise ValueError(f"unsupported Anima patch embedding shape {shapes['x_embedder.proj.1.weight']}")
    rope_sequence_length = shapes.get("pos_embedder.seq", (256,))[0]
    extra_h = shapes.get("extra_pos_embedder.pos_emb_h")
    extra_w = shapes.get("extra_pos_embedder.pos_emb_w")
    extra_t = shapes.get("extra_pos_embedder.pos_emb_t")
    max_img_h = (extra_h[0] if extra_h else rope_sequence_length) * patch_spatial
    max_img_w = (extra_w[0] if extra_w else rope_sequence_length) * patch_spatial
    max_frames = (extra_t[0] if extra_t else min(128, rope_sequence_length))

    adaln_key = "blocks.0.adaln_modulation_self_attn.1.weight"
    use_adaln_lora = shapes.get("t_embedder.1.linear_2.weight", (model_channels,))[0] == 3 * model_channels
    llm_block_indices = {
        int(match.group(1))
        for key in shapes
        if (match := re.match(r"llm_adapter\.blocks\.(\d+)\.", key)) is not None
    }
    use_llm_adapter = bool(llm_block_indices)
    llm_model_dim = shapes.get("llm_adapter.blocks.0.cross_attn.q_proj.weight", (1024,))[0]
    llm_head_dim = shapes.get("llm_adapter.blocks.0.cross_attn.q_norm.weight", (64,))[0]
    return {
        # A RoPE checkpoint stores one sequence buffer sized to the largest
        # configured axis.  Using that capacity for both image axes preserves
        # tensor compatibility even when the original limits were rectangular.
        "max_img_h": max_img_h,
        "max_img_w": max_img_w,
        "max_frames": max_frames,
        "in_channels": 16,
        "out_channels": out_channels,
        "patch_spatial": patch_spatial,
        "patch_temporal": 1,
        "model_channels": model_channels,
        "concat_padding_mask": patched_input_channels == 17,
        "crossattn_emb_channels": shapes["blocks.0.cross_attn.k_proj.weight"][1],
        "pos_emb_cls": "rope3d",
        "pos_emb_learnable": True,
        "pos_emb_interpolation": "crop",
        "min_fps": 1,
        "max_fps": 30,
        "use_adaln_lora": use_adaln_lora,
        "adaln_lora_dim": shapes[adaln_key][0] if use_adaln_lora and adaln_key in shapes else 256,
        "num_blocks": max(block_indices) + 1,
        "num_heads": model_channels // head_dim,
        "mlp_ratio": shapes["blocks.0.mlp.layer1.weight"][0] / model_channels,
        "extra_per_block_abs_pos_emb": any(key.startswith("extra_pos_embedder.") for key in shapes),
        "rope_h_extrapolation_ratio": 4.0,
        "rope_w_extrapolation_ratio": 4.0,
        "rope_t_extrapolation_ratio": 1.0,
        "extra_h_extrapolation_ratio": 1.0,
        "extra_w_extrapolation_ratio": 1.0,
        "extra_t_extrapolation_ratio": 1.0,
        "rope_enable_fps_modulation": False,
        "use_llm_adapter": use_llm_adapter,
        "llm_adapter_source_dim": shapes.get("llm_adapter.blocks.0.cross_attn.k_proj.weight", (1024, 1024))[1],
        "llm_adapter_target_dim": shapes.get("llm_adapter.embed.weight", (32128, 1024))[1],
        "llm_adapter_model_dim": llm_model_dim,
        "llm_adapter_num_layers": max(llm_block_indices) + 1 if llm_block_indices else 6,
        "llm_adapter_num_heads": llm_model_dim // llm_head_dim,
        "llm_adapter_self_attn": any(
            key.startswith("llm_adapter.blocks.0.self_attn.") for key in shapes
        ),
    }


def _read_anima_checkpoint_shapes(dit_path: str) -> Dict[str, tuple[int, ...]]:
    from library.lora_utils import get_split_weight_filenames

    paths = get_split_weight_filenames(dit_path) or [dit_path]
    shapes = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                shapes[key] = tuple(checkpoint.get_slice(key).get_shape())
    return shapes


def _normalize_separate_llm_adapter_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state.items():
        key = _normalize_anima_key(key)
        if not key.startswith("llm_adapter."):
            key = f"llm_adapter.{key}"
        normalized[key] = value
    return normalized


def load_anima_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    llm_adapter_path: Optional[str] = None,
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[list[float]] = None,
) -> anima_models.Anima:
    """
    Load Anima model from the specified checkpoint.

    Args:
        device (Union[str, torch.device]): Device for optimization or merging
        dit_path (str): Path to the DiT model checkpoint.
        attn_mode (str): Attention mode to use, e.g., "torch", "flash", etc.
        split_attn (bool): Whether to use split attention.
        loading_device (Union[str, torch.device]): Device to load the model weights on.
        dit_weight_dtype (Optional[torch.dtype]): Data type of the DiT weights.
            If None, it will be loaded as is (same as the state_dict) or scaled for fp8. if not None, model weights will be casted to this dtype.
        fp8_scaled (bool): Whether to use fp8 scaling for the model weights.
        lora_weights_list (Optional[List[Dict[str, torch.Tensor]]]): LoRA weights to apply, if any.
        lora_multipliers (Optional[List[float]]): LoRA multipliers for the weights, if any.
    """
    # dit_weight_dtype is None for fp8_scaled
    assert (
        not fp8_scaled and dit_weight_dtype is not None
    ) or dit_weight_dtype is None, "dit_weight_dtype should be None when fp8_scaled is True"

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    checkpoint_shapes = _read_anima_checkpoint_shapes(dit_path)
    separate_llm_state = None
    if llm_adapter_path is not None:
        separate_llm_state = _normalize_separate_llm_adapter_state(load_file(llm_adapter_path, device="cpu"))
        checkpoint_shapes.update({key: tuple(value.shape) for key, value in separate_llm_state.items()})
    dit_config = _infer_anima_dit_config_from_shapes(checkpoint_shapes)
    dit_config.update(attn_mode=attn_mode, split_attn=split_attn)
    with init_empty_weights():
        model = anima_models.Anima(**dit_config)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    # load model weights with dynamic fp8 optimization and LoRA merging if needed
    logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")

    def rename_hook(key: str) -> str:
        # Rename keys to remove "net." prefix for anima-base-v1.0.safetensors and previous versions
        if key.startswith("net."):
            return key[len("net.") :]
        # Also remove "model.diffusion_model." prefix for anima-aesthetics-v1.0.safetensors and later versions
        if key.startswith("model.diffusion_model."):
            return key[len("model.diffusion_model.") :]
        return key

    rename_hooks = WeightTransformHooks(rename_hook=rename_hook)

    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
        weight_transform_hooks=rename_hooks,
    )
    if separate_llm_state is not None:
        for key, value in separate_llm_state.items():
            if dit_weight_dtype is not None and value.dtype.is_floating_point:
                value = value.to(dtype=dit_weight_dtype)
            sd[key] = value.to(loading_device) if loading_device.type != "cpu" else value

    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)

        if loading_device.type != "cpu":
            # make sure all the model weights are on the loading_device
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing:
        # Filter out expected missing buffers (initialized in __init__, not saved in checkpoint)
        unexpected_missing = [
            k
            for k in missing
            if not any(buf_name in k for buf_name in ("seq", "dim_spatial_range", "dim_temporal_range", "inv_freq"))
        ]
        if unexpected_missing:
            # Raise error to avoid silent failures
            raise RuntimeError(
                f"Missing keys in checkpoint: {unexpected_missing[:10]}{'...' if len(unexpected_missing) > 10 else ''}"
            )
        missing = {}  # all missing keys were expected
    if unexpected:
        # Raise error to avoid silent failures
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    logger.info(f"Loaded DiT model from {dit_path}, unexpected missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    return model


def load_qwen3_tokenizer(qwen3_path: str):
    """Load Qwen3 tokenizer only (without the text encoder model).

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file.
                     If a directory, loads tokenizer from it directly.
                     If a file, uses configs/qwen3_06b/ for tokenizer config.
    Returns:
        tokenizer
    """
    from transformers import AutoTokenizer

    if os.path.isdir(qwen3_path):
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
    else:
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "qwen3_06b")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )
        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_qwen3_text_encoder(
    qwen3_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    lora_weights: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[List[float]] = None,
):
    """Load Qwen3-0.6B text encoder.

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file
        dtype: Model dtype
        device: Device to load to

    Returns:
        (text_encoder_model, tokenizer)
    """
    import transformers
    from transformers import AutoTokenizer

    logger.info(f"Loading Qwen3 text encoder from {qwen3_path}")

    if os.path.isdir(qwen3_path):
        # Directory with full model
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(qwen3_path, torch_dtype=dtype, local_files_only=True).model
    else:
        # Single safetensors file - use configs/qwen3_06b/ for config
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "qwen3_06b")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )

        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)
        qwen3_config = transformers.Qwen3Config.from_pretrained(config_dir, local_files_only=True)
        model = transformers.Qwen3ForCausalLM(qwen3_config).model

        # Load weights
        if qwen3_path.endswith(".safetensors"):
            if lora_weights is None:
                state_dict = load_file(qwen3_path, device="cpu")
            else:
                state_dict = load_safetensors_with_lora_and_fp8(
                    model_files=qwen3_path,
                    lora_weights_list=lora_weights,
                    lora_multipliers=lora_multipliers,
                    fp8_optimization=False,
                    calc_device=device,
                    move_to_device=True,
                    dit_weight_dtype=None,
                )
        else:
            assert lora_weights is None, "LoRA weights merging is only supported for safetensors checkpoints"
            state_dict = torch.load(qwen3_path, map_location="cpu", weights_only=True)

        # Remove 'model.' prefix if present
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_sd[k[len("model.") :]] = v
            else:
                new_sd[k] = v

        info = model.load_state_dict(new_sd, strict=False)
        logger.info(f"Loaded Qwen3 state dict: {info}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model = model.requires_grad_(False).to(device, dtype=dtype)

    logger.info(f"Loaded Qwen3 text encoder. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def load_t5_tokenizer(t5_tokenizer_path: Optional[str] = None):
    """Load T5 tokenizer for LLM Adapter target tokens.

    Args:
        t5_tokenizer_path: Optional path to T5 tokenizer directory. If None, uses default configs.
    """
    from transformers import T5TokenizerFast

    if t5_tokenizer_path is not None:
        return T5TokenizerFast.from_pretrained(t5_tokenizer_path, local_files_only=True)

    # Use bundled config
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "t5_old")
    if os.path.exists(config_dir):
        return T5TokenizerFast(
            vocab_file=os.path.join(config_dir, "spiece.model"),
            tokenizer_file=os.path.join(config_dir, "tokenizer.json"),
        )

    raise FileNotFoundError(
        f"T5 tokenizer config directory not found at {config_dir}. "
        "Expected configs/t5_old/ with spiece.model and tokenizer.json. "
        "You can download these from the google/t5-v1_1-xxl HuggingFace repository."
    )


def save_anima_model(
    save_path: str, dit_state_dict: Dict[str, torch.Tensor], metadata: Dict[str, any], dtype: Optional[torch.dtype] = None
):
    """Save Anima DiT model with 'net.' prefix for ComfyUI compatibility.

    Args:
        save_path: Output path (.safetensors)
        dit_state_dict: State dict from dit.state_dict()
        metadata: Metadata dict to include in the safetensors file
        dtype: Optional dtype to cast to before saving
    """
    prefixed_sd = {}
    for k, v in dit_state_dict.items():
        if dtype is not None:
            # v = v.to(dtype)
            v = v.detach().clone().to("cpu").to(dtype)  # Reduce GPU memory usage during save
        prefixed_sd["net." + k] = v.contiguous()

    if metadata is None:
        metadata = {}
    metadata["format"] = "pt"  # For compatibility with the official .safetensors file

    save_file(prefixed_sd, save_path, metadata=metadata)  # safetensors.save_file consumes a lot of memory, but Anima is small enough
    logger.info(f"Saved Anima model to {save_path}")
