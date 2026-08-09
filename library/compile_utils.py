"""torch.compile helpers for DiT-based models.

Ported / adapted from Musubi Tuner (PR kohya-ss/musubi-tuner#722).

The key design choice is **per-block compilation**: instead of compiling the whole
transformer at once, each transformer block (which all share the same structure) is
compiled individually. This keeps the dynamo cache small (one compiled artifact reused
across blocks), avoids recompilation blow-up, and coexists with block swapping (CPU<->GPU
offloading) because swapped blocks can opt out of compilation per Linear layer.

Currently wired up for Anima only. The helpers are model-agnostic, so other DiT trainers
can reuse them by passing their own list of block ModuleLists as ``target_blocks``.
"""

import argparse

import torch

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def set_activation_memory_budget(budget: float | None) -> None:
    """Configure PyTorch's min-cut partitioner activation-memory trade-off.

    This is intentionally isolated here because ``torch._functorch`` is a private,
    version-dependent API. A requested budget must never be silently ignored.
    """
    if budget is None:
        return
    if not 0.0 <= budget <= 1.0:
        raise ValueError("--activation_memory_budget must be between 0 and 1")
    try:
        import torch._functorch.config as functorch_config
    except ImportError as exc:
        raise RuntimeError(
            "This PyTorch build does not expose activation_memory_budget; upgrade PyTorch or omit the option"
        ) from exc
    if not hasattr(functorch_config, "activation_memory_budget"):
        raise RuntimeError(
            "This PyTorch build does not expose activation_memory_budget; upgrade PyTorch or omit the option"
        )
    functorch_config.activation_memory_budget = budget
    logger.info(f"torch.compile activation memory budget set to {budget:.3f}")


def _mark_anima_sequence_dynamic(args: argparse.Namespace, module: torch.nn.Module) -> None:
    """Install an outer pre-hook that marks the Anima token grid dynamic.

    The hook is registered on the ``OptimizedModule`` after ``torch.compile`` so
    ``mark_dynamic`` executes before Dynamo enters the compiled block. The exact
    token-count bounds are checked here; per-axis upper bounds help symbolic-shape
    inference without assuming a particular aspect ratio.
    """
    min_tokens = getattr(args, "compile_dynamic_sequence_min_tokens", None)
    max_tokens = getattr(args, "compile_dynamic_sequence_max_tokens", None)

    def mark_dynamic(_module, inputs):
        if not inputs or not isinstance(inputs[0], torch.Tensor) or inputs[0].ndim != 5:
            raise RuntimeError("Anima dynamic-sequence compilation expects a B,T,H,W,D tensor as block input")
        x = inputs[0]
        token_count = x.shape[1] * x.shape[2] * x.shape[3]
        if min_tokens is not None and token_count < min_tokens:
            raise ValueError(f"Anima token count {token_count} is below configured minimum {min_tokens}")
        if max_tokens is not None and token_count > max_tokens:
            raise ValueError(f"Anima token count {token_count} exceeds configured maximum {max_tokens}")
        axis_max = max_tokens if max_tokens is not None else None
        for axis in (1, 2, 3):
            torch._dynamo.mark_dynamic(x, axis, min=1, max=axis_max)

    module.register_forward_pre_hook(mark_dynamic)


def disable_linear_from_compile(module: torch.nn.Module):
    """Disable torch.compile for every Linear-like submodule (class name ending with 'Linear').

    Used for blocks that are swapped between CPU and GPU: their weights move across devices
    each step, which conflicts with a compiled graph. We replace ``forward`` with a
    ``torch._dynamo.disable()``-wrapped eager version so dynamo treats it as a graph break.
    """
    for sub_module in module.modules():
        if sub_module.__class__.__name__.endswith("Linear"):
            if not hasattr(sub_module, "_forward_before_disable_compile"):
                sub_module._forward_before_disable_compile = sub_module.forward
                sub_module._eager_forward = torch._dynamo.disable()(sub_module.forward)
            sub_module.forward = sub_module._eager_forward  # override forward to disable compile


def apply_cuda_optimizations(args: argparse.Namespace):
    """Apply optional CUDA performance switches (TF32 / cuDNN benchmark) based on args."""
    if getattr(args, "cuda_allow_tf32", False):
        logger.info("Enabling TF32 for matmul and cuDNN (Ampere or newer GPUs)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if getattr(args, "cuda_cudnn_benchmark", False):
        logger.info("Enabling cuDNN benchmark mode")
        torch.backends.cudnn.benchmark = True


def compile_transformer(
    args: argparse.Namespace,
    transformer: torch.nn.Module,
    target_blocks: list[torch.nn.ModuleList | list[torch.nn.Module]],
    disable_linear: bool,
) -> torch.nn.Module:
    """Compile each block in ``target_blocks`` individually with torch.compile.

    Args:
        args: parsed arguments providing ``compile_backend`` / ``compile_mode`` /
            ``compile_dynamic`` / ``compile_fullgraph`` / ``compile_cache_size_limit``.
        transformer: the model owning the blocks (returned as-is for convenience).
        target_blocks: list of ModuleLists (or plain lists) whose entries are compiled
            in place; ``blocks[i]`` is replaced by its compiled version.
        disable_linear: when True, disable compilation for Linear layers in the given
            blocks first (required for swapped blocks under block swapping).
    """
    if disable_linear:
        logger.info("Disabling Linear layers from torch.compile for block-swapped blocks...")
        for blocks in target_blocks:
            for block in blocks:
                disable_linear_from_compile(block)

    set_activation_memory_budget(getattr(args, "activation_memory_budget", None))

    compile_dynamic = None
    if args.compile_dynamic is not None:
        compile_dynamic = {"true": True, "false": False, "auto": None}[args.compile_dynamic.lower()]

    if getattr(args, "compile_dynamic_sequence", False):
        compile_dynamic = True

    logger.info(
        f"Compiling DiT blocks with torch.compile: backend={args.compile_backend}, mode={args.compile_mode}, "
        f"dynamic={compile_dynamic}, fullgraph={args.compile_fullgraph}"
    )

    if args.compile_cache_size_limit is not None:
        torch._dynamo.config.cache_size_limit = args.compile_cache_size_limit

    # nn.Module の tensor 属性 (例: ControlNet-LLLite が注入する self.cond_emb) は、
    # 既定では dynamic=True でも shape が static に specialize され、解像度バケット毎に
    # recompile を誘発する。forward 入力ではなく属性経由で渡る可変長テンソルを dynamic
    # 対象に含めるため、この強制 static 化を無効化する。Parameter (学習重み) は形状不変
    # なので force_parameter_static_shapes は既定 (True) のままにしておく。
    if hasattr(torch._dynamo.config, "force_nn_module_property_static_shapes"):
        torch._dynamo.config.force_nn_module_property_static_shapes = False

    for blocks in target_blocks:
        for i, block in enumerate(blocks):
            compiled_block = torch.compile(
                block,
                backend=args.compile_backend,
                mode=args.compile_mode,
                dynamic=compile_dynamic,
                fullgraph=args.compile_fullgraph,
            )
            if getattr(args, "compile_dynamic_sequence", False):
                _mark_anima_sequence_dynamic(args, compiled_block)
            blocks[i] = compiled_block
    return transformer
