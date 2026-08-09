from argparse import Namespace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from library import anima_models, compile_utils
from library.anima_repa import repa_interpolation_code, validate_repa_sidecar
from library.controlnet_dataset import ControlNetDataset
from library.dataset import BucketManager
from library.utils import trim_and_resize_if_required
from networks.lora_anima import LoRAModule
from tools.cache_anima_repa_features import free_fit_geometry, free_fit_size, prepare_repa_image


def test_free_fit_bucket_preserves_aspect_and_target_area():
    manager = BucketManager(False, (1024, 1024), 256, 2048, 16, free_fit=True)
    reso, resized, ar_error = manager.select_bucket(1600, 900)

    assert reso[0] % 16 == 0 and reso[1] % 16 == 0
    assert abs(ar_error) < 0.02
    assert abs(reso[0] * reso[1] - 1024 * 1024) / (1024 * 1024) < 0.04
    assert resized[0] >= reso[0] and resized[1] >= reso[1]


def test_free_fit_allows_upscale_and_rejects_no_upscale_combination():
    manager = BucketManager(False, (1024, 1024), 256, 2048, 16, free_fit=True)
    reso, resized, _ = manager.select_bucket(320, 320)
    assert reso == (1024, 1024)
    assert resized == (1024, 1024)
    with pytest.raises(ValueError):
        BucketManager(True, (1024, 1024), 256, 2048, 16, free_fit=True)


@pytest.mark.parametrize("image_size", [(256, 523), (400, 1600), (1600, 900), (2048, 2048), (701, 256)])
def test_repa_cache_free_fit_exactly_matches_training_bucket(image_size):
    manager = BucketManager(False, (1024, 1024), 256, 2048, 16, free_fit=True)
    expected, expected_resized, _ = manager.select_bucket(*image_size)
    actual, actual_resized = free_fit_geometry(*image_size, 1024 * 1024, 16, 2048)
    assert free_fit_size(*image_size, 1024 * 1024, 16, 2048) == expected
    assert (actual, actual_resized) == (expected, expected_resized)


def test_repa_preprocessing_preserves_aspect_then_center_crops():
    image = Image.fromarray(np.arange(80 * 160 * 3, dtype=np.uint8).reshape(80, 160, 3))
    prepared = prepare_repa_image(image, (64, 64), (128, 64))
    expected, _, _ = trim_and_resize_if_required(
        False, np.asarray(image)[:, :, ::-1].copy(), (64, 64), (128, 64)
    )
    np.testing.assert_array_equal(np.asarray(prepared), expected[:, :, ::-1])


def test_repa_preprocessing_honors_dataset_interpolation():
    image = Image.fromarray(np.arange(17 * 31 * 3, dtype=np.uint8).reshape(17, 31, 3))
    prepared = prepare_repa_image(image, (16, 16), (29, 16), "nearest")
    expected, _, _ = trim_and_resize_if_required(
        False, np.asarray(image)[:, :, ::-1].copy(), (16, 16), (29, 16), resize_interpolation="nearest"
    )
    np.testing.assert_array_equal(np.asarray(prepared), expected[:, :, ::-1])


def test_repa_preprocessing_preserves_rgb_channels_with_pil_interpolation():
    image = Image.new("RGB", (32, 16), color=(240, 20, 5))
    prepared = prepare_repa_image(image, (16, 16), (32, 16), resize_interpolation="lanczos")
    red, green, blue = np.asarray(prepared)[8, 8]

    assert red > 200
    assert green < 40
    assert blue < 40


def test_repa_sidecar_requires_bucket_metadata():
    with pytest.raises(ValueError, match="no bucket metadata"):
        validate_repa_sidecar({"tokens": torch.zeros(2, 3)}, "sample.safetensors", "tokens", (64, 64))


def test_repa_sidecar_rejects_mismatched_interpolation():
    sidecar = {
        "tokens": torch.zeros(2, 3),
        "anima_bucket_size": torch.tensor([64, 64]),
        "anima_resize_interpolation": torch.tensor(repa_interpolation_code("bicubic")),
    }
    with pytest.raises(ValueError, match="resize interpolation"):
        validate_repa_sidecar(sidecar, "sample.safetensors", "tokens", (64, 64), "lanczos")


def test_repa_cache_tool_supports_documented_direct_entrypoint():
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repository_root / "tools/cache_anima_repa_features.py"), "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_controlnet_repa_configuration_reaches_delegate():
    dataset = object.__new__(ControlNetDataset)
    delegate = Namespace(repa_feature_suffix=None, repa_feature_key=None, subsets=[])
    dataset.dreambooth_dataset_delegate = delegate

    dataset.enable_repa_features("_features.safetensors", "tokens")

    assert delegate.repa_feature_suffix == "_features.safetensors"
    assert delegate.repa_feature_key == "tokens"
    assert dataset.is_repa_feature_compatible()


def _attach_lora(projection: torch.nn.Linear, name: str) -> LoRAModule:
    adapter = LoRAModule(name, projection, lora_dim=2, alpha=2)
    torch.nn.init.normal_(adapter.lora_down.weight)
    torch.nn.init.normal_(adapter.lora_up.weight)
    adapter.apply_to()
    adapter.eval()
    return adapter


@pytest.mark.parametrize("is_self_attention", [True, False])
def test_projection_fusion_matches_separate_qkv_with_standard_lora_keys(is_self_attention):
    torch.manual_seed(123)
    attention = anima_models.Attention(8, None if is_self_attention else 6, n_heads=2, head_dim=4)
    adapters = [
        _attach_lora(attention.q_proj, "q"),
        _attach_lora(attention.k_proj, "k"),
        _attach_lora(attention.v_proj, "v"),
    ]
    x = torch.randn(2, 5, 8)
    context = None if is_self_attention else torch.randn(2, 7, 6)

    expected = attention.compute_qkv(x, context)
    assert attention.enable_projection_fusion()
    actual = attention.compute_qkv(x, context)

    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-6)
    assert all(key in attention.state_dict() for key in ("q_proj.weight", "k_proj.weight", "v_proj.weight"))
    assert adapters  # hold strong references while projection weakrefs are active


def test_activation_memory_budget_validation():
    with pytest.raises(ValueError):
        compile_utils.set_activation_memory_budget(1.1)


def test_dynamic_sequence_compile_marks_anima_grid(monkeypatch):
    compile_calls = []
    marked_axes = []

    def fake_compile(module, **kwargs):
        compile_calls.append(kwargs)
        return module

    def fake_mark_dynamic(tensor, axis, min=None, max=None):
        marked_axes.append((axis, min, max))

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch._dynamo, "mark_dynamic", fake_mark_dynamic)
    args = Namespace(
        compile_backend="inductor",
        compile_mode="default",
        compile_dynamic=None,
        compile_fullgraph=False,
        compile_cache_size_limit=None,
        compile_dynamic_sequence=True,
        compile_dynamic_sequence_min_tokens=8,
        compile_dynamic_sequence_max_tokens=64,
        activation_memory_budget=None,
    )
    blocks = torch.nn.ModuleList([torch.nn.Identity()])
    compile_utils.compile_transformer(args, torch.nn.Identity(), [blocks], disable_linear=False)
    blocks[0](torch.randn(1, 1, 4, 4, 8))

    assert compile_calls[0]["dynamic"] is True
    assert marked_axes == [(1, 1, 64), (2, 1, 64), (3, 1, 64)]


def test_unsloth_checkpoint_recompute_reuses_forward_rng():
    torch.manual_seed(1234)
    x = torch.ones(32, requires_grad=True)

    output = anima_models.unsloth_checkpoint(lambda value: value * torch.rand_like(value), x)
    output.sum().backward()

    # For x=1, both the forward output and d(output)/dx are the sampled mask.
    torch.testing.assert_close(x.grad, output.detach())
