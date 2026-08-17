import argparse
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from anima_train_control_net_lllite import add_anima_lllite_arguments
from networks.control_net_lllite_anima import (
    LATENT_COND_CHANNELS,
    ControlNetLLLiteDiT,
    build_cond_tensors,
    load_lllite_weights,
    save_lllite_model,
)
from tools.dev.manual_test_anima_lllite_dryrun import main
from tools.dev.run_anima_lllite_cond_ab import build_command, validate_shared_config


class Attention(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.is_selfattn = True
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.output_proj = nn.Linear(dim, dim, bias=False)


class _DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()


class _DummyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock()])


class _FakeVAE:
    device = torch.device("cpu")
    dtype = torch.float32

    def encode_pixels_to_latents(self, pixels):
        pooled = F.avg_pool2d(pixels, 8)
        return pooled.repeat(1, 6, 1, 1)[:, :LATENT_COND_CHANNELS]


def test_anima_lllite_end_to_end_dryrun():
    main()


def test_lllite_conditioning_cli_defaults_to_pixel_and_allows_latent():
    parser = argparse.ArgumentParser()
    add_anima_lllite_arguments(parser)

    assert parser.parse_args([]).lllite_cond_input == "pixel"
    assert parser.parse_args(["--lllite_cond_input", "latent"]).lllite_cond_input == "latent"


def test_build_cond_tensors_preserves_pixel_default_and_encodes_latent():
    torch.manual_seed(17)
    rgb = torch.randn(2, 3, 128, 96).clamp(-1, 1)

    pixel, pixel_mask = build_cond_tensors(rgb)
    latent, latent_mask = build_cond_tensors(rgb, cond_input_space="latent", vae=_FakeVAE())

    assert pixel is rgb
    assert pixel_mask is None
    assert latent.shape == (2, LATENT_COND_CHANNELS, 16, 12)
    assert latent_mask is None


@pytest.mark.parametrize(
    ("input_space", "cond_shape"),
    [
        pytest.param("pixel", (1, 3, 128, 128), id="pixel"),
        pytest.param("latent", (1, LATENT_COND_CHANNELS, 16, 16), id="latent"),
    ],
)
def test_lllite_pixel_and_latent_stems_reach_same_token_grid(input_space, cond_shape):
    lllite = ControlNetLLLiteDiT(
        _DummyDiT(),
        cond_emb_dim=16,
        mlp_dim=16,
        cond_dim=16,
        cond_resblocks=0,
        cond_input_space=input_space,
    )

    lllite.set_cond_image(torch.randn(cond_shape))

    assert lllite.lllite_modules[0].cond_emb.shape == (1, 64, 16)


def test_lllite_checkpoint_rejects_pixel_latent_mismatch(tmp_path):
    pixel = ControlNetLLLiteDiT(_DummyDiT(), cond_dim=16, cond_resblocks=0, cond_input_space="pixel")
    checkpoint = tmp_path / "pixel.safetensors"
    save_lllite_model(str(checkpoint), pixel, metadata={"lllite.cond_input_space": "pixel"})
    latent = ControlNetLLLiteDiT(_DummyDiT(), cond_dim=16, cond_resblocks=0, cond_input_space="latent")

    with pytest.raises(RuntimeError, match="cond input space mismatch"):
        load_lllite_weights(latent, str(checkpoint), strict=False)


def test_latent_inpaint_keeps_mask_separate_until_stem():
    rgb = torch.randn(1, 3, 128, 128)
    mask = torch.zeros(1, 1, 128, 128)
    mask[:, :, 32:96, 32:96] = 1
    cond_latent, cond_mask = build_cond_tensors(
        rgb,
        mask,
        cond_input_space="latent",
        cond_in_channels=4,
        inpaint_masked_input=True,
        vae=_FakeVAE(),
    )
    lllite = ControlNetLLLiteDiT(
        _DummyDiT(), cond_dim=16, cond_resblocks=0, cond_in_channels=4, cond_input_space="latent"
    )

    lllite.set_cond_image(cond_latent, cond_mask)

    assert cond_latent.shape == (1, LATENT_COND_CHANNELS, 16, 16)
    assert cond_mask.shape == (1, 1, 128, 128)
    assert set(cond_mask.unique().tolist()) == {-1.0, 1.0}
    assert lllite.lllite_modules[0].cond_emb.shape == (1, 64, 32)


def test_lllite_cuda_ab_commands_share_config_seed_and_steps(tmp_path):
    args = SimpleNamespace(
        accelerate="accelerate",
        config_file=str(tmp_path / "shared.toml"),
        output_root=str(tmp_path / "ab"),
        seed=123,
        max_train_steps=80,
        extra_args=["--", "--sample_every_n_steps", "40"],
    )

    pixel = build_command(args, "pixel")
    latent = build_command(args, "latent")

    for command in (pixel, latent):
        assert command[command.index("--seed") + 1] == "123"
        assert command[command.index("--max_train_steps") + 1] == "80"
        assert command[command.index("--config_file") + 1] == str((tmp_path / "shared.toml").resolve())
        assert "--sample_every_n_steps" in command
    assert pixel[pixel.index("--lllite_cond_input") + 1] == "pixel"
    assert latent[latent.index("--lllite_cond_input") + 1] == "latent"


def test_lllite_cuda_ab_rejects_epoch_override(tmp_path):
    config = tmp_path / "shared.toml"
    config.write_text("max_train_epochs = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="max_train_epochs"):
        validate_shared_config(config)
