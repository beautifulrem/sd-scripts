import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from library import (
    anima_args,
    anima_flow_matching,
    anima_loss,
    anima_prompt_utils,
    anima_train_utils,
    config_util,
)


def _sampling_args(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        timestep_sampling=mode,
        sigmoid_scale=1.3,
        discrete_flow_shift=2.5,
        weighting_scheme="none",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        ip_noise_gamma=0.0,
        ip_noise_gamma_random_strength=False,
        min_timestep=None,
        max_timestep=None,
    )


def test_flow_scheduler_applies_shift():
    scheduler = anima_flow_matching.AnimaFlowMatchScheduler(num_train_timesteps=4, shift=2.0)
    base = torch.tensor([1.0, 0.75, 0.5, 0.25])
    expected = 2.0 * base / (1.0 + base)

    assert torch.allclose(scheduler.sigmas, expected)
    assert torch.allclose(scheduler.timesteps, expected * 4)


@pytest.mark.parametrize("mode", ["uniform", "sigmoid", "shift", "flux_shift", "sigma"])
def test_flow_sampling_produces_consistent_mixture(mode):
    args = _sampling_args(mode)
    scheduler = anima_flow_matching.AnimaFlowMatchScheduler(shift=args.discrete_flow_shift)
    latents = torch.zeros(3, 2, 4, 6)
    noise = torch.ones_like(latents)

    torch.manual_seed(7)
    noisy, timesteps, sigmas = anima_flow_matching.get_noisy_model_input_and_timesteps(
        args, scheduler, latents, noise, torch.device("cpu"), torch.float32
    )

    assert noisy.shape == latents.shape
    assert timesteps.shape == (latents.shape[0],)
    assert sigmas.shape == (latents.shape[0], 1, 1, 1)
    assert torch.all((timesteps >= 0) & (timesteps <= 1000))
    assert torch.allclose(noisy, sigmas.expand_as(noisy))


def test_flow_offset_moves_sigmoid_samples_later():
    args = _sampling_args("sigmoid")
    scheduler = anima_flow_matching.AnimaFlowMatchScheduler()
    latents = torch.zeros(8, 1, 2, 2)
    noise = torch.ones_like(latents)

    torch.manual_seed(11)
    _, base, _ = anima_flow_matching.get_noisy_model_input_and_timesteps(
        args, scheduler, latents, noise, "cpu", torch.float32
    )
    torch.manual_seed(11)
    _, shifted, _ = anima_flow_matching.get_noisy_model_input_and_timesteps(
        args,
        scheduler,
        latents,
        noise,
        "cpu",
        torch.float32,
        timestep_sampling_offset=torch.full((8,), 0.5),
    )

    assert torch.all(shifted > base)


@pytest.mark.parametrize("mode", ["uniform", "sigmoid", "shift", "flux_shift", "sigma"])
def test_flow_sampling_honors_timestep_range(mode):
    args = _sampling_args(mode)
    args.min_timestep = 200
    args.max_timestep = 400
    scheduler = anima_flow_matching.AnimaFlowMatchScheduler(shift=args.discrete_flow_shift)
    latents = torch.zeros(64, 1, 2, 2)
    noise = torch.ones_like(latents)

    _, timesteps, sigmas = anima_flow_matching.get_noisy_model_input_and_timesteps(
        args, scheduler, latents, noise, "cpu", torch.float32
    )

    assert torch.all((timesteps >= 200) & (timesteps <= 400))
    torch.testing.assert_close(sigmas.flatten(), timesteps / 1000)


def test_flow_sampling_supports_fixed_validation_timestep():
    args = _sampling_args("sigmoid")
    args.min_timestep = args.max_timestep = 375
    scheduler = anima_flow_matching.AnimaFlowMatchScheduler(shift=args.discrete_flow_shift)
    latents = torch.zeros(3, 1, 2, 2)
    noise = torch.ones_like(latents)

    noisy, timesteps, sigmas = anima_flow_matching.get_noisy_model_input_and_timesteps(
        args, scheduler, latents, noise, "cpu", torch.float32
    )

    torch.testing.assert_close(timesteps, torch.full((3,), 375.0))
    torch.testing.assert_close(sigmas.flatten(), torch.full((3,), 0.375))
    torch.testing.assert_close(noisy, sigmas.expand_as(noisy))


def test_sigma_density_closed_endpoint_does_not_overrun_scheduler(monkeypatch):
    args = _sampling_args("sigma")
    scheduler = anima_flow_matching.AnimaFlowMatchScheduler(shift=args.discrete_flow_shift)
    monkeypatch.setattr(
        anima_flow_matching,
        "_compute_timestep_density",
        lambda _scheme, batch_size, *_args: torch.ones(batch_size),
    )

    _, timesteps, sigmas = anima_flow_matching.get_noisy_model_input_and_timesteps(
        args, scheduler, torch.zeros(2, 1, 2, 2), torch.ones(2, 1, 2, 2), "cpu", torch.float32
    )

    assert torch.isfinite(timesteps).all()
    assert torch.isfinite(sigmas).all()


def test_anima_loss_weighting_and_huber_thresholds():
    sigmas = torch.tensor([0.25, 0.5])
    assert torch.allclose(anima_train_utils.compute_loss_weighting_for_anima("sigma_sqrt", sigmas), sigmas**-2)
    assert torch.equal(anima_train_utils.compute_loss_weighting_for_anima("none", sigmas), torch.ones_like(sigmas))

    scheduler = anima_flow_matching.AnimaFlowMatchScheduler()
    args = SimpleNamespace(loss_type="huber", huber_schedule="constant", huber_c=0.1, huber_scale=2.0)
    threshold = anima_loss.get_huber_threshold_if_needed(args, torch.tensor([10.0, 20.0]), scheduler)
    assert torch.equal(threshold, torch.tensor([0.2, 0.2]))

    prediction = torch.tensor([[[[1.0]]], [[[2.0]]]])
    target = torch.zeros_like(prediction)
    loss = anima_loss.conditional_loss(prediction, target, "huber", "none", threshold)
    assert loss.shape == prediction.shape
    assert torch.isfinite(loss).all()

    args.huber_schedule = "exponential"
    threshold = anima_loss.get_huber_threshold_if_needed(
        args, torch.tensor([0.0, 500.0, 1000.0]), scheduler
    )
    torch.testing.assert_close(threshold, torch.tensor([2.0, 2.0 * 0.1**0.5, 0.2]))


def test_reduce_weighted_loss_keeps_weights_paired_with_samples():
    elementwise = torch.tensor([1.0, 3.0]).view(2, 1, 1, 1)
    timestep_weights = torch.tensor([2.0, 5.0]).view(2, 1, 1, 1)
    sample_weights = torch.tensor([7.0, 11.0])

    per_sample = anima_loss.reduce_weighted_loss(elementwise, timestep_weights, sample_weights)

    torch.testing.assert_close(per_sample, torch.tensor([14.0, 165.0]))


def test_masked_loss_prefers_explicit_alpha_mask_over_control_image():
    loss = torch.ones(1, 1, 2, 2)
    batch = {
        "conditioning_images": -torch.ones(1, 3, 2, 2),
        "alpha_masks": torch.ones(1, 2, 2),
    }
    torch.testing.assert_close(anima_loss.apply_masked_loss(loss, batch), loss)


def test_masked_loss_can_reject_control_image_as_mask_source():
    loss = torch.ones(1, 1, 2, 2)
    batch = {"conditioning_images": -torch.ones(1, 3, 16, 16)}

    masked = anima_loss.apply_masked_loss(loss, batch, conditioning_image_is_mask=False)

    assert torch.equal(masked, loss)


def test_target_alpha_normalization_is_explicit_and_per_sample():
    loss = torch.ones(2, 1, 2, 2)
    alpha_masks = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ]
    )

    unnormalized = anima_loss.apply_masked_loss(loss, {"alpha_masks": alpha_masks})
    normalized = anima_loss.apply_masked_loss(loss, {"alpha_masks": alpha_masks}, normalize=True)

    torch.testing.assert_close(unnormalized.mean(dim=(1, 2, 3)), torch.tensor([1.0, 0.25]))
    torch.testing.assert_close(normalized.mean(dim=(1, 2, 3)), torch.ones(2))


def test_target_alpha_normalization_keeps_empty_masks_finite():
    normalized = anima_loss.apply_masked_loss(
        torch.ones(1, 1, 2, 2),
        {"alpha_masks": torch.zeros(1, 2, 2)},
        normalize=True,
    )

    assert torch.isfinite(normalized).all()
    assert torch.count_nonzero(normalized) == 0


def test_target_alpha_normalization_cli_is_opt_in():
    parser = argparse.ArgumentParser()
    anima_args.add_masked_loss_arguments(parser)

    assert parser.parse_args([]).normalize_alpha_mask_loss is False
    assert parser.parse_args(["--normalize_alpha_mask_loss"]).normalize_alpha_mask_loss is True


def test_controlnet_toml_alpha_mask_reaches_dataset_delegate(tmp_path: Path):
    train_dir = tmp_path / "train"
    control_dir = tmp_path / "control"
    train_dir.mkdir()
    control_dir.mkdir()
    Image.new("RGBA", (8, 8), color=(100, 120, 140, 128)).save(train_dir / "sample.png")
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(control_dir / "sample.png")

    sanitizer = config_util.ConfigSanitizer(True, True, True, True)
    blueprint = config_util.BlueprintGenerator(sanitizer).generate(
        {
            "datasets": [
                {
                    "resolution": [8, 8],
                    "subsets": [
                        {
                            "image_dir": str(train_dir),
                            "conditioning_data_dir": str(control_dir),
                            "alpha_mask": True,
                        }
                    ],
                }
            ]
        },
        argparse.Namespace(),
    )
    dataset_group, _ = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    assert dataset_group.datasets[0].subsets[0].alpha_mask is True


def test_prompt_file_parser_keeps_anima_sampling_fields(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(
        "# comment\nportrait --w 768 --h 512 --s 24 --d 42 --l 4.5 --fs 3.0 --am 0.8,1.2\n",
        encoding="utf-8",
    )

    prompts = anima_prompt_utils.load_prompts(str(prompt_file))

    assert prompts == [
        {
            "prompt": "portrait",
            "width": 768,
            "height": 512,
            "sample_steps": 24,
            "seed": 42,
            "scale": 4.5,
            "flow_shift": 3.0,
            "additional_network_multiplier": [0.8, 1.2],
            "enum": 0,
        }
    ]
