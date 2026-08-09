from types import SimpleNamespace

import pytest
import torch

from library import anima_flow_matching, anima_loss, anima_prompt_utils, anima_train_utils


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
