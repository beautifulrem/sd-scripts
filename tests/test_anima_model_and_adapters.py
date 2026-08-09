import contextlib
from types import SimpleNamespace

import pytest
import torch

import anima_minimal_inference
from anima_train_network import AnimaNetworkTrainer, setup_parser
from library.anima_models import Anima, LLMAdapter
from networks import (
    chimera_lora_anima,
    easycontrol_anima,
    flow_map_lora_anima,
    hydra_lora_anima,
    loha,
    lokr,
    lora_anima,
    soft_tokens_anima,
    turbo_dmd_anima,
)


def _tiny_anima() -> Anima:
    return Anima(
        max_img_h=8,
        max_img_w=8,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=64,
        num_blocks=1,
        num_heads=4,
        mlp_ratio=2.0,
        crossattn_emb_channels=32,
        pos_emb_cls="rope3d",
        pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        use_adaln_lora=True,
        adaln_lora_dim=8,
        rope_enable_fps_modulation=False,
        use_llm_adapter=False,
        attn_mode="torch",
        split_attn=False,
    )


def _inputs():
    return torch.randn(1, 16, 1, 4, 4), torch.tensor([500.0]), torch.randn(1, 3, 32)


def test_tiny_anima_full_finetune_forward_and_backward():
    torch.manual_seed(1)
    model = _tiny_anima().train()
    output = model(*_inputs())

    assert output.shape == (1, 16, 1, 4, 4)
    assert torch.isfinite(output).all()

    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_anima_lora_forward_backward_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(2)
    model = _tiny_anima().eval().requires_grad_(False)
    inputs = _inputs()
    with torch.no_grad():
        base_output = model(*inputs)

    network = lora_anima.create_network(1.0, 4, 1.0, None, [], model)
    assert network.unet_loras
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)

    output = model(*inputs)
    assert torch.allclose(output, base_output, atol=1e-6)
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in network.parameters())

    checkpoint = tmp_path / "anima_lora.safetensors"
    network.save_weights(str(checkpoint), torch.float32, {"test": "1"})
    reloaded_model = _tiny_anima()
    reloaded, state_dict = lora_anima.create_network_from_weights(
        1.0, str(checkpoint), None, [], reloaded_model, for_inference=False
    )
    reloaded.apply_to([], reloaded_model, apply_text_encoder=False, apply_unet=True)
    incompatible = reloaded.load_state_dict(state_dict, strict=False)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys


def test_anima_lora_svd_down_is_zero_delta_and_uses_principal_input_basis():
    torch.manual_seed(7)
    original = torch.nn.Linear(6, 6, bias=False)
    original.weight.data.copy_(torch.diag(torch.tensor([9.0, 7.0, 5.0, 3.0, 2.0, 1.0])))
    module = lora_anima.LoRAModule("test", original, lora_dim=2, alpha=2, down_init="weight_svd")

    assert torch.count_nonzero(module.lora_up.weight) == 0
    gram = module.lora_down.weight.float() @ module.lora_down.weight.float().T
    assert torch.allclose(gram, torch.eye(2) / 3.0, atol=1e-4)
    principal_energy = module.lora_down.weight[:, :2].square().sum()
    remaining_energy = module.lora_down.weight[:, 2:].square().sum()
    assert principal_energy > remaining_energy

    x = torch.randn(3, 6)
    expected = original(x)
    module.apply_to()
    assert torch.allclose(original(x), expected, atol=1e-6)


def test_anima_tlora_masks_rank_by_noise_and_keeps_checkpoint_standard():
    original = torch.nn.Linear(4, 4, bias=False)
    original.weight.data.zero_()
    module = lora_anima.LoRAModule(
        "test", original, lora_dim=4, alpha=4, use_timestep_mask=True, min_rank=1, alpha_rank_scale=1.0
    )
    module.lora_down.weight.data.copy_(torch.eye(4))
    module.lora_up.weight.data.copy_(torch.eye(4))
    module.apply_to()

    module.set_timestep_mask(torch.tensor([1.0, 0.0]))
    output = original(torch.ones(2, 4))
    assert torch.equal(output[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(output[1], torch.ones(4))
    assert not any("timestep" in key for key in module.state_dict())

    module.clear_timestep_mask()
    assert torch.equal(original(torch.ones(1, 4)), torch.ones(1, 4))


def test_anima_adaln_sugar_uses_independent_rank_alpha_and_lr():
    model = _tiny_anima().requires_grad_(False)
    network = lora_anima.create_network(
        1.0,
        4,
        4.0,
        None,
        [],
        model,
        train_adaln="true",
        adaln_rank="2",
        adaln_alpha="1.5",
        adaln_lr="2e-5",
    )
    adaln_loras = [lora for lora in network.unet_loras if "adaln_modulation_" in lora.original_name]
    assert adaln_loras
    assert all(lora.lora_dim == 2 for lora in adaln_loras)
    assert all(float(lora.alpha) == 1.5 for lora in adaln_loras)

    param_groups, _ = network.prepare_optimizer_params_with_multiple_te_lrs(None, None, 1e-4)
    assert any(group.get("lr") == 2e-5 for group in param_groups)
    assert any(group.get("lr") == 1e-4 for group in param_groups)


def test_anima_channel_scaling_preserves_forward_and_exports_standard_lora(tmp_path):
    torch.manual_seed(77)
    base_a = torch.nn.Linear(4, 3, bias=False)
    base_b = torch.nn.Linear(4, 3, bias=False)
    base_b.weight.data.copy_(base_a.weight)
    plain = lora_anima.LoRAModule("plain", base_a, lora_dim=2, alpha=2)
    torch.manual_seed(77)
    scaled = lora_anima.LoRAModule(
        "scaled", base_b, lora_dim=2, alpha=2, channel_scale=torch.tensor([0.25, 1.0, 4.0, 2.0])
    )
    # Align the unscaled initialization explicitly; scaled stores down*s.
    normalized_scale = torch.tensor([0.25, 1.0, 4.0, 2.0])
    normalized_scale = normalized_scale / normalized_scale.mean()
    scaled.lora_down.weight.data.copy_(plain.lora_down.weight * normalized_scale)
    up = torch.randn_like(plain.lora_up.weight)
    plain.lora_up.weight.data.copy_(up)
    scaled.lora_up.weight.data.copy_(up)
    x = torch.randn(5, 4)
    plain.apply_to()
    scaled.apply_to()
    torch.testing.assert_close(base_a(x), base_b(x), rtol=1e-5, atol=1e-6)

    model = _tiny_anima().requires_grad_(False)
    stats_path = tmp_path / "stats.safetensors"
    from safetensors.torch import load_file, save_file

    probe = lora_anima.create_network(1.0, 2, 2.0, None, [], model)
    stats = {
        adapter.lora_name: torch.ones(adapter.lora_down.in_features)
        for adapter in probe.unet_loras
        if isinstance(adapter.lora_down, torch.nn.Linear)
    }
    save_file(stats, str(stats_path))
    network = lora_anima.create_network(
        1.0,
        2,
        2.0,
        None,
        [],
        model,
        channel_scaling_alpha="0.5",
        channel_scaling_stats=str(stats_path),
    )
    output_path = tmp_path / "scaled_lora.safetensors"
    network.save_weights(str(output_path), torch.float32, {})
    saved = load_file(str(output_path))
    assert not any(key.endswith("inv_scale") for key in saved)


def test_anima_loha_and_lokr_forward_backward():
    for adapter_module in (loha, lokr):
        torch.manual_seed(3)
        model = _tiny_anima().eval().requires_grad_(False)
        inputs = _inputs()
        with torch.no_grad():
            base_output = model(*inputs)

        network = adapter_module.create_network(1.0, 4, 1.0, None, [], model)
        assert network.unet_loras
        network.apply_to([], model, apply_text_encoder=False, apply_unet=True)

        output = model(*inputs)
        assert torch.allclose(output, base_output, atol=1e-6)
        output.square().mean().backward()
        assert any(parameter.grad is not None for parameter in network.parameters())


def test_anima_llm_adapter_forward_and_backward():
    adapter = LLMAdapter(source_dim=32, target_dim=32, model_dim=32, num_layers=1, num_heads=4, self_attn=True)
    source = torch.randn(2, 5, 32)
    token_ids = torch.randint(0, 128, (2, 4))
    source_mask = torch.ones(2, 5, dtype=torch.bool)
    target_mask = torch.ones(2, 4, dtype=torch.bool)

    output = adapter(source, token_ids, target_attention_mask=target_mask, source_attention_mask=source_mask)

    assert output.shape == (2, 4, 32)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in adapter.parameters())


def test_anima_soft_tokens_are_layer_and_timestep_conditioned(tmp_path):
    model = _tiny_anima().requires_grad_(False)
    network = soft_tokens_anima.create_network(
        1.0, 2, None, None, [], model, num_tokens="2", num_timestep_bins="4"
    )
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    network.set_step_sigmas(torch.tensor([0.75]))
    output = model(*_inputs())
    output.square().mean().backward()
    assert network.soft_tokens.grad is not None
    # sigma=.75 selects the final bin while other bins stay untouched.
    assert network.soft_tokens.grad[0, 3].abs().sum() > 0
    assert network.soft_tokens.grad[0, :3].abs().sum() == 0

    checkpoint = tmp_path / "soft_tokens.safetensors"
    network.save_weights(str(checkpoint), torch.float32, {"adapter": "soft_tokens"})
    restored_model = _tiny_anima()
    restored, state = soft_tokens_anima.create_network_from_weights(
        1.0, str(checkpoint), None, [], restored_model
    )
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.soft_tokens, network.soft_tokens)


def test_anima_hydra_lora_full_and_standard_mean_exports(tmp_path):
    model = _tiny_anima().requires_grad_(False)
    inputs = _inputs()
    with torch.no_grad():
        base_output = model(*inputs)
    network = hydra_lora_anima.create_network(
        1.0, 2, 2.0, None, [], model, num_experts="3", balance_weight="0.1", orthogonal_weight="0.1"
    )
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    output = model(*inputs)
    torch.testing.assert_close(output, base_output, rtol=1e-5, atol=1e-6)
    (output.square().mean() + network.get_auxiliary_loss()).backward()
    assert any(module.router.weight.grad is not None and module.router.weight.grad.abs().sum() > 0 for module in network.unet_loras)

    full_path = tmp_path / "hydra_full.safetensors"
    network.save_weights(str(full_path), torch.float32, {})
    from safetensors.torch import load_file

    full = load_file(str(full_path))
    assert any(".expert_ups." in key for key in full)
    assert any(".router." in key for key in full)

    network.export_mode = "standard_mean"
    standard_path = tmp_path / "hydra_standard.safetensors"
    network.save_weights(str(standard_path), torch.float32, {})
    standard = load_file(str(standard_path))
    assert not any(".expert_ups." in key or ".router." in key for key in standard)
    assert any(key.endswith(".lora_down.weight") for key in standard)


def test_anima_chimera_dual_pool_and_rank_concat_export(tmp_path):
    model = _tiny_anima().requires_grad_(False)
    network = chimera_lora_anima.create_network(
        1.0, 2, 2.0, None, [], model, num_experts="2", num_frequency_experts="3"
    )
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    network.set_frequency_context(torch.tensor([0.6]), torch.randn(1, 16, 1, 4, 4))
    output = model(*_inputs())
    (output.square().mean() + network.get_auxiliary_loss()).backward()
    assert any(
        module.frequency_router.weight.grad is not None and module.frequency_router.weight.grad.abs().sum() > 0
        for module in network.unet_loras
    )

    network.export_mode = "standard_mean"
    path = tmp_path / "chimera_standard.safetensors"
    network.save_weights(str(path), torch.float32, {})
    from safetensors.torch import load_file

    state = load_file(str(path))
    assert not any("frequency_" in key or "expert_ups" in key or ".router." in key for key in state)
    first_down = next(value for key, value in state.items() if key.endswith(".lora_down.weight"))
    assert first_down.shape[0] == 4  # two rank-2 pools concatenate into standard rank 4


def test_anima_turbo_dmd_saves_standard_generator_and_resume_critic(tmp_path):
    model = _tiny_anima().requires_grad_(False)
    network = turbo_dmd_anima.create_network(1.0, 2, 2.0, None, [], model, down_init="kaiming")
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    network.set_adapter_mode("critic")
    model(*_inputs()).square().mean().backward()
    assert any(module.critic_up.weight.grad is not None for module in network.unet_loras)

    path = tmp_path / "turbo.safetensors"
    network.save_weights(str(path), torch.float32, {})
    critic_path = tmp_path / "turbo_dmd_critic.safetensors"
    assert critic_path.is_file()
    from safetensors.torch import load_file

    assert not any("critic_" in key for key in load_file(str(path)))
    assert all("critic_" in key for key in load_file(str(critic_path)))

    restored_model = _tiny_anima().requires_grad_(False)
    restored, state = turbo_dmd_anima.create_network_from_weights(1.0, str(path), None, [], restored_model)
    restored.apply_to([], restored_model, apply_text_encoder=False, apply_unet=True)
    incompatible = restored.load_state_dict(state, strict=False)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys

    trainer = AnimaNetworkTrainer()
    trainer._remove_model(
        args=SimpleNamespace(output_dir=str(tmp_path)),
        accelerator=SimpleNamespace(print=lambda *args, **kwargs: None),
        old_ckpt_name=path.name,
        unwrapped_nw=restored,
    )
    assert not path.exists()
    assert not critic_path.exists()


def test_tlora_checkpoint_recompute_restores_forward_rank_mask():
    torch.manual_seed(91)
    model = _tiny_anima().train().requires_grad_(False)
    network = lora_anima.create_network(
        1.0,
        4,
        4.0,
        None,
        [],
        model,
        use_timestep_mask="true",
        min_rank="1",
    )
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    for module in network.unet_loras:
        torch.nn.init.normal_(module.lora_up.weight, std=0.02)
    inputs = _inputs()

    model.disable_gradient_checkpointing()
    network.set_timestep_mask(torch.tensor([1.0]))
    model(*inputs).square().mean().backward()
    expected = {name: parameter.grad.detach().clone() for name, parameter in network.named_parameters()}
    network.zero_grad(set_to_none=True)
    network.clear_timestep_mask()

    model.enable_gradient_checkpointing()
    network.set_timestep_mask(torch.tensor([1.0]))
    output = model(*inputs)
    network.clear_timestep_mask()  # simulate trainer cleanup before backward recomputation
    output.square().mean().backward()
    for name, parameter in network.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name], rtol=1e-5, atol=1e-6)


def test_fullgraph_checkpointing_rejects_context_dependent_adapter():
    args = setup_parser().parse_args([])
    args.compile = True
    args.compile_fullgraph = True
    args.gradient_checkpointing = True
    args.network_module = "networks.lora_anima"
    args.network_args = ["use_timestep_mask=true"]
    dataset = SimpleNamespace(verify_bucket_reso_steps=lambda _step: None)

    with pytest.raises(ValueError, match="compile_fullgraph"):
        AnimaNetworkTrainer().assert_extra_args(args, dataset, None)


def test_turbo_dmd_checkpoint_recompute_restores_critic_mode():
    torch.manual_seed(92)
    model = _tiny_anima().train().requires_grad_(False)
    model.enable_gradient_checkpointing()
    network = turbo_dmd_anima.create_network(1.0, 2, 2.0, None, [], model, down_init="kaiming")
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    for module in network.unet_loras:
        torch.nn.init.normal_(module.critic_up.weight, std=0.02)
    network.set_adapter_mode("critic")
    output = model(*_inputs())
    network.set_adapter_mode("generator")
    output.square().mean().backward()
    assert any(module.critic_up.weight.grad is not None for module in network.unet_loras)
    assert all(module.lora_up.weight.grad is None for module in network.unet_loras)


def test_easycontrol_checkpoint_recompute_restores_condition_latents():
    torch.manual_seed(93)
    model = _tiny_anima().train().requires_grad_(False)
    model.enable_gradient_checkpointing()
    network = easycontrol_anima.create_network(1.0, 4, None, None, [], model)
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    network.gates.data.fill_(0.5)
    network.set_condition_latents(torch.randn(1, 16, 4, 4))
    output = model(*_inputs())
    network.clear_condition_latents()
    output.square().mean().backward()
    assert network.condition_patch.weight.grad is not None
    assert network.condition_patch.weight.grad.abs().sum() > 0


def test_anima_flow_map_interval_conditioner_and_checkpoint(tmp_path):
    model = _tiny_anima().requires_grad_(False)
    network = flow_map_lora_anima.create_network(
        1.0, 2, 2.0, None, [], model, down_init="kaiming", interval_hidden_dim="16"
    )
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    network.set_flow_interval(torch.tensor([0.8]), torch.tensor([0.2]))
    model(*_inputs()).square().mean().backward()
    assert network.interval_embedder[-1].weight.grad is not None

    path = tmp_path / "flow_map.safetensors"
    network.save_weights(str(path), torch.float32, {})
    restored_model = _tiny_anima().requires_grad_(False)
    restored, state = flow_map_lora_anima.create_network_from_weights(1.0, str(path), None, [], restored_model)
    restored.apply_to([], restored_model, apply_text_encoder=False, apply_unet=True)
    incompatible = restored.load_state_dict(state, strict=False)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys


def test_minimal_inference_loads_full_flow_map_adapter(tmp_path, monkeypatch):
    source_model = _tiny_anima().requires_grad_(False)
    source_network = flow_map_lora_anima.create_network(
        1.0, 2, 2.0, None, [], source_model, down_init="kaiming", interval_hidden_dim="16"
    )
    source_network.apply_to([], source_model, apply_text_encoder=False, apply_unet=True)
    path = tmp_path / "flow_map.safetensors"
    source_network.save_weights(str(path), torch.float32, {})

    inference_model = _tiny_anima().requires_grad_(False)
    monkeypatch.setattr(anima_minimal_inference.anima_utils, "load_anima_model", lambda *args, **kwargs: inference_model)
    args = SimpleNamespace(
        lora_weight=None,
        lora_multiplier=1.0,
        fp8_scaled=False,
        fp8=False,
        dit="unused",
        attn_mode="torch",
        adapter_module="networks.flow_map_lora_anima",
        adapter_weight=str(path),
        adapter_multiplier=0.75,
    )

    loaded = anima_minimal_inference.load_dit_model(args, torch.device("cpu"), torch.float32)

    adapter = loaded._anima_external_adapter
    assert isinstance(adapter, flow_map_lora_anima.FlowMapLoRANetwork)
    assert loaded._anima_flow_map_network() is adapter
    assert adapter.multiplier == 0.75


def test_anima_easycontrol_is_zero_gated_then_conditions_output(tmp_path):
    model = _tiny_anima().eval().requires_grad_(False)
    inputs = _inputs()
    with torch.no_grad():
        base = model(*inputs)
    network = easycontrol_anima.create_network(1.0, 4, None, None, [], model)
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    network.set_condition_latents(torch.randn(1, 16, 4, 4))
    initial = model(*inputs)
    torch.testing.assert_close(initial, base, rtol=1e-5, atol=1e-6)
    initial.square().mean().backward()
    assert network.gates.grad is not None

    network.gates.data.fill_(0.5)
    with torch.no_grad():
        controlled = model(*inputs)
    assert not torch.allclose(controlled, base)

    path = tmp_path / "easycontrol.safetensors"
    network.save_weights(str(path), torch.float32, {})
    assert path.is_file()


def test_anima_network_trainer_noise_prediction_path():
    model = _tiny_anima()
    model.llm_adapter = LLMAdapter(source_dim=32, target_dim=32, model_dim=32, num_layers=1, num_heads=4, self_attn=True)
    args = SimpleNamespace(
        timestep_sampling="shift",
        sigmoid_scale=1.0,
        discrete_flow_shift=2.0,
        weighting_scheme="none",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        ip_noise_gamma=0.0,
        ip_noise_gamma_random_strength=False,
        gradient_checkpointing=False,
    )
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=contextlib.nullcontext)
    latents = torch.randn(2, 16, 4, 4)
    text_conditions = [
        torch.randn(2, 5, 32),
        torch.ones(2, 5, dtype=torch.bool),
        torch.randint(0, 128, (2, 4)),
        torch.ones(2, 4, dtype=torch.bool),
    ]
    trainer = AnimaNetworkTrainer()
    scheduler = trainer.get_noise_scheduler(args, torch.device("cpu"))

    prediction, target, timesteps, weighting = trainer.get_noise_pred_and_target(
        args,
        accelerator,
        scheduler,
        latents,
        {"custom_attributes": [{}, {}]},
        text_conditions,
        model,
        None,
        torch.float32,
        True,
        is_train=True,
    )

    assert prediction.shape == target.shape == latents.shape
    assert timesteps.shape == (2,)
    assert weighting.shape == (2, 1, 1, 1)
    assert all(torch.isfinite(tensor).all() for tensor in (prediction, target, timesteps, weighting))

    (prediction - target).square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_anima_dp_dmd_training_path_updates_generator_and_critic():
    model = _tiny_anima().requires_grad_(False)
    model.llm_adapter = LLMAdapter(source_dim=32, target_dim=32, model_dim=32, num_layers=1, num_heads=4, self_attn=True)
    model.llm_adapter.requires_grad_(False)
    network = turbo_dmd_anima.create_network(1.0, 2, 2.0, None, [], model, down_init="kaiming")
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    args = SimpleNamespace(
        timestep_sampling="shift",
        sigmoid_scale=1.0,
        discrete_flow_shift=2.0,
        weighting_scheme="none",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        ip_noise_gamma=0.0,
        ip_noise_gamma_random_strength=False,
        gradient_checkpointing=False,
        repa_weight=0.0,
        self_flow_weight=0.0,
        dp_dmd_weight=0.1,
        dp_dmd_critic_weight=1.0,
        dp_dmd_steps=2,
    )
    accelerator = SimpleNamespace(
        device=torch.device("cpu"), autocast=contextlib.nullcontext, unwrap_model=lambda value: value
    )
    latents = torch.randn(1, 16, 4, 4)
    text_conditions = [
        torch.randn(1, 5, 32),
        torch.ones(1, 5, dtype=torch.bool),
        torch.randint(0, 128, (1, 4)),
        torch.ones(1, 4, dtype=torch.bool),
    ]
    trainer = AnimaNetworkTrainer()
    scheduler = trainer.get_noise_scheduler(args, torch.device("cpu"))
    prediction, target, _, _ = trainer.get_noise_pred_and_target(
        args,
        accelerator,
        scheduler,
        latents,
        {"custom_attributes": [{}]},
        text_conditions,
        model,
        network,
        torch.float32,
        True,
        is_train=True,
    )
    loss = torch.nn.functional.mse_loss(prediction, target) + trainer._advanced_aux_loss
    loss.backward()
    assert any(module.lora_up.weight.grad is not None for module in network.unet_loras)
    assert any(module.critic_up.weight.grad is not None for module in network.unet_loras)


def test_anima_anyflow_training_path_updates_interval_and_lora():
    model = _tiny_anima().requires_grad_(False)
    model.llm_adapter = LLMAdapter(source_dim=32, target_dim=32, model_dim=32, num_layers=1, num_heads=4, self_attn=True)
    model.llm_adapter.requires_grad_(False)
    network = flow_map_lora_anima.create_network(1.0, 2, 2.0, None, [], model, down_init="kaiming")
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    args = SimpleNamespace(
        timestep_sampling="shift",
        sigmoid_scale=1.0,
        discrete_flow_shift=2.0,
        weighting_scheme="none",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        ip_noise_gamma=0.0,
        ip_noise_gamma_random_strength=False,
        gradient_checkpointing=False,
        repa_weight=0.0,
        self_flow_weight=0.0,
        dp_dmd_weight=0.0,
        anyflow_weight=0.2,
        anyflow_teacher_steps=1,
        anyflow_min_interval=0.05,
    )
    accelerator = SimpleNamespace(
        device=torch.device("cpu"), autocast=contextlib.nullcontext, unwrap_model=lambda value: value
    )
    trainer = AnimaNetworkTrainer()
    latents = torch.randn(1, 16, 4, 4)
    conditions = [
        torch.randn(1, 5, 32),
        torch.ones(1, 5, dtype=torch.bool),
        torch.randint(0, 128, (1, 4)),
        torch.ones(1, 4, dtype=torch.bool),
    ]
    prediction, target, _, _ = trainer.get_noise_pred_and_target(
        args,
        accelerator,
        trainer.get_noise_scheduler(args, torch.device("cpu")),
        latents,
        {"custom_attributes": [{}]},
        conditions,
        model,
        network,
        torch.float32,
        True,
    )
    (torch.nn.functional.mse_loss(prediction, target) + trainer._advanced_aux_loss).backward()
    assert network.interval_embedder[-1].weight.grad is not None
    assert any(module.lora_up.weight.grad is not None for module in network.unet_loras)
