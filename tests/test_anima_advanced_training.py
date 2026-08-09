import torch

from library.anima_advanced_training import (
    relational_repa_loss,
    transported_self_flow_input,
)


def test_relational_repa_is_zero_for_same_relations_across_feature_dimensions():
    torch.manual_seed(5)
    # Four patch tokens in a 2x2 Anima grid. An orthogonal embedding into a
    # larger target space preserves all pairwise cosine relations.
    captured = torch.randn(2, 1, 2, 2, 3)
    projected = torch.nn.functional.pad(captured.reshape(2, 4, 3), (0, 2))
    cls = torch.zeros(2, 1, 5)
    features = torch.cat([cls, projected], dim=1)
    loss = relational_repa_loss(
        captured,
        features,
        (4, 4),
        patch_size=2,
        has_cls_token=True,
        use_dog=False,
        max_tokens=None,
    )
    assert loss < 1e-10


def test_repa_dog_path_is_finite_and_differentiable():
    captured = torch.randn(1, 1, 4, 4, 8, requires_grad=True)
    features = torch.randn(1, 17, 12)
    loss = relational_repa_loss(captured, features, (8, 8), use_dog=True, max_tokens=8)
    loss.backward()
    assert torch.isfinite(loss)
    assert captured.grad is not None and torch.isfinite(captured.grad).all()


def test_self_flow_transport_follows_rectified_flow_line():
    noisy = torch.tensor([[[[3.0]]], [[[5.0]]]])
    velocity = torch.tensor([[[[2.0]]], [[[4.0]]]])
    transported, sigma_next = transported_self_flow_input(noisy, velocity, torch.tensor([0.8, 0.1]), 0.25)
    torch.testing.assert_close(sigma_next, torch.tensor([0.55, 0.0]))
    torch.testing.assert_close(transported, torch.tensor([[[[2.5]]], [[[4.6]]]]))
