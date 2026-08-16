from __future__ import annotations

import pytest
import torch

from openhomie_isaaclab import elf3_constants as C
from openhomie_isaaclab.him_rl import HIMEstimator, extract_estimator_targets, sinkhorn


def make_estimator() -> HIMEstimator:
    return HIMEstimator(
        history_dim=C.num_actor_obs(),
        one_step_obs_dim=C.num_one_step_actor_obs(),
        latent_dim=C.NUM_ESTIMATOR_LATENT,
        encoder_hidden_dims=(24, 16),
        target_hidden_dims=(24, 16),
        num_prototypes=16,
    )


def test_sinkhorn_is_deterministic_finite_and_balanced():
    logits = torch.tensor(
        [[1000.0, -1000.0, 500.0, -500.0], [-800.0, 900.0, -700.0, 600.0],
         [750.0, 700.0, -950.0, -900.0], [-600.0, -650.0, 850.0, 800.0]]
    )
    first = sinkhorn(logits, epsilon=0.05, iterations=20)
    second = sinkhorn(logits, epsilon=0.05, iterations=20)
    assert torch.equal(first, second)
    assert first.shape == logits.shape
    assert torch.isfinite(first).all() and torch.all(first >= 0)
    assert torch.allclose(first.sum(1), torch.ones(logits.shape[0]), atol=1.0e-4)

    reachable = sinkhorn(logits / 1000.0, epsilon=0.05, iterations=30)
    expected_columns = torch.full((logits.shape[1],), logits.shape[0] / logits.shape[1])
    assert torch.allclose(reachable.sum(1), torch.ones(logits.shape[0]), atol=1.0e-4)
    assert torch.allclose(reachable.sum(0), expected_columns, atol=1.0e-3)


@pytest.mark.parametrize(
    ("logits", "epsilon", "iterations", "match"),
    [
        (torch.zeros(2, 3, 1), 0.05, 3, "rank-2"),
        (torch.zeros(0, 3), 0.05, 3, "at least one"),
        (torch.zeros(2, 3), 0.0, 3, "positive"),
        (torch.zeros(2, 3), 0.05, 0, "at least one"),
    ],
)
def test_sinkhorn_rejects_invalid_inputs(logits, epsilon, iterations, match):
    with pytest.raises(ValueError, match=match):
        sinkhorn(logits, epsilon=epsilon, iterations=iterations)


def test_estimator_shapes_target_layout_losses_and_gradients():
    torch.manual_seed(1)
    estimator = make_estimator()
    batch = 7
    history = torch.randn(batch, C.num_actor_obs())
    critic = torch.arange(batch * C.num_critic_obs(), dtype=torch.float32).reshape(
        batch, C.num_critic_obs()
    )
    velocity, latent = estimator(history)
    target_velocity, target_obs = estimator.extract_targets(critic)

    command_velocity = C.NUM_COMMAND_OBS - 1
    assert velocity.shape == (batch, C.NUM_ESTIMATOR_VELOCITY)
    assert latent.shape == (batch, C.NUM_ESTIMATOR_LATENT)
    assert torch.allclose(latent.norm(dim=-1), torch.ones(batch), atol=1.0e-6)
    assert torch.equal(target_velocity, critic[:, -C.NUM_PRIVILEGED_OBS :])
    assert torch.equal(target_obs, critic[:, command_velocity:])

    velocity_loss, swap_loss = estimator.loss(history, critic)
    (velocity_loss + swap_loss).backward()
    assert velocity_loss.ndim == swap_loss.ndim == 0
    assert torch.isfinite(velocity_loss) and torch.isfinite(swap_loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in estimator.parameters())


def test_estimator_mask_matches_explicit_selection_and_empty_mask_is_zero():
    torch.manual_seed(2)
    estimator = make_estimator()
    history = torch.randn(4, C.num_actor_obs())
    critic = torch.randn(4, C.num_critic_obs())
    valid = torch.tensor([True, False, True, False])
    masked = estimator.loss(history, critic, valid)
    selected = estimator.loss(history[valid], critic[valid])
    assert torch.allclose(masked[0], selected[0])
    assert torch.allclose(masked[1], selected[1])

    empty = estimator.loss(history, critic, torch.zeros(4, dtype=torch.bool))
    assert all(loss.ndim == 0 and loss.item() == 0.0 for loss in empty)
    sum(empty).backward()


def test_target_layout_uses_explicit_actor_latest_and_critic_tail_normalization():
    batch = 2
    critic = torch.arange(batch * C.num_critic_obs(), dtype=torch.float32).reshape(
        batch, C.num_critic_obs()
    )
    actor_mean = torch.arange(C.num_actor_obs(), dtype=torch.float32)
    actor_std = torch.arange(1, C.num_actor_obs() + 1, dtype=torch.float32)
    critic_mean = torch.arange(C.num_critic_obs(), dtype=torch.float32) + 1000.0
    critic_std = torch.arange(1, C.num_critic_obs() + 1, dtype=torch.float32) + 10.0

    velocity, target = extract_estimator_targets(
        critic,
        actor_history_mean=actor_mean,
        actor_history_std=actor_std,
        critic_mean=critic_mean,
        critic_std=critic_std,
        eps=0.0,
    )
    frame = C.num_one_step_actor_obs()
    dropped = C.NUM_COMMAND_OBS - 1
    expected_actor = (
        critic[:, :frame] - actor_mean[-frame:]
    ) / actor_std[-frame:]
    expected_velocity = (
        critic[:, frame:] - critic_mean[-C.NUM_PRIVILEGED_OBS :]
    ) / critic_std[-C.NUM_PRIVILEGED_OBS :]
    assert torch.equal(velocity, expected_velocity)
    assert torch.equal(target, torch.cat((expected_actor[:, dropped:], expected_velocity), dim=-1))


def test_estimator_rejects_wrong_dimensions_and_normalizes_prototypes():
    estimator = make_estimator()
    with pytest.raises(ValueError, match="history dimension"):
        estimator(torch.zeros(2, C.num_actor_obs() - 1))
    with pytest.raises(ValueError, match="critic observation dimension"):
        estimator.extract_targets(torch.zeros(2, C.num_critic_obs() - 1))
    with torch.no_grad():
        estimator.prototypes.weight.mul_(7.0)
    estimator.normalize_prototypes_()
    assert torch.allclose(estimator.prototypes.weight.norm(dim=-1), torch.ones(16), atol=1.0e-6)
