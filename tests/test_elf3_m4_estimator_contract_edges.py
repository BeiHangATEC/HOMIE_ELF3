from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

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


def test_estimator_defaults_to_approved_homie_temperature():
    assert make_estimator().temperature == 3.0


def test_swap_loss_uses_elementwise_mean_over_samples_and_prototypes():
    torch.manual_seed(12)
    estimator = make_estimator()
    history = torch.randn(5, C.num_actor_obs())
    critic = torch.randn(5, C.num_critic_obs())
    _velocity_loss, actual_swap = estimator.loss(history, critic)

    _, source_latent = estimator(history)
    _, target_obs = estimator.extract_targets(critic)
    target_latent = F.normalize(estimator.target_encoder(target_obs), dim=-1)
    source_scores = estimator.prototype_scores(source_latent)
    target_scores = estimator.prototype_scores(target_latent)
    source_assignments = sinkhorn(
        source_scores.detach(),
        epsilon=estimator.sinkhorn_epsilon,
        iterations=estimator.sinkhorn_iterations,
    )
    target_assignments = sinkhorn(
        target_scores.detach(),
        epsilon=estimator.sinkhorn_epsilon,
        iterations=estimator.sinkhorn_iterations,
    )
    expected_swap = -0.5 * (
        source_assignments * F.log_softmax(target_scores / estimator.temperature, dim=-1)
        + target_assignments * F.log_softmax(source_scores / estimator.temperature, dim=-1)
    ).mean()
    assert torch.allclose(actual_swap, expected_swap, atol=1.0e-7, rtol=0.0)


@pytest.mark.parametrize("eps", [float("nan"), float("inf")])
def test_target_normalization_rejects_nonfinite_epsilon(eps):
    critic = torch.zeros(2, C.num_critic_obs())
    with pytest.raises(ValueError, match="finite and nonnegative"):
        extract_estimator_targets(
            critic,
            actor_history_mean=torch.zeros(C.num_actor_obs()),
            actor_history_std=torch.ones(C.num_actor_obs()),
            critic_mean=torch.zeros(C.num_critic_obs()),
            critic_std=torch.ones(C.num_critic_obs()),
            eps=eps,
        )


def test_estimator_loss_accepts_enabled_target_normalization_contract():
    estimator = make_estimator()
    with torch.no_grad():
        for parameter in estimator.source_encoder.parameters():
            parameter.zero_()
    history = torch.zeros(2, C.num_actor_obs())
    critic = torch.zeros(2, C.num_critic_obs())
    critic[:, -C.NUM_PRIVILEGED_OBS :] = 10.0
    normalization = {
        "actor_history_mean": torch.zeros(C.num_actor_obs()),
        "actor_history_std": torch.ones(C.num_actor_obs()),
        "critic_mean": torch.cat(
            (torch.zeros(C.num_one_step_actor_obs()), torch.full((C.NUM_PRIVILEGED_OBS,), 4.0))
        ),
        "critic_std": torch.cat(
            (torch.ones(C.num_one_step_actor_obs()), torch.full((C.NUM_PRIVILEGED_OBS,), 2.0))
        ),
        "eps": 0.0,
    }
    velocity_loss, swap_loss = estimator.loss(
        history, critic, normalization=normalization
    )
    assert torch.allclose(velocity_loss, torch.tensor(9.0))
    assert torch.isfinite(swap_loss)
