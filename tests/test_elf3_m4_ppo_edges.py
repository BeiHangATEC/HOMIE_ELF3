from __future__ import annotations

import torch

from openhomie_isaaclab import elf3_constants as C
from openhomie_isaaclab.him_rl import HIMPPO

from _elf3_m4_helpers import make_obs, make_policy, mirror_spec


def test_timeout_bootstrap_is_separate_from_terminal_estimator_supervision():
    policy = make_policy(3)
    algorithm = HIMPPO(policy, use_flip=False, num_learning_epochs=1, num_mini_batches=1)
    obs = make_obs(3)
    algorithm.init_storage("rl", 3, 1, obs, [C.NUM_POLICY_ACTIONS])
    algorithm.act(obs)
    original_values = algorithm.transition.values.squeeze(-1).clone()
    next_obs = make_obs(3)
    dones = torch.tensor([False, True, True])
    terminal = torch.full((3, C.num_critic_obs()), 77.0)
    extras = {
        "time_outs": torch.tensor([False, False, True]),
        "terminal_critic_obs": terminal,
        "terminal_critic_obs_mask": dones,
    }
    rewards = torch.ones(3)
    algorithm.process_env_step(next_obs, rewards, dones, extras)
    expected = rewards.clone()
    expected[2] += algorithm.gamma * original_values[2]
    assert torch.allclose(algorithm.storage.rewards[0, :, 0], expected)
    assert torch.equal(
        algorithm.storage.estimator_masks[0, :, 0], torch.ones(3, dtype=torch.bool)
    )
    assert torch.equal(algorithm.storage.next_critic_observations[0, 1:], terminal[1:])


def test_mirrored_branch_is_sampled_from_its_own_distribution():
    torch.manual_seed(71)
    policy = make_policy(3)
    algorithm = HIMPPO(policy, use_flip=True, mirror=mirror_spec())
    obs = make_obs(3)
    algorithm.act(obs)
    mirrored_obs = algorithm.mirror_transform.observations(obs, policy.obs_groups)
    stored_actions = algorithm.transition_sym.actions.clone()
    stored_mean = algorithm.transition_sym.action_mean.clone()
    stored_log_prob = algorithm.transition_sym.actions_log_prob.clone()
    policy.act(mirrored_obs)
    assert torch.equal(stored_mean, policy.action_mean)
    assert torch.equal(stored_log_prob, policy.get_actions_log_prob(stored_actions))
    assert not torch.equal(stored_mean, algorithm.transition.action_mean)
