from __future__ import annotations

import copy

import pytest
import torch

from openhomie_isaaclab import elf3_constants as C
from openhomie_isaaclab.him_rl import HIMPPO, HIMRolloutStorage, NonFiniteTrainingError
from rsl_rl.storage import RolloutStorage

from _elf3_m4_helpers import make_obs, make_policy, mirror_spec


def fill_transition(transition, obs, marker: float, mask=None):
    envs = obs.batch_size[0]
    transition.observations = obs
    transition.actions = torch.full((envs, C.NUM_POLICY_ACTIONS), marker)
    transition.rewards = torch.full((envs,), marker)
    transition.dones = torch.zeros(envs, dtype=torch.bool)
    transition.values = torch.full((envs, 1), marker)
    transition.actions_log_prob = torch.full((envs,), marker)
    transition.action_mean = torch.full((envs, C.NUM_POLICY_ACTIONS), marker)
    transition.action_sigma = torch.ones(envs, C.NUM_POLICY_ACTIONS)
    transition.next_critic_observations = torch.full((envs, C.num_critic_obs()), marker + 10)
    transition.estimator_masks = torch.ones(envs, dtype=torch.bool) if mask is None else mask


def test_storage_preserves_stock_minibatch_prefix_and_appends_supervision():
    obs = make_obs(2)
    storage = HIMRolloutStorage(
        "rl", 2, 2, obs, [C.NUM_POLICY_ACTIONS], [C.num_critic_obs()], "cpu"
    )
    transition = HIMRolloutStorage.Transition()
    for marker in (1.0, 2.0):
        fill_transition(transition, obs, marker, torch.tensor([True, False]))
        storage.add_transitions(transition)
        transition.clear()
    storage.compute_returns(torch.zeros(2, 1), 0.99, 0.95)
    batch = next(storage.mini_batch_generator(1, 1))
    assert len(batch) == 12
    assert batch[-2].shape == (4, C.num_critic_obs())
    assert batch[-1].shape == (4, 1)
    assert torch.equal(storage.estimator_masks[..., 0], torch.tensor([[True, False], [True, False]]))


def test_single_branch_gae_exactly_matches_rsl_storage():
    torch.manual_seed(6)
    obs = make_obs(3)
    him = HIMRolloutStorage("rl", 3, 4, obs, [C.NUM_POLICY_ACTIONS], [C.num_critic_obs()], "cpu")
    stock = RolloutStorage("rl", 3, 4, obs, [C.NUM_POLICY_ACTIONS], "cpu")
    him.values.copy_(torch.randn_like(him.values))
    him.rewards.copy_(torch.randn_like(him.rewards))
    him.dones.copy_(torch.randint(0, 2, him.dones.shape, dtype=torch.uint8))
    stock.values.copy_(him.values); stock.rewards.copy_(him.rewards); stock.dones.copy_(him.dones)
    last = torch.randn(3, 1)
    him.compute_returns(last, 0.99, 0.95)
    stock.compute_returns(last, 0.99, 0.95)
    assert torch.equal(him.returns, stock.returns)
    assert torch.equal(him.advantages, stock.advantages)


def test_two_branches_compute_independent_gae_without_leakage():
    torch.manual_seed(7)
    envs, steps, branches = 2, 3, 2
    obs = make_obs(envs)
    storage = HIMRolloutStorage(
        "rl", envs, steps, obs, [C.NUM_POLICY_ACTIONS], [C.num_critic_obs()], "cpu",
        num_branches=branches,
    )
    values = storage.values.reshape(steps, branches, envs, 1)
    rewards = storage.rewards.reshape(steps, branches, envs, 1)
    dones = storage.dones.reshape(steps, branches, envs, 1)
    values.copy_(torch.randn_like(values)); rewards.copy_(torch.randn_like(rewards))
    dones.copy_(torch.randint(0, 2, dones.shape, dtype=torch.uint8))
    last = torch.randn(branches, envs, 1)
    references = []
    for branch in range(branches):
        stock = RolloutStorage("rl", envs, steps, obs, [C.NUM_POLICY_ACTIONS], "cpu")
        stock.values.copy_(values[:, branch]); stock.rewards.copy_(rewards[:, branch]); stock.dones.copy_(dones[:, branch])
        stock.compute_returns(last[branch], 0.99, 0.95, normalize_advantage=False)
        references.append(stock)
    storage.compute_returns(last, 0.99, 0.95, normalize_advantage=False)
    actual_returns = storage.returns.reshape(steps, branches, envs, 1)
    for branch, stock in enumerate(references):
        assert torch.equal(actual_returns[:, branch], stock.returns)


def test_storage_rejects_bad_mask_overflow_incomplete_and_final_value_shape():
    obs = make_obs(2)
    storage = HIMRolloutStorage("rl", 2, 1, obs, [C.NUM_POLICY_ACTIONS], [C.num_critic_obs()], "cpu")
    transition = HIMRolloutStorage.Transition()
    fill_transition(transition, obs, 1.0, torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="mask"):
        storage.add_transitions(transition)
    with pytest.raises(RuntimeError, match="full"):
        storage.compute_returns(torch.zeros(2, 1), 0.99, 0.95)
    fill_transition(transition, obs, 1.0)
    storage.add_transitions(transition)
    with pytest.raises(OverflowError):
        storage.add_transitions(transition)
    with pytest.raises(ValueError, match="last values"):
        storage.compute_returns(torch.zeros(2, 2, 1), 0.99, 0.95)


def test_episode_safe_supervision_truth_table_and_invalid_terminal_contract():
    policy = make_policy(4)
    algorithm = HIMPPO(policy, use_flip=False, num_learning_epochs=1, num_mini_batches=1)
    obs = make_obs(4)
    dones = torch.tensor([False, True, True, True])
    terminal = torch.full((4, C.num_critic_obs()), 123.0)
    extras = {
        "terminal_critic_obs": terminal,
        "terminal_critic_obs_mask": torch.tensor([False, True, True, False]),
    }
    next_critic, mask = algorithm._next_critic_supervision(obs, dones, extras)
    assert torch.equal(next_critic[0], obs["critic"][0])
    assert torch.equal(next_critic[1:3], terminal[1:3])
    assert torch.equal(next_critic[3], obs["critic"][3])
    assert torch.equal(mask, torch.tensor([True, True, True, False]))
    with pytest.raises(ValueError, match="provided together"):
        algorithm._next_critic_supervision(obs, dones, {"terminal_critic_obs": terminal})
    bad = dict(extras); bad["terminal_critic_obs_mask"] = torch.tensor([True, True, True, False])
    with pytest.raises(ValueError, match="completed"):
        algorithm._next_critic_supervision(obs, dones, bad)


def test_ppo_optimizers_are_disjoint_and_update_intended_parameters():
    torch.manual_seed(8)
    envs = 4
    policy = make_policy(envs)
    algorithm = HIMPPO(
        policy, use_flip=True, mirror=mirror_spec(identity=True), num_learning_epochs=1,
        num_mini_batches=1, learning_rate=1.0e-4, estimator_learning_rate=2.0e-4,
        schedule="fixed",
    )
    policy_ids = {id(p) for g in algorithm.optimizer.param_groups for p in g["params"]}
    estimator_ids = {id(p) for g in algorithm.estimator_optimizer.param_groups for p in g["params"]}
    assert policy_ids.isdisjoint(estimator_ids)
    obs = make_obs(envs)
    algorithm.init_storage("rl", envs, 2, obs, [C.NUM_POLICY_ACTIONS])
    for _ in range(2):
        algorithm.act(obs); next_obs = make_obs(envs)
        algorithm.process_env_step(next_obs, torch.randn(envs), torch.zeros(envs, dtype=torch.bool), {})
        obs = next_obs
    algorithm.compute_returns(obs)
    actor_before = copy.deepcopy(policy.actor.state_dict())
    estimator_before = copy.deepcopy(policy.estimator.state_dict())
    losses = algorithm.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert any(not torch.equal(actor_before[k], policy.actor.state_dict()[k]) for k in actor_before)
    assert any(not torch.equal(estimator_before[k], policy.estimator.state_dict()[k]) for k in estimator_before)


def test_nonfinite_policy_loss_steps_neither_optimizer():
    policy = make_policy(2)
    algorithm = HIMPPO(policy, use_flip=False, num_learning_epochs=1, num_mini_batches=1, schedule="fixed")
    obs = make_obs(2); algorithm.init_storage("rl", 2, 1, obs, [C.NUM_POLICY_ACTIONS])
    algorithm.act(obs); next_obs = make_obs(2)
    algorithm.process_env_step(next_obs, torch.ones(2), torch.zeros(2, dtype=torch.bool), {})
    algorithm.compute_returns(next_obs); algorithm.storage.returns[0, 0, 0] = float("nan")
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    with pytest.raises(NonFiniteTrainingError):
        algorithm.update()
    assert all(torch.equal(before[k], v) for k, v in policy.state_dict().items())
    assert algorithm.optimizer.state_dict()["state"] == {}
    assert algorithm.estimator_optimizer.state_dict()["state"] == {}
