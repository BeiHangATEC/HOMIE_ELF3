from __future__ import annotations

import copy

import pytest
import torch

from openhomie_isaaclab import elf3_constants as C
from openhomie_isaaclab.him_rl import HIMPPO, NonFiniteTrainingError

from _elf3_m4_helpers import make_obs, make_policy, mirror_spec


def assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def make_ready_algorithm(envs=2, **kwargs):
    policy = make_policy(envs)
    kwargs.setdefault("schedule", "fixed")
    algorithm = HIMPPO(
        policy, use_flip=False, num_learning_epochs=1, num_mini_batches=1, **kwargs
    )
    obs = make_obs(envs)
    algorithm.init_storage("rl", envs, 1, obs, [C.NUM_POLICY_ACTIONS])
    algorithm.act(obs)
    next_obs = make_obs(envs)
    algorithm.process_env_step(
        next_obs, torch.ones(envs), torch.zeros(envs, dtype=torch.bool), {}
    )
    algorithm.compute_returns(next_obs)
    return policy, algorithm


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


def test_ppo_defaults_match_rsl_rl_312():
    algorithm = HIMPPO(make_policy(2), use_flip=False)
    assert algorithm.learning_rate == 1.0e-3
    assert algorithm.entropy_coef == 0.01
    assert algorithm.schedule == "adaptive"


def test_estimator_learning_rate_can_be_independent_or_follow_policy():
    current_mu = torch.full((2, C.NUM_POLICY_ACTIONS), 10.0)
    old_mu = torch.zeros_like(current_mu)
    sigma = torch.ones_like(current_mu)
    independent = HIMPPO(
        make_policy(2), use_flip=False, learning_rate=3.0e-3,
        estimator_learning_rate=7.0e-4, schedule="adaptive"
    )
    independent._adapt_learning_rate(current_mu, sigma, old_mu, sigma)
    assert independent.learning_rate == pytest.approx(2.0e-3)
    assert independent.estimator_learning_rate == pytest.approx(7.0e-4)
    assert independent.estimator_optimizer.param_groups[0]["lr"] == pytest.approx(7.0e-4)
    following = HIMPPO(
        make_policy(2), use_flip=False, learning_rate=3.0e-3,
        estimator_learning_rate=None, schedule="adaptive"
    )
    following._adapt_learning_rate(current_mu, sigma, old_mu, sigma)
    assert following.learning_rate == pytest.approx(2.0e-3)
    assert following.estimator_learning_rate == pytest.approx(2.0e-3)
    assert following.estimator_optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-3)


def test_estimator_update_receives_policy_target_normalization(monkeypatch):
    policy = make_policy(2, actor_obs_normalization=True, critic_obs_normalization=True)
    policy.update_normalization(make_obs(2))
    algorithm = HIMPPO(policy, use_flip=False, schedule="fixed")
    captured = {}

    def loss(history, critic, mask, normalization=None):
        captured["normalization"] = normalization
        zero = sum(parameter.sum() * 0.0 for parameter in policy.estimator.parameters())
        return zero, zero.clone()

    monkeypatch.setattr(policy.estimator, "loss", loss)
    algorithm._update_estimator(
        policy.prepare_actor_history(make_obs(2)),
        policy.get_critic_obs(make_obs(2)),
        torch.ones(2, dtype=torch.bool),
    )
    expected = policy.estimator_target_normalization()
    assert captured["normalization"].keys() == expected.keys()
    for key in ("actor_history_mean", "actor_history_std", "critic_mean", "critic_std"):
        assert torch.equal(captured["normalization"][key], expected[key])
    assert captured["normalization"]["eps"] == expected["eps"]


@pytest.mark.parametrize("optimizer_name", ["optimizer", "estimator_optimizer"])
@pytest.mark.parametrize("failure_mode", ["raise", "nonfinite"])
def test_optimizer_step_failure_rolls_back_model_and_both_optimizer_states(
    monkeypatch, optimizer_name, failure_mode
):
    policy, algorithm = make_ready_algorithm()
    for optimizer in (algorithm.optimizer, algorithm.estimator_optimizer):
        optimizer.zero_grad()
        loss = sum(
            parameter.square().sum()
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model_before = copy.deepcopy(policy.state_dict())
    policy_optimizer_before = copy.deepcopy(algorithm.optimizer.state_dict())
    estimator_optimizer_before = copy.deepcopy(algorithm.estimator_optimizer.state_dict())
    optimizer = getattr(algorithm, optimizer_name)
    original_step = optimizer.step

    def failing_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        if failure_mode == "raise":
            raise RuntimeError("injected optimizer failure")
        optimizer.param_groups[0]["params"][0].data.fill_(float("nan"))
        return result

    monkeypatch.setattr(optimizer, "step", failing_step)
    with pytest.raises((NonFiniteTrainingError, RuntimeError)):
        algorithm.update()
    assert_nested_equal(policy.state_dict(), model_before)
    assert_nested_equal(algorithm.optimizer.state_dict(), policy_optimizer_before)
    assert_nested_equal(algorithm.estimator_optimizer.state_dict(), estimator_optimizer_before)


def test_nonfinite_estimator_loss_steps_neither_optimizer(monkeypatch):
    policy, algorithm = make_ready_algorithm()

    def nonfinite_loss(history, critic, mask, normalization=None):
        loss = next(policy.estimator.parameters()).sum() * float("nan")
        return loss, loss.clone()

    monkeypatch.setattr(policy.estimator, "loss", nonfinite_loss)
    before = copy.deepcopy(policy.state_dict())
    with pytest.raises(NonFiniteTrainingError):
        algorithm.update()
    assert_nested_equal(policy.state_dict(), before)
    assert algorithm.optimizer.state_dict()["state"] == {}
    assert algorithm.estimator_optimizer.state_dict()["state"] == {}


@pytest.mark.parametrize("parameter_group", ["policy", "estimator"])
def test_nonfinite_gradient_steps_neither_optimizer(parameter_group):
    policy, algorithm = make_ready_algorithm()
    parameters = (
        tuple(policy.policy_parameters())
        if parameter_group == "policy"
        else tuple(policy.estimator.parameters())
    )
    handle = parameters[0].register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
    before = copy.deepcopy(policy.state_dict())
    try:
        with pytest.raises(NonFiniteTrainingError):
            algorithm.update()
    finally:
        handle.remove()
    assert_nested_equal(policy.state_dict(), before)
    assert algorithm.optimizer.state_dict()["state"] == {}
    assert algorithm.estimator_optimizer.state_dict()["state"] == {}


@pytest.mark.parametrize("nonfinite_call", [1, 2])
def test_nonfinite_gradient_norm_steps_neither_optimizer(monkeypatch, nonfinite_call):
    policy, algorithm = make_ready_algorithm()
    original_clip = torch.nn.utils.clip_grad_norm_
    calls = 0

    def nonfinite_clip(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original_clip(*args, **kwargs)
        return torch.tensor(float("inf")) if calls == nonfinite_call else result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", nonfinite_clip)
    before = copy.deepcopy(policy.state_dict())
    with pytest.raises(NonFiniteTrainingError):
        algorithm.update()
    assert_nested_equal(policy.state_dict(), before)
    assert algorithm.optimizer.state_dict()["state"] == {}
    assert algorithm.estimator_optimizer.state_dict()["state"] == {}


def test_mirrored_timeout_bootstrap_uses_each_branch_value():
    envs = 2
    policy = make_policy(envs)
    algorithm = HIMPPO(policy, use_flip=True, mirror=mirror_spec(identity=True))
    obs = make_obs(envs)
    algorithm.init_storage("rl", envs, 1, obs, [C.NUM_POLICY_ACTIONS])
    algorithm.act(obs)
    algorithm.transition.values.fill_(2.0)
    algorithm.transition_sym.values.fill_(5.0)
    rewards = torch.ones(envs)
    algorithm.process_env_step(
        make_obs(envs),
        rewards,
        torch.zeros(envs, dtype=torch.bool),
        {"time_outs": torch.ones(envs, dtype=torch.bool)},
    )
    stored = algorithm.storage.rewards.reshape(1, 2, envs, 1)[0, ..., 0]
    assert torch.allclose(stored[0], rewards + algorithm.gamma * 2.0)
    assert torch.allclose(stored[1], rewards + algorithm.gamma * 5.0)


def test_single_symmetry_scale_is_accepted_and_both_losses_are_reported():
    envs = 2
    policy = make_policy(envs)
    algorithm = HIMPPO(
        policy,
        use_flip=True,
        mirror=mirror_spec(identity=True),
        symmetry_scale=0.25,
        num_learning_epochs=1,
        num_mini_batches=1,
        schedule="fixed",
    )
    obs = make_obs(envs)
    algorithm.init_storage("rl", envs, 1, obs, [C.NUM_POLICY_ACTIONS])
    algorithm.act(obs)
    next_obs = make_obs(envs)
    algorithm.process_env_step(
        next_obs, torch.ones(envs), torch.zeros(envs, dtype=torch.bool), {}
    )
    algorithm.compute_returns(next_obs)
    losses = algorithm.update()
    assert algorithm.symmetry_scale == 0.25
    assert "actor_symmetry" in losses
    assert "critic_symmetry" in losses


def test_adaptive_lr_is_rolled_back_with_failed_atomic_update(monkeypatch):
    policy, algorithm = make_ready_algorithm(
        learning_rate=3.0e-3,
        estimator_learning_rate=None,
        schedule="adaptive",
    )
    algorithm.storage.mu.fill_(100.0)
    model_before = copy.deepcopy(policy.state_dict())
    policy_optimizer_before = copy.deepcopy(algorithm.optimizer.state_dict())
    estimator_optimizer_before = copy.deepcopy(algorithm.estimator_optimizer.state_dict())
    learning_rate_before = algorithm.learning_rate
    estimator_learning_rate_before = algorithm.estimator_learning_rate
    original_step = algorithm.optimizer.step

    def failing_step(*args, **kwargs):
        original_step(*args, **kwargs)
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(algorithm.optimizer, "step", failing_step)
    with pytest.raises((NonFiniteTrainingError, RuntimeError)):
        algorithm.update()
    assert algorithm.learning_rate == learning_rate_before
    assert algorithm.estimator_learning_rate == estimator_learning_rate_before
    assert_nested_equal(policy.state_dict(), model_before)
    assert_nested_equal(algorithm.optimizer.state_dict(), policy_optimizer_before)
    assert_nested_equal(algorithm.estimator_optimizer.state_dict(), estimator_optimizer_before)
