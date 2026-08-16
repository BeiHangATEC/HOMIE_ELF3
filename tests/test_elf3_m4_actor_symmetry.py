from __future__ import annotations

import copy

import pytest
import torch
from tensordict import TensorDict

from openhomie_isaaclab import elf3_constants as C
from openhomie_isaaclab.him_rl import HIMActorCritic, MirrorTransform
from rsl_rl.networks import EmpiricalNormalization

from _elf3_m4_helpers import OBS_GROUPS, make_obs, make_policy, mirror_spec, policy_kwargs


def test_actor_critic_implements_rsl_312_tensordict_contract():
    torch.manual_seed(3)
    obs = make_obs(5)
    policy = make_policy(5)
    actions = policy.act(obs)
    deterministic = policy.act_inference(obs)
    values = policy.evaluate(obs)
    assert actions.shape == deterministic.shape == (5, C.NUM_POLICY_ACTIONS)
    assert values.shape == (5, 1)
    assert policy.get_actions_log_prob(actions).shape == (5,)
    assert policy.action_mean.shape == policy.action_std.shape == actions.shape
    assert policy.entropy.shape == (5,)
    assert policy.is_recurrent is False
    assert policy.load_state_dict(copy.deepcopy(policy.state_dict())) is True


def test_actor_input_uses_latest_normalized_frame_and_detached_estimator_features():
    torch.manual_seed(4)
    policy = make_policy(3)
    obs = make_obs(3)
    prepared = policy.prepare_actor_history(obs)
    actor_input = policy.actor_input(prepared)
    frame = C.num_one_step_actor_obs()
    assert actor_input.shape[-1] == C.num_actor_mlp_input()
    assert torch.equal(actor_input[:, :frame], prepared[:, -frame:])

    for parameter in policy.parameters():
        parameter.grad = None
    policy.act_inference(obs).square().mean().backward()
    assert any(p.grad is not None for p in policy.actor.parameters())
    assert all(p.grad is None for p in policy.estimator.parameters())


def test_policy_and_estimator_parameter_partitions_are_exact():
    policy = make_policy()
    policy_parameters = tuple(policy.policy_parameters())
    estimator_parameters = tuple(policy.estimator.parameters())
    policy_ids = {id(p) for p in policy_parameters}
    estimator_ids = {id(p) for p in estimator_parameters}
    assert policy_ids and estimator_ids and policy_ids.isdisjoint(estimator_ids)
    assert policy_ids | estimator_ids == {id(p) for p in policy.parameters()}


def test_enabled_normalizers_update_once_and_use_separate_actor_critic_paths():
    policy = make_policy(
        2, actor_obs_normalization=True, critic_obs_normalization=True
    )
    assert isinstance(policy.actor_obs_normalizer, EmpiricalNormalization)
    assert isinstance(policy.critic_obs_normalizer, EmpiricalNormalization)
    obs = make_obs(2, random=False)
    obs["policy"][:, -C.num_one_step_actor_obs() :] = 2.0
    obs["critic"][:] = 3.0
    before_actor = policy.actor_obs_normalizer.count.item()
    before_critic = policy.critic_obs_normalizer.count.item()
    policy.update_normalization(obs)
    assert policy.actor_obs_normalizer.count.item() == before_actor + 2
    assert policy.critic_obs_normalizer.count.item() == before_critic + 2
    assert torch.equal(
        policy.prepare_actor_history(obs), policy.actor_obs_normalizer(obs["policy"])
    )
    assert torch.equal(
        policy.prepare_critic_obs(obs), policy.critic_obs_normalizer(obs["critic"])
    )


def test_actor_critic_rejects_bad_groups_and_dimensions():
    obs = make_obs()
    with pytest.raises(ValueError, match="distinct"):
        HIMActorCritic(obs, {"policy": ["policy"], "critic": ["policy"]}, C.NUM_POLICY_ACTIONS,
                       **policy_kwargs())
    bad = TensorDict(
        {"policy": torch.zeros(4, C.num_actor_obs() - 1), "critic": obs["critic"]},
        batch_size=[4],
    )
    with pytest.raises(ValueError, match="history"):
        HIMActorCritic(bad, copy.deepcopy(OBS_GROUPS), C.NUM_POLICY_ACTIONS, **policy_kwargs())


def test_canonical_mirror_is_exact_involution_for_all_paths():
    torch.manual_seed(5)
    transform = MirrorTransform(
        mirror_spec(),
        num_one_step_obs=C.num_one_step_actor_obs(),
        num_one_step_critic_obs=C.num_one_step_critic_obs(),
        actor_history_length=C.NUM_ACTOR_HISTORY,
        critic_history_length=C.NUM_CRITIC_HISTORY,
    )
    actor = torch.randn(7, C.num_actor_obs(), dtype=torch.float64)
    critic = torch.randn(7, C.num_critic_obs(), dtype=torch.float64)
    actions = torch.randn(7, C.NUM_POLICY_ACTIONS, dtype=torch.float64)
    assert torch.equal(transform.actor(transform.actor(actor)), actor)
    assert torch.equal(transform.critic(transform.critic(critic)), critic)
    assert torch.equal(transform.actions(transform.actions(actions)), actions)
    assert transform.actor(actor).dtype == actor.dtype

    obs = TensorDict({"policy": actor, "critic": critic}, batch_size=[7])
    mirrored = transform.observations(obs, OBS_GROUPS)
    assert torch.equal(mirrored["policy"], transform.actor(actor))
    assert torch.equal(mirrored["critic"], transform.critic(critic))


def test_mirror_uses_canonical_28_dof_spec_and_expected_vector_signs():
    spec = mirror_spec()
    assert len(spec["dof_mirror_indices"]) == C.NUM_ROBOT_DOFS
    assert len(spec["action_mirror_indices"]) == C.NUM_POLICY_ACTIONS
    transform = MirrorTransform(
        spec,
        C.num_one_step_actor_obs(),
        C.num_one_step_critic_obs(),
        C.NUM_ACTOR_HISTORY,
        C.NUM_CRITIC_HISTORY,
    )
    critic = torch.zeros(1, C.num_critic_obs())
    critic[:, -C.NUM_PRIVILEGED_OBS :] = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.equal(
        transform.critic(critic)[:, -C.NUM_PRIVILEGED_OBS :],
        torch.tensor([[1.0, -2.0, 3.0]]),
    )


@pytest.mark.parametrize("field", ["dof_mirror_indices", "action_mirror_indices"])
def test_mirror_rejects_non_involutive_permutations(field):
    spec = mirror_spec()
    spec[field][0:3] = [1, 2, 0]
    with pytest.raises(ValueError, match="involution"):
        MirrorTransform(
            spec,
            C.num_one_step_actor_obs(),
            C.num_one_step_critic_obs(),
            C.NUM_ACTOR_HISTORY,
            C.NUM_CRITIC_HISTORY,
        )
