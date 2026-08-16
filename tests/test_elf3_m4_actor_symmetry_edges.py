from __future__ import annotations

import copy

import pytest
import torch

from openhomie_isaaclab import elf3_constants as C
from openhomie_isaaclab.him_rl import HIMActorCritic, MirrorTransform

from _elf3_m4_helpers import OBS_GROUPS, make_obs, make_policy, mirror_spec, policy_kwargs


def make_transform(spec=None):
    return MirrorTransform(
        mirror_spec() if spec is None else spec,
        C.num_one_step_actor_obs(),
        C.num_one_step_critic_obs(),
        C.NUM_ACTOR_HISTORY,
        C.NUM_CRITIC_HISTORY,
    )


def test_normalizer_state_round_trip_preserves_all_paths():
    torch.manual_seed(13)
    policy = make_policy(4, actor_obs_normalization=True, critic_obs_normalization=True)
    training_obs = make_obs(4)
    policy.update_normalization(training_obs)
    probe = make_obs(3)
    expected_history = policy.prepare_actor_history(probe)
    expected_critic = policy.prepare_critic_obs(probe)
    expected_actions = policy.act_inference(probe)
    state = copy.deepcopy(policy.state_dict())

    restored = make_policy(3, actor_obs_normalization=True, critic_obs_normalization=True)
    assert restored.load_state_dict(state) is True
    assert torch.equal(restored.prepare_actor_history(probe), expected_history)
    assert torch.equal(restored.prepare_critic_obs(probe), expected_critic)
    assert torch.equal(restored.act_inference(probe), expected_actions)
    assert restored.actor_obs_normalizer.count.item() == policy.actor_obs_normalizer.count.item()
    assert restored.critic_obs_normalizer.count.item() == policy.critic_obs_normalizer.count.item()


def test_actor_critic_exposes_exact_estimator_target_normalization_mapping():
    policy = make_policy(2, actor_obs_normalization=True, critic_obs_normalization=True)
    policy.update_normalization(make_obs(2))
    normalization = policy.estimator_target_normalization()
    assert set(normalization) == {
        "actor_history_mean", "actor_history_std", "critic_mean", "critic_std", "eps"
    }
    assert torch.equal(normalization["actor_history_mean"], policy.actor_obs_normalizer.mean)
    assert torch.equal(normalization["actor_history_std"], policy.actor_obs_normalizer.std)
    assert torch.equal(normalization["critic_mean"], policy.critic_obs_normalizer.mean)
    assert torch.equal(normalization["critic_std"], policy.critic_obs_normalizer.std)
    assert normalization["eps"] == policy.actor_obs_normalizer.eps
    assert normalization["eps"] == policy.critic_obs_normalizer.eps

    unnormalized = make_policy(2)
    assert unnormalized.estimator_target_normalization() is None


def test_mirror_applies_every_canonical_segment_permutation_and_sign():
    transform = make_transform()
    spec = mirror_spec()
    frame_width = C.num_one_step_actor_obs()
    frame = torch.arange(frame_width, dtype=torch.float64).unsqueeze(0)
    history = frame.repeat(1, C.NUM_ACTOR_HISTORY)
    mirrored = transform.actor(history).reshape(1, C.NUM_ACTOR_HISTORY, frame_width)

    dof_pos = C.NUM_OBS_HEAD
    dof_vel = dof_pos + C.NUM_ROBOT_DOFS
    action = dof_vel + C.NUM_ROBOT_DOFS
    expected = torch.empty_like(frame)
    expected[:, :dof_pos] = frame[:, :dof_pos] * torch.tensor(spec["obs_mirror_signs"])
    expected[:, dof_pos:dof_vel] = (
        frame[:, dof_pos + torch.tensor(spec["dof_mirror_indices"])]
        * torch.tensor(spec["dof_mirror_signs"])
    )
    expected[:, dof_vel:action] = (
        frame[:, dof_vel + torch.tensor(spec["dof_mirror_indices"])]
        * torch.tensor(spec["dof_mirror_signs"])
    )
    expected[:, action:] = (
        frame[:, action + torch.tensor(spec["action_mirror_indices"])]
        * torch.tensor(spec["action_mirror_signs"])
    )
    assert torch.equal(mirrored, expected.unsqueeze(1).expand_as(mirrored))

    actions = torch.arange(C.NUM_POLICY_ACTIONS, dtype=torch.float64).unsqueeze(0)
    expected_actions = (
        actions[:, torch.tensor(spec["action_mirror_indices"])]
        * torch.tensor(spec["action_mirror_signs"])
    )
    assert torch.equal(transform.actions(actions), expected_actions)


def test_mirror_preserves_arbitrary_leading_dimensions():
    transform = make_transform()
    actor = torch.randn(2, 3, C.num_actor_obs())
    critic = torch.randn(2, 3, C.num_critic_obs())
    actions = torch.randn(2, 3, C.NUM_POLICY_ACTIONS)
    assert transform.actor(actor).shape == actor.shape
    assert transform.critic(critic).shape == critic.shape
    assert transform.actions(actions).shape == actions.shape
    assert torch.equal(transform.actor(transform.actor(actor)), actor)


@pytest.mark.parametrize(
    "field",
    ["dof_mirror_signs", "action_mirror_signs", "obs_mirror_signs", "critic_tail_mirror_signs"],
)
def test_mirror_rejects_signs_other_than_plus_or_minus_one(field):
    spec = mirror_spec()
    spec[field][0] = 0
    with pytest.raises(ValueError, match=r"\+1 or -1"):
        make_transform(spec)


@pytest.mark.parametrize(
    ("method", "width"),
    [
        ("actor", C.num_actor_obs() - 1),
        ("critic", C.num_critic_obs() - 1),
        ("actions", C.NUM_POLICY_ACTIONS - 1),
    ],
)
def test_mirror_runtime_rejects_wrong_last_dimension(method, width):
    with pytest.raises(ValueError, match="dimension"):
        getattr(make_transform(), method)(torch.zeros(2, width))


def test_actor_critic_rejects_missing_groups_and_nonpositive_noise():
    obs = make_obs()
    with pytest.raises(ValueError, match="critic"):
        HIMActorCritic(
            obs, {"policy": ["policy"]}, C.NUM_POLICY_ACTIONS, **policy_kwargs()
        )
    with pytest.raises(ValueError, match="noise"):
        HIMActorCritic(
            obs,
            copy.deepcopy(OBS_GROUPS),
            C.NUM_POLICY_ACTIONS,
            **policy_kwargs(init_noise_std=0.0),
        )


@pytest.mark.parametrize(
    "invalid_std",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_distribution_rejects_runtime_nonpositive_or_nonfinite_std(invalid_std):
    policy = make_policy(2)
    with torch.no_grad():
        policy.std.fill_(invalid_std)
    with pytest.raises(ValueError, match="finite and positive"):
        policy.act(make_obs(2))


def test_constructor_accepts_disabled_state_dependent_std_and_rejects_enabled():
    policy = make_policy(2, state_dependent_std=False)
    assert policy.act(make_obs(2)).shape == (2, C.NUM_POLICY_ACTIONS)
    with pytest.raises(ValueError, match="state-dependent"):
        make_policy(2, state_dependent_std=True)


def test_prefixed_estimator_config_is_mapped_to_estimator_parameters():
    policy = make_policy(
        2,
        estimator_temperature=2.5,
        estimator_sinkhorn_epsilon=0.07,
        estimator_sinkhorn_iterations=5,
    )
    assert policy.estimator.temperature == 2.5
    assert policy.estimator.sinkhorn_epsilon == 0.07
    assert policy.estimator.sinkhorn_iterations == 5


def test_actor_critic_requires_three_value_velocity_tail():
    obs = make_obs()
    wider_critic = torch.zeros(4, C.num_critic_obs() + 1)
    bad_obs = obs.clone()
    bad_obs["critic"] = wider_critic
    with pytest.raises(ValueError, match="three-value velocity tail"):
        HIMActorCritic(
            bad_obs,
            copy.deepcopy(OBS_GROUPS),
            C.NUM_POLICY_ACTIONS,
            **policy_kwargs(num_one_step_critic_obs=C.num_one_step_critic_obs() + 1),
        )


def test_actor_critic_rejects_multi_frame_critic_history():
    obs = make_obs()
    bad_obs = obs.clone()
    bad_obs["critic"] = torch.zeros(4, C.num_one_step_critic_obs() * 2)
    with pytest.raises(ValueError, match="critic history.*one"):
        HIMActorCritic(
            bad_obs,
            copy.deepcopy(OBS_GROUPS),
            C.NUM_POLICY_ACTIONS,
            **policy_kwargs(critic_history_length=2),
        )
