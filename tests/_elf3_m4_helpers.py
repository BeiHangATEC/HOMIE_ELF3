from __future__ import annotations

import copy

import torch
from tensordict import TensorDict

from openhomie_isaaclab import elf3_constants as C


OBS_GROUPS = {"policy": ["policy"], "critic": ["critic"]}


def mirror_spec(identity: bool = False) -> dict[str, list[int]]:
    if identity:
        return {
            "dof_mirror_indices": list(range(C.NUM_ROBOT_DOFS)),
            "dof_mirror_signs": [1] * C.NUM_ROBOT_DOFS,
            "action_mirror_indices": list(range(C.NUM_POLICY_ACTIONS)),
            "action_mirror_signs": [1] * C.NUM_POLICY_ACTIONS,
            "obs_mirror_signs": [1] * C.NUM_OBS_HEAD,
            "critic_tail_mirror_signs": [1] * C.NUM_PRIVILEGED_OBS,
        }
    return {
        "dof_mirror_indices": list(C.DOF_MIRROR_INDICES),
        "dof_mirror_signs": list(C.DOF_MIRROR_SIGNS),
        "action_mirror_indices": list(C.ACTION_MIRROR_INDICES),
        "action_mirror_signs": list(C.ACTION_MIRROR_SIGNS),
        "obs_mirror_signs": list(C.OBS_HEAD_MIRROR_SIGNS),
        "critic_tail_mirror_signs": list(C.CRITIC_TAIL_MIRROR_SIGNS),
    }


def make_obs(batch_size: int = 4, *, random: bool = True) -> TensorDict:
    factory = torch.randn if random else torch.zeros
    return TensorDict(
        {
            "policy": factory(batch_size, C.num_actor_obs()),
            "critic": factory(batch_size, C.num_critic_obs()),
        },
        batch_size=[batch_size],
    )


def policy_kwargs(**overrides):
    values = {
        "num_one_step_obs": C.num_one_step_actor_obs(),
        "actor_history_length": C.NUM_ACTOR_HISTORY,
        "num_one_step_critic_obs": C.num_one_step_critic_obs(),
        "actor_hidden_dims": (32, 24),
        "critic_hidden_dims": (32, 24),
        "estimator_hidden_dims": (24, 16),
        "estimator_target_hidden_dims": (24, 16),
        "estimator_latent_dim": C.NUM_ESTIMATOR_LATENT,
    }
    values.update(overrides)
    return values


def make_policy(batch_size: int = 4, **overrides):
    from openhomie_isaaclab.him_rl import HIMActorCritic

    return HIMActorCritic(
        make_obs(batch_size),
        copy.deepcopy(OBS_GROUPS),
        C.NUM_POLICY_ACTIONS,
        **policy_kwargs(**overrides),
    )


def runner_cfg(*, normalization: bool = False, use_flip: bool = True) -> dict:
    return {
        "num_steps_per_env": 2,
        "save_interval": 200,
        "obs_groups": copy.deepcopy(OBS_GROUPS),
        "policy": {
            "class_name": "HIMActorCritic",
            **policy_kwargs(
                actor_obs_normalization=normalization,
                critic_obs_normalization=normalization,
            ),
        },
        "algorithm": {
            "class_name": "HIMPPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "learning_rate": 1.0e-4,
            "estimator_learning_rate": 2.0e-4,
            "schedule": "fixed",
            "use_flip": use_flip,
            "mirror": mirror_spec(identity=True) if use_flip else None,
        },
    }
