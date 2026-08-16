"""Specification-driven mirror transforms for HIM observations and actions."""

from __future__ import annotations

from numbers import Integral
from typing import Mapping, Sequence

import torch
from tensordict import TensorDict


class MirrorTransform:
    """Mirror actor histories, critic histories, actions, and TensorDict groups."""

    def __init__(
        self,
        spec: Mapping[str, Sequence[int]],
        num_one_step_obs: int,
        num_one_step_critic_obs: int,
        actor_history_length: int,
        critic_history_length: int,
    ) -> None:
        self.dof_indices = self._permutation(spec, "dof_mirror_indices")
        self.action_indices = self._permutation(spec, "action_mirror_indices")
        self.dof_signs = self._signs(spec, "dof_mirror_signs", len(self.dof_indices))
        self.action_signs = self._signs(spec, "action_mirror_signs", len(self.action_indices))
        self.obs_signs = self._signs(spec, "obs_mirror_signs")
        self.critic_tail_signs = self._signs(spec, "critic_tail_mirror_signs")
        self._validate_signed_involution(self.dof_indices, self.dof_signs, "DOF")
        self._validate_signed_involution(self.action_indices, self.action_signs, "action")

        self.num_one_step_obs = self._positive_int("actor frame", num_one_step_obs)
        self.num_one_step_critic_obs = self._positive_int("critic frame", num_one_step_critic_obs)
        self.actor_history_length = self._positive_int("actor history", actor_history_length)
        self.critic_history_length = self._positive_int("critic history", critic_history_length)
        expected_actor = len(self.obs_signs) + 2 * len(self.dof_indices) + len(self.action_indices)
        if self.num_one_step_obs != expected_actor:
            raise ValueError(f"actor frame dimension must be {expected_actor}, got {self.num_one_step_obs}")
        expected_critic = self.num_one_step_obs + len(self.critic_tail_signs)
        if self.num_one_step_critic_obs != expected_critic:
            raise ValueError(f"critic frame dimension must be {expected_critic}, got {self.num_one_step_critic_obs}")
        self.actor_dim = self.num_one_step_obs * self.actor_history_length
        self.critic_dim = self.num_one_step_critic_obs * self.critic_history_length
        self.action_dim = len(self.action_indices)

    @staticmethod
    def _positive_int(name: str, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} dimension must be a positive integer")
        return int(value)

    @staticmethod
    def _permutation(spec: Mapping[str, Sequence[int]], name: str) -> tuple[int, ...]:
        if name not in spec:
            raise ValueError(f"mirror specification is missing {name}")
        values = tuple(spec[name])
        if not values or any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
            raise ValueError(f"{name} must be a nonempty integer permutation")
        if any(value < 0 or value >= len(values) for value in values):
            raise ValueError(f"{name} must be a complete bijection")
        if any(values[values[index]] != index for index in range(len(values))):
            raise ValueError(f"{name} must be an involution")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must be a complete bijection")
        return tuple(int(value) for value in values)

    @staticmethod
    def _signs(spec: Mapping[str, Sequence[int]], name: str, width: int | None = None) -> tuple[int, ...]:
        if name not in spec:
            raise ValueError(f"mirror specification is missing {name}")
        values = tuple(spec[name])
        if width is not None and len(values) != width:
            raise ValueError(f"{name} dimension must be {width}")
        if not values or any(value not in (-1, 1) for value in values):
            raise ValueError(f"{name} values must be +1 or -1")
        return tuple(int(value) for value in values)

    @staticmethod
    def _validate_signed_involution(indices: tuple[int, ...], signs: tuple[int, ...], name: str) -> None:
        if any(signs[index] * signs[indices[index]] != 1 for index in range(len(indices))):
            raise ValueError(f"{name} permutation and signs must form an involution")

    @staticmethod
    def _index(values: tuple[int, ...], tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long, device=tensor.device)

    @staticmethod
    def _scale(values: tuple[int, ...], tensor: torch.Tensor) -> torch.Tensor:
        return tensor.new_tensor(values)

    def _actor_frames(self, frames: torch.Tensor) -> torch.Tensor:
        head_end = len(self.obs_signs)
        position_end = head_end + len(self.dof_indices)
        velocity_end = position_end + len(self.dof_indices)
        result = torch.empty_like(frames)
        result[..., :head_end] = frames[..., :head_end] * self._scale(self.obs_signs, frames)
        dof_indices = self._index(self.dof_indices, frames)
        dof_signs = self._scale(self.dof_signs, frames)
        result[..., head_end:position_end] = frames[..., head_end:position_end].index_select(-1, dof_indices) * dof_signs
        result[..., position_end:velocity_end] = frames[..., position_end:velocity_end].index_select(-1, dof_indices) * dof_signs
        result[..., velocity_end:] = frames[..., velocity_end:].index_select(
            -1, self._index(self.action_indices, frames)
        ) * self._scale(self.action_signs, frames)
        return result

    def actor(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim < 1 or observations.shape[-1] != self.actor_dim:
            raise ValueError(f"actor observation dimension must be {self.actor_dim}")
        frames = observations.reshape(*observations.shape[:-1], self.actor_history_length, self.num_one_step_obs)
        return self._actor_frames(frames).reshape_as(observations)

    def critic(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim < 1 or observations.shape[-1] != self.critic_dim:
            raise ValueError(f"critic observation dimension must be {self.critic_dim}")
        frames = observations.reshape(*observations.shape[:-1], self.critic_history_length, self.num_one_step_critic_obs)
        result = torch.empty_like(frames)
        result[..., : self.num_one_step_obs] = self._actor_frames(frames[..., : self.num_one_step_obs])
        result[..., self.num_one_step_obs :] = frames[..., self.num_one_step_obs :] * self._scale(
            self.critic_tail_signs, frames
        )
        return result.reshape_as(observations)

    def actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim < 1 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"action dimension must be {self.action_dim}")
        return actions.index_select(-1, self._index(self.action_indices, actions)) * self._scale(
            self.action_signs, actions
        )

    @staticmethod
    def _group_tensor(observations: TensorDict, groups: list[str]) -> torch.Tensor:
        try:
            return torch.cat([observations[name] for name in groups], dim=-1)
        except KeyError as error:
            raise ValueError(f"observation group {error.args[0]!r} is missing") from error

    @staticmethod
    def _store_groups(observations: TensorDict, groups: list[str], transformed: torch.Tensor) -> None:
        cursor = 0
        for name in groups:
            width = observations[name].shape[-1]
            observations[name] = transformed[..., cursor : cursor + width]
            cursor += width

    def observations(self, observations: TensorDict, obs_groups: Mapping[str, Sequence[str]]) -> TensorDict:
        if "policy" not in obs_groups or "critic" not in obs_groups:
            raise ValueError("policy and critic observation groups are required")
        policy_groups = list(obs_groups["policy"])
        critic_groups = list(obs_groups["critic"])
        if not policy_groups or not critic_groups:
            raise ValueError("policy and critic observation groups must be nonempty")
        if set(policy_groups) & set(critic_groups):
            raise ValueError("policy and critic observation groups must be distinct")
        mirrored = observations.clone()
        self._store_groups(mirrored, policy_groups, self.actor(self._group_tensor(observations, policy_groups)))
        self._store_groups(mirrored, critic_groups, self.critic(self._group_tensor(observations, critic_groups)))
        return mirrored
