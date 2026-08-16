"""Branch-aware rollout storage for HIM PPO."""

from __future__ import annotations

from collections.abc import Generator
from numbers import Integral

import torch
from tensordict import TensorDict

from rsl_rl.storage import RolloutStorage


class HIMRolloutStorage(RolloutStorage):
    """Stock-compatible feed-forward storage with estimator supervision."""

    class Transition(RolloutStorage.Transition):
        def __init__(self) -> None:
            super().__init__()
            self.next_critic_observations: torch.Tensor | None = None
            self.estimator_masks: torch.Tensor | None = None

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        critic_obs_shape: tuple[int] | list[int],
        device: str = "cpu",
        *,
        num_branches: int = 1,
    ) -> None:
        if training_type != "rl":
            raise ValueError("HIM rollout storage only supports rl training")
        if isinstance(num_branches, bool) or not isinstance(num_branches, Integral) or num_branches < 1:
            raise ValueError("number of branches must be a positive integer")
        if not critic_obs_shape or any(dim <= 0 for dim in critic_obs_shape):
            raise ValueError("critic shape must contain positive dimensions")
        self.num_branches = int(num_branches)
        self.num_logical_steps = int(num_transitions_per_env)
        super().__init__(
            training_type,
            num_envs,
            self.num_logical_steps * self.num_branches,
            obs,
            actions_shape,
            device,
        )
        self.next_critic_observations = torch.zeros(
            self.num_transitions_per_env, num_envs, *critic_obs_shape, device=device
        )
        self.estimator_masks = torch.zeros(
            self.num_transitions_per_env, num_envs, 1, dtype=torch.bool, device=device
        )

    def add_transitions(self, transition: Transition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("rollout buffer overflow")
        required = (
            "observations",
            "actions",
            "rewards",
            "dones",
            "values",
            "actions_log_prob",
            "action_mean",
            "action_sigma",
            "next_critic_observations",
            "estimator_masks",
        )
        if any(getattr(transition, name, None) is None for name in required):
            raise ValueError("transition is incomplete or has missing fields")
        mask = transition.estimator_masks
        if mask.dtype != torch.bool or mask.shape not in ((self.num_envs,), (self.num_envs, 1)):
            raise ValueError("estimator mask must be boolean with shape [num_envs] or [num_envs, 1]")
        next_critic = transition.next_critic_observations
        if next_critic.shape != self.next_critic_observations[self.step].shape:
            raise ValueError("next critic observation dimension is invalid")
        self.next_critic_observations[self.step].copy_(next_critic)
        self.estimator_masks[self.step].copy_(mask.reshape(self.num_envs, 1))
        super().add_transitions(transition)

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
        normalize_advantage: bool = True,
    ) -> None:
        if self.step != self.num_transitions_per_env:
            raise RuntimeError("rollout storage must be full before computing returns")
        if self.num_branches == 1:
            expected = (self.num_envs, 1)
            if last_values.shape != expected:
                raise ValueError(f"last values must have shape {expected}")
            branch_last_values = last_values.unsqueeze(0)
        else:
            expected = (self.num_branches, self.num_envs, 1)
            if last_values.shape != expected:
                raise ValueError(f"last values must have shape {expected}")
            branch_last_values = last_values

        values = self.values.reshape(self.num_logical_steps, self.num_branches, self.num_envs, 1)
        rewards = self.rewards.reshape_as(values)
        dones = self.dones.reshape_as(values)
        returns = self.returns.reshape_as(values)
        advantage: torch.Tensor | int = 0
        for step in reversed(range(self.num_logical_steps)):
            next_values = branch_last_values if step == self.num_logical_steps - 1 else values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            returns[step] = advantage + values[step]
        self.advantages = self.returns - self.values
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (
                self.advantages.std() + 1.0e-8
            )

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        batch_size = self.num_envs * self.num_transitions_per_env
        if (
            isinstance(num_mini_batches, bool)
            or not isinstance(num_mini_batches, Integral)
            or num_mini_batches < 1
            or num_mini_batches > batch_size
        ):
            raise ValueError("number of mini-batches is invalid for the rollout batch")
        if isinstance(num_epochs, bool) or not isinstance(num_epochs, Integral) or num_epochs < 1:
            raise ValueError("number of minibatch epochs must be positive")
        if self.step != self.num_transitions_per_env:
            raise RuntimeError("rollout storage must be full before minibatching")
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )
        observations = self.observations.flatten(0, 1)
        fields = tuple(
            tensor.flatten(0, 1)
            for tensor in (
                self.actions,
                self.values,
                self.advantages,
                self.returns,
                self.actions_log_prob,
                self.mu,
                self.sigma,
                self.next_critic_observations,
                self.estimator_masks,
            )
        )
        for _ in range(num_epochs):
            for index in range(num_mini_batches):
                start = index * mini_batch_size
                batch_indices = indices[start : start + mini_batch_size]
                selected = tuple(field[batch_indices] for field in fields)
                yield (
                    observations[batch_indices],
                    *selected[:7],
                    (None, None),
                    None,
                    *selected[7:],
                )

    def recurrent_mini_batch_generator(self, *_: object, **__: object) -> Generator:
        raise RuntimeError("recurrent policies are not supported by HIM rollout storage")
        yield
