"""HIM actor-critic compatible with the rsl-rl 3.1 TensorDict API."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Iterator, Sequence

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP

from .estimator import HIMEstimator


class HIMActorCritic(nn.Module):
    """Feed-forward actor-critic augmented with a History Information Model."""

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        num_one_step_obs: int,
        actor_history_length: int,
        num_one_step_critic_obs: int,
        critic_history_length: int = 1,
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256, 256),
        estimator_hidden_dims: Sequence[int] = (256, 128),
        estimator_target_hidden_dims: Sequence[int] = (256, 128),
        estimator_latent_dim: int = 32,
        estimator_num_prototypes: int = 32,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        normalization_eps: float = 1.0e-2,
        state_dependent_std: bool = False,
        estimator_temperature: float = 3.0,
        estimator_sinkhorn_epsilon: float = 0.05,
        estimator_sinkhorn_iterations: int = 3,
    ) -> None:
        super().__init__()
        self.obs_groups = self._validate_obs_groups(obs, obs_groups)
        if not isinstance(state_dependent_std, bool):
            raise ValueError("state_dependent_std must be boolean")
        if state_dependent_std:
            raise ValueError("state-dependent action noise is not supported")
        self.num_actions = self._positive_int("action", num_actions)
        self.num_one_step_obs = self._positive_int("one-step actor observation", num_one_step_obs)
        self.actor_history_length = self._positive_int("actor history", actor_history_length)
        self.num_one_step_critic_obs = self._positive_int("one-step critic observation", num_one_step_critic_obs)
        self.critic_history_length = self._positive_int("critic history", critic_history_length)
        if self.critic_history_length != 1:
            raise ValueError("critic history length must be one")

        actor_obs_dim = self._group_width(obs, self.obs_groups["policy"])
        critic_obs_dim = self._group_width(obs, self.obs_groups["critic"])
        expected_history_dim = self.num_one_step_obs * self.actor_history_length
        expected_critic_dim = self.num_one_step_critic_obs * self.critic_history_length
        if actor_obs_dim != expected_history_dim:
            raise ValueError(f"actor history dimension must be {expected_history_dim}, got {actor_obs_dim}")
        if critic_obs_dim != expected_critic_dim:
            raise ValueError(f"critic observation dimension must be {expected_critic_dim}, got {critic_obs_dim}")
        velocity_dim = self.num_one_step_critic_obs - self.num_one_step_obs
        if velocity_dim != 3:
            raise ValueError("critic observation must contain a three-value velocity tail")
        if actor_obs_normalization != critic_obs_normalization:
            raise ValueError("actor and critic observation normalization must be enabled together")
        if (
            not isinstance(normalization_eps, Real)
            or isinstance(normalization_eps, bool)
            or not math.isfinite(normalization_eps)
            or normalization_eps <= 0
        ):
            raise ValueError("normalization epsilon must be finite and positive")

        self.actor_obs_normalization = bool(actor_obs_normalization)
        self.critic_obs_normalization = bool(critic_obs_normalization)
        self.actor_obs_normalizer: nn.Module = (
            EmpiricalNormalization(actor_obs_dim, eps=float(normalization_eps))
            if self.actor_obs_normalization
            else nn.Identity()
        )
        self.critic_obs_normalizer: nn.Module = (
            EmpiricalNormalization(critic_obs_dim, eps=float(normalization_eps))
            if self.critic_obs_normalization
            else nn.Identity()
        )
        self.estimator = HIMEstimator(
            history_dim=actor_obs_dim,
            one_step_obs_dim=self.num_one_step_obs,
            latent_dim=estimator_latent_dim,
            encoder_hidden_dims=estimator_hidden_dims,
            target_hidden_dims=estimator_target_hidden_dims,
            num_prototypes=estimator_num_prototypes,
            velocity_dim=velocity_dim,
            activation=activation,
            temperature=estimator_temperature,
            sinkhorn_epsilon=estimator_sinkhorn_epsilon,
            sinkhorn_iterations=estimator_sinkhorn_iterations,
        )
        actor_input_dim = self.num_one_step_obs + velocity_dim + estimator_latent_dim
        self.actor = MLP(actor_input_dim, self.num_actions, list(actor_hidden_dims), activation)
        self.critic = MLP(critic_obs_dim, 1, list(critic_hidden_dims), activation)

        if noise_std_type != "scalar":
            raise ValueError("HIM action noise type must be scalar")
        if (
            not isinstance(init_noise_std, Real)
            or isinstance(init_noise_std, bool)
            or not math.isfinite(init_noise_std)
            or init_noise_std <= 0
        ):
            raise ValueError("initial action noise must be finite and positive")
        self.noise_std_type = noise_std_type
        self.state_dependent_std = False
        self.std = nn.Parameter(torch.full((self.num_actions,), float(init_noise_std)))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _positive_int(name: str, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} dimension must be a positive integer")
        return int(value)

    @staticmethod
    def _validate_obs_groups(obs: TensorDict, obs_groups: dict[str, list[str]]) -> dict[str, list[str]]:
        if "policy" not in obs_groups or not obs_groups["policy"]:
            raise ValueError("policy observation group is required")
        if "critic" not in obs_groups or not obs_groups["critic"]:
            raise ValueError("critic observation group is required")
        policy_groups = list(obs_groups["policy"])
        critic_groups = list(obs_groups["critic"])
        if set(policy_groups) & set(critic_groups):
            raise ValueError("policy and critic observation groups must be distinct")
        for name in policy_groups + critic_groups:
            if name not in obs.keys():
                raise ValueError(f"observation group {name!r} is missing")
            if obs[name].ndim != 2:
                raise ValueError("HIMActorCritic only supports flat observations")
        return {"policy": policy_groups, "critic": critic_groups}

    @staticmethod
    def _group_width(obs: TensorDict, groups: list[str]) -> int:
        return sum(obs[name].shape[-1] for name in groups)

    @staticmethod
    def _cat_groups(obs: TensorDict, groups: list[str]) -> torch.Tensor:
        return torch.cat([obs[name] for name in groups], dim=-1)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._cat_groups(obs, self.obs_groups["policy"])

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._cat_groups(obs, self.obs_groups["critic"])

    def prepare_actor_history(self, obs: TensorDict) -> torch.Tensor:
        return self.actor_obs_normalizer(self.get_actor_obs(obs))

    def prepare_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.critic_obs_normalizer(self.get_critic_obs(obs))

    def actor_input(self, normalized_history: torch.Tensor) -> torch.Tensor:
        expected = self.num_one_step_obs * self.actor_history_length
        if normalized_history.ndim != 2 or normalized_history.shape[-1] != expected:
            raise ValueError(f"actor history dimension must be {expected}")
        velocity, latent = self.estimator(normalized_history)
        latest_frame = normalized_history[..., -self.num_one_step_obs :]
        return torch.cat((latest_frame, velocity.detach(), latent.detach()), dim=-1)

    def _update_distribution(self, obs: TensorDict) -> None:
        if not torch.isfinite(self.std).all() or torch.any(self.std <= 0):
            raise ValueError("action noise must remain finite and positive")
        mean = self.actor(self.actor_input(self.prepare_actor_history(obs)))
        self.distribution = Normal(mean, self.std.expand_as(mean))

    def act(self, obs: TensorDict, **_: Any) -> torch.Tensor:
        self._update_distribution(obs)
        assert self.distribution is not None
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.actor(self.actor_input(self.prepare_actor_history(obs)))

    def evaluate(self, obs: TensorDict, **_: Any) -> torch.Tensor:
        return self.critic(self.prepare_critic_obs(obs))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been initialized")
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been initialized")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been initialized")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been initialized")
        return self.distribution.entropy().sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def estimator_target_normalization(self) -> dict[str, object] | None:
        if not self.actor_obs_normalization:
            return None
        actor = self.actor_obs_normalizer
        critic = self.critic_obs_normalizer
        assert isinstance(actor, EmpiricalNormalization)
        assert isinstance(critic, EmpiricalNormalization)
        if actor.eps != critic.eps:
            raise RuntimeError("actor and critic normalization epsilons must match")
        return {
            "actor_history_mean": actor.mean,
            "actor_history_std": actor.std,
            "critic_mean": critic.mean,
            "critic_std": critic.std,
            "eps": actor.eps,
        }

    def policy_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.actor.parameters()
        yield from self.critic.parameters()
        yield self.std

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def forward(self, *_: Any) -> None:
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
