"""Atomic dual-optimizer PPO for History Information Model policies."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.algorithms import PPO

from .storage import HIMRolloutStorage
from .symmetry import MirrorTransform


class NonFiniteTrainingError(RuntimeError):
    """Raised before a non-finite training update can be committed."""


class HIMPPO(PPO):
    """PPO with branch-aware symmetry and an independently owned HIM optimizer."""

    def __init__(
        self,
        policy,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1.0e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        estimator_learning_rate: float | None = 1.0e-3,
        estimator_max_grad_norm: float | None = None,
        use_flip: bool = False,
        mirror: dict[str, list[int]] | None = None,
        symmetry_scale: float = 1.0,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        if rnd_cfg is not None:
            raise ValueError("RND is not supported by HIMPPO")
        if symmetry_cfg is not None:
            raise ValueError("generic symmetry_cfg is not supported by HIMPPO")
        if multi_gpu_cfg is not None:
            raise ValueError("multi-GPU training is not supported by HIMPPO")
        if policy.is_recurrent:
            raise ValueError("recurrent policies are not supported by HIMPPO")
        if use_flip and mirror is None:
            raise ValueError("mirror specification is required when flip symmetry is enabled")
        if estimator_learning_rate is not None and (
            not math.isfinite(estimator_learning_rate) or estimator_learning_rate <= 0
        ):
            raise ValueError("estimator learning rate must be finite and positive")

        self.policy = policy.to(device)
        self.device = device
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.estimator_max_grad_norm = max_grad_norm if estimator_max_grad_norm is None else estimator_max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.is_multi_gpu = False
        self.gpu_global_rank = 0
        self.gpu_world_size = 1
        self.rnd = None
        self.symmetry = None

        policy_parameters = tuple(policy.policy_parameters())
        estimator_parameters = tuple(policy.estimator.parameters())
        if not policy_parameters or not estimator_parameters:
            raise ValueError("policy and estimator parameter sets must be nonempty")
        if {id(parameter) for parameter in policy_parameters} & {
            id(parameter) for parameter in estimator_parameters
        }:
            raise ValueError("policy and estimator parameter sets must be disjoint")
        self.optimizer = torch.optim.Adam(policy_parameters, lr=learning_rate)
        self.estimator_lr_follows_policy = estimator_learning_rate is None
        effective_estimator_lr = learning_rate if estimator_learning_rate is None else estimator_learning_rate
        self.estimator_learning_rate = effective_estimator_lr
        self.estimator_optimizer = torch.optim.Adam(estimator_parameters, lr=effective_estimator_lr)

        self.use_flip = bool(use_flip)
        self.symmetry_scale = symmetry_scale
        self.mirror_transform = (
            MirrorTransform(
                mirror,
                policy.num_one_step_obs,
                policy.num_one_step_critic_obs,
                policy.actor_history_length,
                policy.critic_history_length,
            )
            if self.use_flip
            else None
        )
        self.storage: HIMRolloutStorage | None = None
        self.transition = HIMRolloutStorage.Transition()
        self.transition_sym = HIMRolloutStorage.Transition() if self.use_flip else None

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        self.storage = HIMRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            [self.policy.num_one_step_critic_obs],
            self.device,
            num_branches=2 if self.use_flip else 1,
        )

    def _record_action(self, transition: HIMRolloutStorage.Transition, obs: TensorDict) -> torch.Tensor:
        actions = self.policy.act(obs)
        transition.observations = obs
        transition.actions = actions.detach()
        transition.values = self.policy.evaluate(obs).detach()
        transition.actions_log_prob = self.policy.get_actions_log_prob(actions).detach()
        transition.action_mean = self.policy.action_mean.detach()
        transition.action_sigma = self.policy.action_std.detach()
        return transition.actions

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = self._record_action(self.transition, obs)
        if self.use_flip:
            assert self.transition_sym is not None and self.mirror_transform is not None
            mirrored_obs = self.mirror_transform.observations(obs, self.policy.obs_groups)
            self._record_action(self.transition_sym, mirrored_obs)
        return actions

    def _next_critic_supervision(
        self, obs: TensorDict, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_critic = self.policy.get_critic_obs(obs).clone()
        if dones.dtype != torch.bool or dones.shape != (next_critic.shape[0],):
            raise ValueError("dones must be boolean with one value per environment")
        mask = ~dones.clone()
        has_terminal = "terminal_critic_obs" in extras
        has_terminal_mask = "terminal_critic_obs_mask" in extras
        if has_terminal != has_terminal_mask:
            raise ValueError("terminal critic observation and mask must be provided together")
        if not has_terminal:
            return next_critic, mask
        terminal = extras["terminal_critic_obs"]
        terminal_mask = extras["terminal_critic_obs_mask"]
        if terminal.shape != next_critic.shape:
            raise ValueError("terminal critic observation shape is invalid")
        if terminal_mask.dtype != torch.bool or terminal_mask.shape not in (
            (next_critic.shape[0],),
            (next_critic.shape[0], 1),
        ):
            raise ValueError("terminal critic observation mask must be boolean with one value per environment")
        terminal_mask = terminal_mask.reshape(-1)
        if torch.any(terminal_mask & ~dones):
            raise ValueError("terminal critic mask may select only completed environments")
        next_critic[terminal_mask] = terminal[terminal_mask]
        mask[terminal_mask] = True
        return next_critic, mask

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        if self.storage is None:
            raise RuntimeError("rollout storage has not been initialized")
        self.policy.update_normalization(obs)
        next_critic, estimator_mask = self._next_critic_supervision(obs, dones, extras)
        base_rewards = rewards.clone()
        self.transition.rewards = base_rewards.clone()
        self.transition.dones = dones
        self.transition.next_critic_observations = next_critic
        self.transition.estimator_masks = estimator_mask
        time_outs = extras.get("time_outs")
        if time_outs is not None:
            time_outs = time_outs.to(self.device).reshape(-1, 1)
            self.transition.rewards += self.gamma * (self.transition.values * time_outs).squeeze(-1)
        self.storage.add_transitions(self.transition)

        if self.use_flip:
            assert self.transition_sym is not None and self.mirror_transform is not None
            self.transition_sym.rewards = base_rewards.clone()
            if time_outs is not None:
                self.transition_sym.rewards += self.gamma * (self.transition_sym.values * time_outs).squeeze(-1)
            self.transition_sym.dones = dones
            self.transition_sym.next_critic_observations = self.mirror_transform.critic(next_critic)
            self.transition_sym.estimator_masks = estimator_mask
            self.storage.add_transitions(self.transition_sym)
            self.transition_sym.clear()
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        if self.storage is None:
            raise RuntimeError("rollout storage has not been initialized")
        values = self.policy.evaluate(obs).detach()
        if self.use_flip:
            assert self.mirror_transform is not None
            mirrored_obs = self.mirror_transform.observations(obs, self.policy.obs_groups)
            values = torch.stack((values, self.policy.evaluate(mirrored_obs).detach()))
        self.storage.compute_returns(
            values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def _adapt_learning_rate(
        self,
        current_mu: torch.Tensor,
        current_sigma: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
    ) -> None:
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        with torch.inference_mode():
            kl = torch.sum(
                torch.log(current_sigma / old_sigma + 1.0e-5)
                + (old_sigma.square() + (old_mu - current_mu).square())
                / (2.0 * current_sigma.square())
                - 0.5,
                dim=-1,
            ).mean()
        if kl > self.desired_kl * 2.0:
            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
        elif 0.0 < kl < self.desired_kl / 2.0:
            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        if self.estimator_lr_follows_policy:
            self.estimator_learning_rate = self.learning_rate
            for group in self.estimator_optimizer.param_groups:
                group["lr"] = self.learning_rate

    def _update_estimator(
        self,
        normalized_history: torch.Tensor,
        next_critic_observations: torch.Tensor,
        estimator_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy.estimator.loss(
            normalized_history,
            next_critic_observations,
            estimator_masks.reshape(-1),
            normalization=self.policy.estimator_target_normalization(),
        )

    @staticmethod
    def _require_finite(name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value).all():
            raise NonFiniteTrainingError(f"{name} is non-finite")

    @classmethod
    def _require_finite_gradients(cls, name: str, parameters: tuple[torch.nn.Parameter, ...]) -> None:
        for parameter in parameters:
            if parameter.grad is not None:
                cls._require_finite(f"{name} gradient", parameter.grad)

    def _restore_update(self, model_state: dict, policy_state: dict, estimator_state: dict) -> None:
        self.policy.load_state_dict(model_state)
        self.optimizer.load_state_dict(policy_state)
        self.estimator_optimizer.load_state_dict(estimator_state)
        self.optimizer.zero_grad()
        self.estimator_optimizer.zero_grad()

    def update(self) -> dict[str, float]:
        model_state = copy.deepcopy(self.policy.state_dict())
        policy_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        estimator_optimizer_state = copy.deepcopy(self.estimator_optimizer.state_dict())
        learning_rate = self.learning_rate
        estimator_learning_rate = self.estimator_learning_rate
        try:
            return self._update_impl()
        except Exception:
            self._restore_update(model_state, policy_optimizer_state, estimator_optimizer_state)
            self.learning_rate = learning_rate
            self.estimator_learning_rate = estimator_learning_rate
            raise

    def _update_impl(self) -> dict[str, float]:
        if self.storage is None:
            raise RuntimeError("rollout storage has not been initialized")
        totals = {
            "value_function": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "estimator_velocity": 0.0,
            "estimator_swap": 0.0,
            "actor_symmetry": 0.0,
            "critic_symmetry": 0.0,
        }
        updates = 0
        policy_parameters = tuple(self.policy.policy_parameters())
        estimator_parameters = tuple(self.policy.estimator.parameters())
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                _,
                _,
                next_critic_batch,
                estimator_masks_batch,
            ) = batch
            if self.normalize_advantage_per_mini_batch:
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                    advantages_batch.std() + 1.0e-8
                )

            self.optimizer.zero_grad()
            self.estimator_optimizer.zero_grad()
            self.policy.act(obs_batch)
            actions_log_prob = self.policy.get_actions_log_prob(actions_batch)
            values = self.policy.evaluate(obs_batch)
            mu = self.policy.action_mean
            sigma = self.policy.action_std
            entropy = self.policy.entropy
            self._adapt_learning_rate(mu, sigma, old_mu_batch, old_sigma_batch)

            ratio = torch.exp(actions_log_prob - old_actions_log_prob_batch.squeeze(-1))
            advantages = advantages_batch.squeeze(-1)
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (values - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.maximum(
                    (values - returns_batch).square(),
                    (value_clipped - returns_batch).square(),
                ).mean()
            else:
                value_loss = (returns_batch - values).square().mean()
            policy_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            actor_symmetry = policy_loss.new_zeros(())
            critic_symmetry = policy_loss.new_zeros(())

            if self.use_flip:
                assert self.mirror_transform is not None
                mirrored_obs = self.mirror_transform.observations(obs_batch, self.policy.obs_groups)
                mirrored_actions = self.policy.act_inference(mirrored_obs)
                target_actions = self.mirror_transform.actions(self.policy.act_inference(obs_batch)).detach()
                actor_symmetry = (mirrored_actions - target_actions).square().sum(dim=-1).mean()
                mirrored_values = self.policy.evaluate(mirrored_obs)
                target_critic = self.policy.evaluate(obs_batch).detach()
                critic_symmetry = F.mse_loss(mirrored_values, target_critic)
                policy_loss = (
                    policy_loss
                    + self.symmetry_scale * actor_symmetry
                    + self.symmetry_scale * critic_symmetry
                )

            normalized_history = self.policy.prepare_actor_history(obs_batch)
            velocity_loss, swap_loss = self._update_estimator(
                normalized_history, next_critic_batch, estimator_masks_batch
            )
            estimator_loss = velocity_loss + swap_loss
            for name, loss in (
                ("policy loss", policy_loss),
                ("velocity loss", velocity_loss),
                ("swap loss", swap_loss),
                ("estimator loss", estimator_loss),
            ):
                self._require_finite(name, loss)

            has_estimator_samples = bool(estimator_masks_batch.any().item())
            if has_estimator_samples:
                estimator_loss.backward()
            policy_loss.backward()
            self._require_finite_gradients("estimator", estimator_parameters)
            self._require_finite_gradients("policy", policy_parameters)
            if has_estimator_samples:
                estimator_norm = torch.nn.utils.clip_grad_norm_(
                    estimator_parameters, self.estimator_max_grad_norm
                )
                self._require_finite("estimator gradient norm", torch.as_tensor(estimator_norm))
            policy_norm = torch.nn.utils.clip_grad_norm_(policy_parameters, self.max_grad_norm)
            self._require_finite("policy gradient norm", torch.as_tensor(policy_norm))

            model_state = copy.deepcopy(self.policy.state_dict())
            policy_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
            estimator_optimizer_state = copy.deepcopy(self.estimator_optimizer.state_dict())
            try:
                self.optimizer.step()
                if has_estimator_samples:
                    self.estimator_optimizer.step()
                    self.policy.estimator.normalize_prototypes_()
                for parameter in self.policy.parameters():
                    self._require_finite("model parameter", parameter)
            except Exception:
                self._restore_update(model_state, policy_optimizer_state, estimator_optimizer_state)
                raise

            totals["value_function"] += value_loss.item()
            totals["surrogate"] += surrogate_loss.item()
            totals["entropy"] += entropy.mean().item()
            totals["estimator_velocity"] += velocity_loss.item()
            totals["estimator_swap"] += swap_loss.item()
            totals["actor_symmetry"] += actor_symmetry.item()
            totals["critic_symmetry"] += critic_symmetry.item()
            updates += 1

        self.storage.clear()
        self.optimizer.zero_grad()
        self.estimator_optimizer.zero_grad()
        return {name: value / updates for name, value in totals.items()}
