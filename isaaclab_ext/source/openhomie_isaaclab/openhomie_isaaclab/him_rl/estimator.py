"""History Information Model estimator and balanced prototype objective."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.networks import MLP


def _is_finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class TargetLayout:
    """Named widths used to construct estimator next-state targets."""

    one_step_obs_dim: int
    velocity_dim: int = 3
    dropped_velocity_command_dim: int = 3

    def __post_init__(self) -> None:
        for name, value in (
            ("one-step observation", self.one_step_obs_dim),
            ("velocity", self.velocity_dim),
            ("dropped velocity command", self.dropped_velocity_command_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} dimension must be a positive integer")
        if self.dropped_velocity_command_dim >= self.one_step_obs_dim:
            raise ValueError("dropped velocity command dimension must be smaller than the actor frame")

    @property
    def critic_obs_dim(self) -> int:
        return self.one_step_obs_dim + self.velocity_dim

    @property
    def target_obs_dim(self) -> int:
        return self.one_step_obs_dim - self.dropped_velocity_command_dim + self.velocity_dim


def _validate_stat(name: str, value: torch.Tensor, width: int) -> None:
    if value.ndim != 1 or value.shape[0] < width:
        raise ValueError(f"{name} must be a rank-1 tensor with at least {width} values")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def extract_estimator_targets(
    critic_observations: torch.Tensor,
    *,
    actor_history_mean: torch.Tensor | None = None,
    actor_history_std: torch.Tensor | None = None,
    critic_mean: torch.Tensor | None = None,
    critic_std: torch.Tensor | None = None,
    eps: float = 1.0e-8,
    layout: TargetLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract normalized true velocity and target-encoder input by named regions."""
    if critic_observations.ndim != 2:
        raise ValueError("critic observations must be rank-2")
    if not torch.isfinite(critic_observations).all():
        raise ValueError("critic observations must be finite")
    if not _is_finite_real(eps) or eps < 0:
        raise ValueError("normalization epsilon must be finite and nonnegative")

    if layout is None:
        velocity_dim = 3
        layout = TargetLayout(critic_observations.shape[-1] - velocity_dim, velocity_dim)
    if critic_observations.shape[-1] != layout.critic_obs_dim:
        raise ValueError(
            f"critic observation dimension must be {layout.critic_obs_dim}, "
            f"got {critic_observations.shape[-1]}"
        )

    actor_frame = critic_observations[:, : layout.one_step_obs_dim]
    velocity = critic_observations[:, layout.one_step_obs_dim :]
    stats = (actor_history_mean, actor_history_std, critic_mean, critic_std)
    if any(value is not None for value in stats):
        if any(value is None for value in stats):
            raise ValueError("actor-history and critic normalization statistics must be supplied together")
        assert actor_history_mean is not None and actor_history_std is not None
        assert critic_mean is not None and critic_std is not None
        _validate_stat("actor history mean", actor_history_mean, layout.one_step_obs_dim)
        _validate_stat("actor history std", actor_history_std, layout.one_step_obs_dim)
        _validate_stat("critic mean", critic_mean, layout.critic_obs_dim)
        _validate_stat("critic std", critic_std, layout.critic_obs_dim)
        actor_mean = actor_history_mean[-layout.one_step_obs_dim :]
        actor_std = actor_history_std[-layout.one_step_obs_dim :]
        velocity_mean = critic_mean[-layout.velocity_dim :]
        velocity_std = critic_std[-layout.velocity_dim :]
        if torch.any(actor_std + eps <= 0) or torch.any(velocity_std + eps <= 0):
            raise ValueError("normalization standard deviations must be positive")
        actor_frame = (actor_frame - actor_mean) / (actor_std + eps)
        velocity = (velocity - velocity_mean) / (velocity_std + eps)

    target_actor = actor_frame[:, layout.dropped_velocity_command_dim :]
    target = torch.cat((target_actor, velocity), dim=-1)
    if target.shape[-1] != layout.target_obs_dim:
        raise RuntimeError("estimator target layout cursor mismatch")
    return velocity, target


def sinkhorn(logits: torch.Tensor, *, epsilon: float, iterations: int) -> torch.Tensor:
    """Compute balanced assignments in log space for numerical stability."""
    if logits.ndim != 2:
        raise ValueError("Sinkhorn logits must be rank-2")
    if logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("Sinkhorn logits must have at least one sample and prototype")
    if not torch.isfinite(logits).all():
        raise ValueError("Sinkhorn logits must be finite")
    if not _is_finite_real(epsilon) or epsilon <= 0:
        raise ValueError("Sinkhorn epsilon must be positive")
    if isinstance(iterations, bool) or not isinstance(iterations, Integral) or iterations < 1:
        raise ValueError("Sinkhorn iterations must be at least one")

    batch_size, num_prototypes = logits.shape
    log_assignments = (logits / float(epsilon)).transpose(0, 1)
    log_assignments = log_assignments - torch.logsumexp(log_assignments.reshape(-1), dim=0)
    log_prototype_mass = -torch.log(logits.new_tensor(float(num_prototypes)))
    log_sample_mass = -torch.log(logits.new_tensor(float(batch_size)))
    for _ in range(int(iterations)):
        log_assignments = (
            log_assignments
            - torch.logsumexp(log_assignments, dim=1, keepdim=True)
            + log_prototype_mass
        )
        log_assignments = (
            log_assignments
            - torch.logsumexp(log_assignments, dim=0, keepdim=True)
            + log_sample_mass
        )
    log_assignments = log_assignments + torch.log(logits.new_tensor(float(batch_size)))
    assignments = torch.exp(log_assignments).transpose(0, 1)
    if not torch.isfinite(assignments).all():
        raise RuntimeError("Sinkhorn produced non-finite assignments")
    return assignments


class HIMEstimator(nn.Module):
    """Predict base velocity and learn a prototype-structured history latent."""

    def __init__(
        self,
        *,
        history_dim: int,
        one_step_obs_dim: int,
        latent_dim: int,
        encoder_hidden_dims: Sequence[int],
        target_hidden_dims: Sequence[int],
        num_prototypes: int,
        velocity_dim: int = 3,
        dropped_velocity_command_dim: int = 3,
        activation: str = "elu",
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 3,
        temperature: float = 3.0,
    ) -> None:
        super().__init__()
        dimensions = {
            "history": history_dim,
            "latent": latent_dim,
            "prototype": num_prototypes,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} dimension must be a positive integer")
        if not encoder_hidden_dims or not target_hidden_dims:
            raise ValueError("estimator hidden dimensions must be nonempty")
        if not _is_finite_real(temperature) or temperature <= 0:
            raise ValueError("estimator temperature must be positive")
        # Validate Sinkhorn configuration without manufacturing a training sample.
        if not _is_finite_real(sinkhorn_epsilon) or sinkhorn_epsilon <= 0:
            raise ValueError("Sinkhorn epsilon must be positive")
        if isinstance(sinkhorn_iterations, bool) or not isinstance(sinkhorn_iterations, Integral) or sinkhorn_iterations < 1:
            raise ValueError("Sinkhorn iterations must be at least one")

        self.history_dim = int(history_dim)
        self.latent_dim = int(latent_dim)
        self.layout = TargetLayout(
            one_step_obs_dim=int(one_step_obs_dim),
            velocity_dim=int(velocity_dim),
            dropped_velocity_command_dim=int(dropped_velocity_command_dim),
        )
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.temperature = float(temperature)
        self.source_encoder = MLP(
            self.history_dim,
            self.layout.velocity_dim + self.latent_dim,
            list(encoder_hidden_dims),
            activation,
        )
        self.target_encoder = MLP(
            self.layout.target_obs_dim,
            self.latent_dim,
            list(target_hidden_dims),
            activation,
        )
        self.prototypes = nn.Linear(self.latent_dim, int(num_prototypes), bias=False)
        self.normalize_prototypes_()

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history.ndim != 2 or history.shape[-1] != self.history_dim:
            raise ValueError(f"history dimension must be {self.history_dim}")
        if not torch.isfinite(history).all():
            raise ValueError("history must be finite")
        encoded = self.source_encoder(history)
        velocity = encoded[:, : self.layout.velocity_dim]
        latent = F.normalize(encoded[:, self.layout.velocity_dim :], dim=-1)
        return velocity, latent

    def extract_targets(self, critic_observations: torch.Tensor, **normalization):
        return extract_estimator_targets(
            critic_observations, layout=self.layout, **normalization
        )

    def prototype_scores(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"latent dimension must be {self.latent_dim}")
        normalized_weight = F.normalize(self.prototypes.weight, dim=-1)
        return F.linear(latent, normalized_weight)

    @torch.no_grad()
    def normalize_prototypes_(self) -> None:
        self.prototypes.weight.copy_(F.normalize(self.prototypes.weight, dim=-1))

    def loss(
        self,
        history: torch.Tensor,
        next_critic_observations: torch.Tensor,
        mask: torch.Tensor | None = None,
        normalization: Mapping[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history.ndim != 2 or history.shape[-1] != self.history_dim:
            raise ValueError(f"history dimension must be {self.history_dim}")
        if next_critic_observations.ndim != 2 or next_critic_observations.shape[0] != history.shape[0]:
            raise ValueError("critic observations must have the same batch size as history")
        if mask is None:
            valid = torch.ones(history.shape[0], dtype=torch.bool, device=history.device)
        else:
            if mask.ndim == 2 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            if mask.ndim != 1 or mask.shape[0] != history.shape[0] or mask.dtype != torch.bool:
                raise ValueError("estimator mask must be boolean with one value per sample")
            valid = mask.to(device=history.device)
        if not torch.any(valid):
            zero = sum((parameter.sum() * 0.0 for parameter in self.parameters()), history.sum() * 0.0)
            return zero, zero.clone()

        selected_history = history[valid]
        selected_critic = next_critic_observations[valid]
        predicted_velocity, source_latent = self(selected_history)
        target_velocity, target_observations = self.extract_targets(
            selected_critic, **({} if normalization is None else normalization)
        )
        target_latent = F.normalize(self.target_encoder(target_observations), dim=-1)
        source_scores = self.prototype_scores(source_latent)
        target_scores = self.prototype_scores(target_latent)
        with torch.no_grad():
            source_assignments = sinkhorn(
                source_scores.detach(),
                epsilon=self.sinkhorn_epsilon,
                iterations=self.sinkhorn_iterations,
            )
            target_assignments = sinkhorn(
                target_scores.detach(),
                epsilon=self.sinkhorn_epsilon,
                iterations=self.sinkhorn_iterations,
            )
        source_log_prob = F.log_softmax(source_scores / self.temperature, dim=-1)
        target_log_prob = F.log_softmax(target_scores / self.temperature, dim=-1)
        swap_loss = -0.5 * (
            source_assignments * target_log_prob
            + target_assignments * source_log_prob
        ).mean()
        velocity_loss = F.mse_loss(predicted_velocity, target_velocity)
        if not torch.isfinite(velocity_loss) or not torch.isfinite(swap_loss):
            raise RuntimeError("estimator loss is non-finite")
        return velocity_loss, swap_loss
