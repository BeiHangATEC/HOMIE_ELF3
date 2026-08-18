"""Plain-Python ELF3 configuration for the HIM PPO runner."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

from openhomie_isaaclab import elf3_constants as C


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(nested) for nested in value]
    return value


def _mirror_spec() -> dict[str, list[int]]:
    return {
        "dof_mirror_indices": list(C.DOF_MIRROR_INDICES),
        "dof_mirror_signs": list(C.DOF_MIRROR_SIGNS),
        "action_mirror_indices": list(C.ACTION_MIRROR_INDICES),
        "action_mirror_signs": list(C.ACTION_MIRROR_SIGNS),
        "obs_mirror_signs": list(C.OBS_HEAD_MIRROR_SIGNS),
        "critic_tail_mirror_signs": list(C.CRITIC_TAIL_MIRROR_SIGNS),
    }


@dataclass
class Elf3HIMPolicyCfg:
    class_name: str = "HIMActorCritic"
    num_one_step_obs: int = field(default_factory=C.num_one_step_actor_obs)
    actor_history_length: int = C.NUM_ACTOR_HISTORY
    num_one_step_critic_obs: int = field(default_factory=C.num_one_step_critic_obs)
    critic_history_length: int = C.NUM_CRITIC_HISTORY
    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 256])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 256])
    estimator_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    estimator_target_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    estimator_latent_dim: int = C.NUM_ESTIMATOR_LATENT
    estimator_num_prototypes: int = 64
    activation: str = "elu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    state_dependent_std: bool = False
    estimator_temperature: float = 3.0
    estimator_sinkhorn_epsilon: float = 0.05
    estimator_sinkhorn_iterations: int = 3


@dataclass
class Elf3HIMAlgorithmCfg:
    class_name: str = "HIMPPO"
    rnd_cfg: dict[str, Any] | None = None
    symmetry_cfg: dict[str, Any] | None = None
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    normalize_advantage_per_mini_batch: bool = False
    estimator_learning_rate: float | None = None
    estimator_max_grad_norm: float = 10.0
    use_flip: bool = True
    mirror: dict[str, list[int]] = field(default_factory=_mirror_spec)
    symmetry_scale: float = 1.0


@dataclass
class Elf3HIMRunnerCfg:
    class_name: str = "HIMOnPolicyRunner"
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 50
    max_iterations: int = 100_000
    clip_actions: float = 100.0
    experiment_name: str = "elf3_homie_him_isaaclab"
    save_interval: int = 200
    obs_groups: dict[str, list[str]] = field(
        default_factory=lambda: {"policy": ["policy"], "critic": ["critic"]}
    )
    policy: Elf3HIMPolicyCfg = field(default_factory=Elf3HIMPolicyCfg)
    algorithm: Elf3HIMAlgorithmCfg = field(default_factory=Elf3HIMAlgorithmCfg)
    logger: str = "tensorboard"

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)
