"""Fresh single-stage ELF3 training with the G1 command mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


TASK_ID = "OpenHomie-Elf3-Homie-SingleStage-v0"
COMMAND_PROFILE = "g1_single_stage"
_DEVICE_PREFIX = "cuda:"

_ENV_ENTRY_POINT = (
    "openhomie_isaaclab.tasks.locomotion.elf3."
    "elf3_single_stage_env:Elf3SingleStageEnv"
)
_ENV_CFG_ENTRY_POINT = (
    "openhomie_isaaclab.tasks.locomotion.elf3."
    "elf3_single_stage_env_cfg:Elf3SingleStageEnvCfg"
)
_AGENT_CFG_ENTRY_POINT = (
    "openhomie_isaaclab.tasks.locomotion.elf3.agents."
    "him_ppo_cfg:Elf3HIMRunnerCfg"
)


@dataclass(frozen=True)
class SingleStageTrainRequest:
    run_dir: Path
    device: str
    seed: int
    num_envs: int
    iterations: int
    headless: bool
    command_profile: str = COMMAND_PROFILE

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "run_dir": str(self.run_dir),
        }


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _device(value: str) -> str:
    if value == "cpu":
        return value
    if value.startswith(_DEVICE_PREFIX) and value.removeprefix(_DEVICE_PREFIX).isdigit():
        return value
    raise argparse.ArgumentTypeError("device must be cpu or cuda:<index>")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ELF3 with a single G1-proportioned command profile",
        allow_abbrev=False,
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", required=True, type=_device)
    parser.add_argument("--seed", required=True, type=_nonnegative)
    parser.add_argument("--num-envs", required=True, type=_positive)
    parser.add_argument("--iterations", required=True, type=_positive)
    parser.add_argument("--headless", action="store_true")
    return parser


def parse_request(argv: Sequence[str] | None = None) -> SingleStageTrainRequest:
    args = build_parser().parse_args(argv)
    return SingleStageTrainRequest(
        run_dir=Path(args.run_dir).expanduser().resolve(),
        device=args.device,
        seed=args.seed,
        num_envs=args.num_envs,
        iterations=args.iterations,
        headless=args.headless,
    )


def request_log_payload(request: SingleStageTrainRequest) -> dict[str, Any]:
    """Return the launch request printed before Isaac Lab starts."""
    return {
        "event": "ELF3_SINGLE_STAGE_REQUEST",
        "task_id": TASK_ID,
        "profile": request.command_profile,
        "request": request.to_dict(),
    }


def effective_config_log_payload(
    request: SingleStageTrainRequest, env_cfg: Any, agent_cfg: Any
) -> dict[str, Any]:
    """Return the stable training settings users need beside live progress."""
    algorithm = agent_cfg.algorithm
    policy = agent_cfg.policy
    return {
        "event": "ELF3_SINGLE_STAGE_EFFECTIVE_CONFIG",
        "task_id": TASK_ID,
        "profile": request.command_profile,
        "commands": {
            "mode_probabilities": {
                "walk": 0.5,
                "height_hold": 1.0 / 3.0,
                "stand": 1.0 / 6.0,
            },
            "height_range": list(env_cfg.single_stage_height_range),
            "stand_height": env_cfg.single_stage_stand_height,
            "lin_vel_x_range": list(env_cfg.lin_vel_x_range),
            "lin_vel_y_range": list(env_cfg.lin_vel_y_range),
            "ang_vel_yaw_range": list(env_cfg.ang_vel_yaw_range),
        },
        "network": {
            "actor_hidden_dims": list(policy.actor_hidden_dims),
            "critic_hidden_dims": list(policy.critic_hidden_dims),
            "estimator_hidden_dims": list(policy.estimator_hidden_dims),
            "estimator_target_hidden_dims": list(policy.estimator_target_hidden_dims),
            "estimator_latent_dim": policy.estimator_latent_dim,
            "estimator_num_prototypes": policy.estimator_num_prototypes,
        },
        "ppo": {
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "max_iterations": agent_cfg.max_iterations,
            "save_interval": agent_cfg.save_interval,
            "learning_rate": algorithm.learning_rate,
            "schedule": algorithm.schedule,
            "entropy_coef": algorithm.entropy_coef,
            "estimator_learning_rate": algorithm.estimator_learning_rate,
            "estimator_lr_follows_policy": algorithm.estimator_learning_rate is None,
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {
            "start": _json_safe(value.start),
            "stop": _json_safe(value.stop),
            "step": _json_safe(value.step),
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(nested) for nested in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON output contains NaN or infinity")
    return value


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(encoded)


def _create_run_dir(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"single-stage run directory already exists: {path}")
    path.mkdir(mode=0o755, parents=True)
    return path


def _adapt_bool_dones(env):
    original_step = env.step

    def step(actions):
        observations, rewards, dones, extras = original_step(actions)
        return observations, rewards, dones.bool(), extras

    env.step = step
    return env


def _register_single_stage_task() -> None:
    import gymnasium as gym

    import openhomie_isaaclab.tasks.locomotion.elf3  # noqa: F401

    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point=_ENV_ENTRY_POINT,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": _ENV_CFG_ENTRY_POINT,
                "rsl_rl_cfg_entry_point": _AGENT_CFG_ENTRY_POINT,
            },
        )
        return
    specification = gym.spec(TASK_ID)
    if (
        specification.entry_point != _ENV_ENTRY_POINT
        or specification.kwargs.get("env_cfg_entry_point") != _ENV_CFG_ENTRY_POINT
        or specification.kwargs.get("rsl_rl_cfg_entry_point") != _AGENT_CFG_ENTRY_POINT
    ):
        raise RuntimeError("ELF3 single-stage task registration does not match its contract")


def run_training(request: SingleStageTrainRequest) -> dict[str, Any]:
    """Run training after the caller has launched Isaac Lab's application."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry

    import gymnasium as gym
    import torch

    from openhomie_isaaclab.him_rl.runner import HIMOnPolicyRunner

    _register_single_stage_task()
    run_dir = _create_run_dir(request.run_dir)
    environment = None
    try:
        torch.manual_seed(request.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.seed)
        env_cfg = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
        env_cfg.scene.num_envs = request.num_envs
        env_cfg.seed = request.seed
        env_cfg.sim.device = request.device
        env_cfg.log_dir = str(run_dir)
        if env_cfg.command_profile != request.command_profile:
            raise RuntimeError("ELF3 single-stage configuration profile mismatch")
        agent_cfg.seed = request.seed
        agent_cfg.device = request.device
        agent_cfg.max_iterations = request.iterations
        print(
            json.dumps(
                effective_config_log_payload(request, env_cfg, agent_cfg),
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        _write_json_once(
            run_dir / "manifest.json",
            {
                "schema_version": 1,
                "command": "train",
                "profile": request.command_profile,
                "cli": request.to_dict(),
                "configs": {
                    "env": env_cfg.to_dict(),
                    "agent": agent_cfg.to_dict(),
                },
            },
        )
        raw_env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
        environment = _adapt_bool_dones(
            RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        )
        runner = HIMOnPolicyRunner(
            environment,
            agent_cfg.to_dict(),
            log_dir=str(run_dir),
            device=request.device,
        )
        runner.learn(request.iterations, init_at_random_ep_len=True)
        checkpoint = run_dir / f"model_{request.iterations}.pt"
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise RuntimeError("single-stage training did not produce its final checkpoint")
        result = {
            "status": "PASS",
            "profile": request.command_profile,
            "final_iteration": request.iterations,
            "checkpoint": {
                "path": str(checkpoint.resolve(strict=True)),
                "sha256": _sha256_file(checkpoint),
            },
            "metrics": runner.get_training_metrics(),
        }
        _write_json_once(run_dir / "result.json", result)
        return result
    except Exception as error:
        if run_dir.is_dir() and not (run_dir / "result.json").exists():
            _write_json_once(
                run_dir / "result.json",
                {"status": "FAIL", "profile": request.command_profile, "error": str(error)},
            )
        raise
    finally:
        if environment is not None:
            environment.close()
