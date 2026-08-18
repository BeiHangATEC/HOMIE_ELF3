#!/usr/bin/env python3
"""Collect the immutable ELF3 C3 signed-response grid after Isaac Sim starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from openhomie_isaaclab.workflows.elf3_c3_signed_response import (
    FROZEN_C1_SHA256,
    NUM_ENVS,
    REQUIRED_ARRAYS,
    STEPS,
    canonical_grid_actions,
    validate_plan,
    validate_trajectory_arrays,
)
from openhomie_isaaclab.workflows.elf3_run import sha256_file, validate_manifest, write_json_once


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "OpenHomie-Elf3-Homie-Direct-v0"
DEVICE = "cuda:0"
_SHA256_LENGTH = 64


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


@dataclass(frozen=True)
class GridRequest:
    plan: Path
    output_root: Path
    device: str = DEVICE


def _exists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _regular_file(value: str | os.PathLike[str], name: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _new_directory(value: str | os.PathLike[str], name: str) -> Path:
    path = Path(value).expanduser()
    if _exists(path):
        raise FileExistsError(f"{name} already exists: {path}")
    parent = path.parent.resolve(strict=True)
    resolved = parent / path.name
    if not path.name or _exists(resolved):
        raise FileExistsError(f"{name} already exists: {resolved}")
    return resolved


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    source = _regular_file(path, name)

    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains {value}")

    payload = json.loads(
        source.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return payload


def _load_plan(path: Path) -> Mapping[str, Any]:
    payload = _load_json(path, "signed-response plan")
    validate_plan(payload)
    return payload


def parse_request(argv: Sequence[str] | None = None) -> GridRequest:
    parser = _ArgumentParser(
        description="Collect the fixed 33-run ELF3 C3 signed-response grid"
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", choices=(DEVICE,), default=DEVICE)
    args = parser.parse_args(argv)
    plan = _regular_file(args.plan, "signed-response plan")
    _load_plan(plan)
    return GridRequest(
        plan=plan,
        output_root=_new_directory(args.output_root, "grid output root"),
        device=args.device,
    )


def _require_source(plan: Mapping[str, Any]) -> tuple[Path, Path, str, str]:
    source = plan["source"]
    checkpoint = _regular_file(source["checkpoint_path"], "source checkpoint")
    manifest = _regular_file(source["manifest_path"], "source manifest")
    checkpoint_hash = sha256_file(checkpoint)
    manifest_hash = sha256_file(manifest)
    if checkpoint_hash != source["checkpoint_sha256"]:
        raise RuntimeError("source checkpoint SHA-256 does not match the plan")
    if manifest_hash != source["manifest_sha256"]:
        raise RuntimeError("source manifest SHA-256 does not match the plan")
    payload = _load_json(manifest, "source manifest")
    validate_manifest(payload)
    if (
        payload.get("command") != "train"
        or payload.get("start_iteration") != 2000
        or payload.get("iterations") != 2000
        or payload.get("configs", {}).get("env", {}).get("command_stage")
        != "S0"
    ):
        raise RuntimeError("source manifest is not the S0 2000-to-4000 continuation")
    if source["checkpoint_iteration"] != 4000 or checkpoint.name != "model_4000.pt":
        raise RuntimeError("signed-response grid requires model_4000.pt")
    return checkpoint, manifest, checkpoint_hash, manifest_hash


def _verify_frozen_sources() -> None:
    for relative, expected in FROZEN_C1_SHA256.items():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"missing frozen C1 source: {relative}")
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen C1 source changed: {relative}")


def _array_copy(value, dtype):
    array = value.detach().cpu().numpy()
    return array.astype(dtype, copy=True)


def _require_finite(name: str, value) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise RuntimeError(f"{name} is non-finite")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _require_finite(f"{name}.{key}", nested)
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _require_finite(f"{name}[{index}]", nested)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"{name} is non-finite")


def _new_arrays() -> dict[str, Any]:
    import numpy as np

    arrays: dict[str, Any] = {}
    for name, spec in REQUIRED_ARRAYS.items():
        arrays[name] = np.empty(spec["shape"], dtype=np.dtype(spec["dtype"]))
    arrays["step_index"][:] = np.arange(STEPS, dtype=np.int64)
    return arrays


def _run_action(request: GridRequest, plan: Mapping[str, Any], action: Mapping[str, Any], checkpoint: Path) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from openhomie_isaaclab import elf3_constants as constants
    from openhomie_isaaclab.him_rl.runner import HIMOnPolicyRunner
    from openhomie_isaaclab.workflows import elf3_sim

    action_id = action["action_id"]
    run_dir = request.output_root / action_id
    run_dir.mkdir(mode=0o755)
    raw_env = None
    env = None
    runner = None
    started = time.perf_counter()
    try:
        elf3_sim._seed_runtime(action["seed"])
        loader_request = SimpleNamespace(
            command="play",
            run_dir=run_dir,
            num_envs=NUM_ENVS,
            seed=action["seed"],
            device=request.device,
        )
        env_cfg, agent_cfg, configs = elf3_sim._load_configs(loader_request)
        if configs["env"].get("command_stage") != "S0":
            raise RuntimeError("signed-response environment is not in S0")
        raw_env = gym.make(elf3_sim.TASK_ID, cfg=env_cfg, render_mode=None)
        raw_env.unwrapped.set_evaluation_command(
            tuple(action["command"]), action["mode"]
        )
        env = elf3_sim._adapt_bool_dones(
            RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        )
        raw_env = None
        runner = HIMOnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=request.device
        )
        runner.load(str(checkpoint), load_optimizer=False, map_location=request.device)
        runner.eval_mode()
        policy = runner.get_inference_policy(device=request.device)
        observations = env.get_observations().to(request.device)
        arrays = _new_arrays()
        active = torch.ones(NUM_ENVS, dtype=torch.bool, device=env.device)
        expected_command = torch.tensor(
            action["command"], dtype=torch.float32, device=env.device
        ).unsqueeze(0).expand(NUM_ENVS, -1)

        with torch.inference_mode():
            for step in range(STEPS):
                _require_finite("signed-response observations", observations)
                snapshot = env.unwrapped.get_evaluation_observables()
                if not torch.equal(snapshot["command"], expected_command):
                    raise RuntimeError("signed-response command changed during rollout")
                arrays["command"][step] = _array_copy(snapshot["command"], "float32")
                arrays["mode"][step].fill(action["mode"])
                arrays["root_lin_vel_b"][step] = _array_copy(snapshot["root_lin_vel_b"], "float32")
                arrays["root_ang_vel_b"][step] = _array_copy(snapshot["root_ang_vel_b"], "float32")
                arrays["roll_pitch"][step] = _array_copy(snapshot["roll_pitch"], "float32")
                arrays["tracking_height"][step] = _array_copy(snapshot["tracking_height"], "float32")
                arrays["active_before"][step] = _array_copy(active, "bool")
                actions = policy(observations)
                if tuple(actions.shape) != (NUM_ENVS, constants.NUM_POLICY_ACTIONS):
                    raise RuntimeError("signed-response action shape is invalid")
                _require_finite("signed-response action", actions)
                arrays["action"][step] = _array_copy(actions, "float32")
                observations, rewards, dones, extras = env.step(actions.to(env.device))
                observations = observations.to(request.device)
                _require_finite("signed-response rewards", rewards)
                timeouts = extras.get("time_outs")
                if (
                    not isinstance(timeouts, torch.Tensor)
                    or timeouts.dtype != torch.bool
                    or tuple(timeouts.shape) != (NUM_ENVS,)
                ):
                    raise RuntimeError("signed-response timeouts are invalid")
                if dones.dtype != torch.bool or tuple(dones.shape) != (NUM_ENVS,):
                    raise RuntimeError("signed-response dones are invalid")
                arrays["reward"][step] = _array_copy(rewards, "float32")
                arrays["done"][step] = _array_copy(dones, "bool")
                arrays["timeout"][step] = _array_copy(timeouts, "bool")
                active &= ~(dones & ~timeouts)

        summary = validate_trajectory_arrays(arrays, action)
        npz_path = run_dir / "trajectory.npz"
        import numpy as np

        with npz_path.open("xb") as stream:
            np.savez(stream, **arrays)
        result = {
            "status": "PASS",
            "action": dict(action),
            "trajectory": {
                "path": "trajectory.npz",
                "sha256": sha256_file(npz_path),
                **summary,
            },
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        write_json_once(run_dir / "result.json", result)
        return result
    finally:
        failure: Exception | None = None
        if runner is not None and runner.writer is not None:
            try:
                runner.writer.flush()
                runner.writer.close()
            except Exception as exc:
                failure = exc
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                if failure is None:
                    failure = exc
        elif raw_env is not None:
            try:
                raw_env.close()
            except Exception as exc:
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure


def run(request: GridRequest) -> dict[str, Any]:
    """Run every canonical action exactly once and preserve raw evidence."""
    started = time.perf_counter()
    plan = _load_plan(request.plan)
    checkpoint, manifest, checkpoint_hash, manifest_hash = _require_source(plan)
    _verify_frozen_sources()
    request.output_root.mkdir(mode=0o755)
    plan_hash = hashlib.sha256(request.plan.read_bytes()).hexdigest()
    top_manifest = {
        "schema_version": 1,
        "kind": "elf3_m5_c3_signed_response_raw",
        "plan": {"path": str(request.plan), "sha256": plan_hash},
        "source": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "manifest_path": str(manifest),
            "manifest_sha256": manifest_hash,
        },
        "device": request.device,
        "actions": canonical_grid_actions(),
    }
    write_json_once(request.output_root / "manifest.json", top_manifest)
    completed: list[str] = []
    failure: Exception | None = None
    try:
        for action in canonical_grid_actions():
            _run_action(request, plan, action, checkpoint)
            completed.append(action["action_id"])
        _verify_frozen_sources()
        if sha256_file(checkpoint) != checkpoint_hash:
            raise RuntimeError("source checkpoint was mutated")
        if sha256_file(manifest) != manifest_hash:
            raise RuntimeError("source manifest was mutated")
        if hashlib.sha256(request.plan.read_bytes()).hexdigest() != plan_hash:
            raise RuntimeError("signed-response plan was mutated")
    except Exception as exc:
        failure = exc

    if failure is None:
        result = {
            "status": "PASS",
            "completed_actions": completed,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
    else:
        result = {
            "status": "FAIL",
            "failure_code": "C3_SIGNED_RESPONSE_RUNTIME_FAILURE",
            "error_type": type(failure).__name__,
            "error": str(failure),
            "completed_actions": completed,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
    write_json_once(request.output_root / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    request = parse_request(argv)
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=True, device=request.device).app
    exit_code = 1
    result: Mapping[str, Any] = {
        "status": "FAIL",
        "failure_code": "UNHANDLED_EXCEPTION",
    }
    try:
        result = run(request)
        exit_code = 0 if result.get("status") == "PASS" else 1
    except Exception:
        traceback.print_exc()
    finally:
        print(
            "C3_SIGNED_RESPONSE_PASS"
            if exit_code == 0
            else "C3_SIGNED_RESPONSE_FAIL",
            flush=True,
        )
        print(f"C3_SIGNED_RESPONSE_EXIT_CODE={exit_code}", flush=True)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        simulation_app.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
