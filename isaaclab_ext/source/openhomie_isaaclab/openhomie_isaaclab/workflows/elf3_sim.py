"""Post-launch ELF3 HIM train, play, and export workflows."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import gymnasium as gym
import isaaclab
import isaaclab_rl
import numpy as np
import onnx
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
from tensordict import TensorDict

from openhomie_isaaclab import elf3_constants as C
from openhomie_isaaclab.him_rl.exporter import (
    export_him_policy_onnx,
    export_him_policy_torchscript,
    verify_him_policy_onnx,
    verify_him_policy_torchscript,
)
from openhomie_isaaclab.him_rl.runner import HIMOnPolicyRunner
from openhomie_isaaclab.tasks.locomotion import elf3 as elf3_task
from openhomie_isaaclab.workflows.elf3_run import (
    MANIFEST_SCHEMA_VERSION,
    create_run_directory,
    final_iteration,
    sha256_file,
    validate_checkpoint_payload,
    validate_manifest,
    write_json_once,
)

TASK_ID = "OpenHomie-Elf3-Homie-Direct-v0"
AGENT_ENTRY_POINT = "openhomie_isaaclab.tasks.locomotion.elf3.agents.him_ppo_cfg:Elf3HIMRunnerCfg"
PLAY_STEPS = 100
MIN_FREE_GPU_MIB = 4096
TORCHSCRIPT_MAX_ERROR = 1e-7
ONNX_MAX_ERROR = 1e-5

M4_PATHS = (
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/__init__.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/actor_critic.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/estimator.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/exporter.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/ppo.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/runner.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/storage.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/symmetry.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/tasks/locomotion/elf3/agents/__init__.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/tasks/locomotion/elf3/agents/him_ppo_cfg.py",
)

EXPECTED_VERSIONS = {
    "isaaclab": "0.54.2",
    "isaaclab-rl": "0.4.7",
    "rsl-rl-lib": "3.1.2",
    "torch": "2.7.0+cu128",
    "onnx": "1.21.0",
    "onnxruntime": "1.28.0",
}

_FRESH_TORCHSCRIPT_PROBE = r"""
import json
import sys

import numpy as np
import torch

model = torch.jit.load(sys.argv[1], map_location="cpu")
model.eval()
values = np.load(sys.argv[2])
maximum = 0.0
input_shapes = []
output_shapes = []
with torch.inference_mode():
    for batch in (1, 4):
        history = torch.from_numpy(values[f"history_{batch}"])
        expected = values[f"expected_{batch}"]
        actual = model(history).detach().cpu().numpy()
        if (
            actual.shape != expected.shape
            or not torch.isfinite(history).all()
            or not np.isfinite(expected).all()
            or not np.isfinite(actual).all()
        ):
            raise RuntimeError("fresh TorchScript output is invalid")
        input_shapes.append(list(history.shape))
        output_shapes.append(list(actual.shape))
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
print("M5_PARITY_JSON=" + json.dumps({
    "input_shapes": input_shapes,
    "max_abs_error": maximum,
    "output_shapes": output_shapes,
}, sort_keys=True))
"""

_FRESH_ONNX_PROBE = r"""
import json
import sys

import numpy as np
import onnx
import onnxruntime as ort

model = onnx.load(sys.argv[1])
onnx.checker.check_model(model)
session = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
values = np.load(sys.argv[2])
input_name = session.get_inputs()[0].name
maximum = 0.0
input_shapes = []
output_shapes = []
for batch in (1, 4):
    history = values[f"history_{batch}"]
    expected = values[f"expected_{batch}"]
    actual = session.run(None, {input_name: history})[0]
    if (
        actual.shape != expected.shape
        or not np.isfinite(history).all()
        or not np.isfinite(expected).all()
        or not np.isfinite(actual).all()
    ):
        raise RuntimeError("fresh ONNX output is invalid")
    input_shapes.append(list(history.shape))
    output_shapes.append(list(actual.shape))
    maximum = max(maximum, float(np.max(np.abs(actual - expected))))
print("M5_PARITY_JSON=" + json.dumps({
    "checker_passed": True,
    "input_shapes": input_shapes,
    "max_abs_error": maximum,
    "output_shapes": output_shapes,
    "providers": session.get_providers(),
}, sort_keys=True))
"""


def _normalize_json(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize_json(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _normalize_json(value.value)
    if isinstance(value, np.generic):
        return _normalize_json(value.item())
    if isinstance(value, np.ndarray):
        return _normalize_json(value.tolist())
    if isinstance(value, torch.Tensor):
        return _normalize_json(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("configuration keys must be strings")
        return {key: _normalize_json(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(nested) for nested in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_json(nested) for nested in value]
        return sorted(normalized, key=lambda nested: json.dumps(nested, sort_keys=True))
    if isinstance(value, slice):
        return {
            "start": _normalize_json(value.start),
            "stop": _normalize_json(value.stop),
            "step": _normalize_json(value.step),
        }
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("configuration contains NaN or infinity")
        return value
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}:{value.__qualname__}"
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", getattr(value, "__name__", ""))
        if module and name:
            return f"{module}:{name}"
    raise TypeError(f"configuration value is not JSON-safe: {type(value).__name__}")


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_process(argv: Sequence[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"command failed ({completed.returncode}): {detail}")
    return completed


def _repository_root() -> Path:
    completed = _run_process(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=Path(__file__).resolve().parent,
    )
    return Path(completed.stdout.strip()).resolve(strict=True)


def _git_identity(repo_root: Path) -> dict[str, Any]:
    commit = _run_process(("git", "rev-parse", "HEAD"), cwd=repo_root).stdout.strip()
    status = _run_process(
        ("git", "status", "--porcelain=v1", "--untracked-files=normal"),
        cwd=repo_root,
    ).stdout
    dirty_paths: set[str] = set()
    for line in status.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            dirty_paths.update(path.split(" -> ", maxsplit=1))
        else:
            dirty_paths.add(path)
    return {"commit": commit, "dirty_paths": sorted(dirty_paths)}


def _dependency_versions() -> dict[str, str]:
    versions = {
        name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS
    }
    if versions != EXPECTED_VERSIONS:
        raise RuntimeError(f"dependency versions do not match: {versions}")
    return versions


def _runtime_identity() -> dict[str, Any]:
    isaaclab_path = Path(isaaclab.__file__).resolve(strict=True)
    app_path = isaaclab_path.parent / "app" / "__init__.py"
    isaaclab_rl_path = Path(isaaclab_rl.__file__).resolve(strict=True)
    if not app_path.is_file():
        raise FileNotFoundError(str(app_path))
    return {
        "python": platform.python_version(),
        "isaaclab_path": str(isaaclab_path),
        "isaaclab_app_path": str(app_path.resolve(strict=True)),
        "isaaclab_rl_path": str(isaaclab_rl_path),
        "versions": _dependency_versions(),
    }


def _gpu_identity(device: str) -> dict[str, Any]:
    parsed = torch.device(device)
    if parsed.type != "cuda":
        return {
            "name": "CPU",
            "driver_version": "unavailable",
            "cuda_version": str(torch.version.cuda or "unavailable"),
            "total_mib": 0,
            "free_mib": 0,
            "capability": [0, 0],
            "arch_list": list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else [],
            "cuda_probe_passed": False,
        }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    index = parsed.index if parsed.index is not None else torch.cuda.current_device()
    capability = [int(value) for value in torch.cuda.get_device_capability(index)]
    arch_list = [str(value) for value in torch.cuda.get_arch_list()]
    probe = float(torch.ones(32, device=device).square().sum().item())
    if not math.isfinite(probe) or probe != 32.0:
        raise RuntimeError("CUDA tensor probe failed")
    completed = _run_process(
        (
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        )
    )
    selected: tuple[str, str, int, int] | None = None
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5 and fields[0] == str(index):
            selected = (fields[1], fields[2], int(fields[3]), int(fields[4]))
            break
    if selected is None:
        raise RuntimeError(f"nvidia-smi did not report GPU index {index}")
    name, driver_version, total_mib, free_mib = selected
    if free_mib < MIN_FREE_GPU_MIB:
        raise RuntimeError(f"GPU has only {free_mib} MiB free")
    if capability < [12, 0] or "sm_120" not in arch_list:
        raise RuntimeError("CUDA runtime does not satisfy the sm_120 contract")
    return {
        "name": name,
        "driver_version": driver_version,
        "cuda_version": str(torch.version.cuda or ""),
        "total_mib": total_mib,
        "free_mib": free_mib,
        "capability": capability,
        "arch_list": arch_list,
        "cuda_probe_passed": True,
    }


def _source_identity(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (C.URDF_PATH, C.USD_PATH):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"ELF3 asset is not a regular file: {path}")
    assets = {
        "urdf_sha256": sha256_file(C.URDF_PATH),
        "usd_sha256": sha256_file(C.USD_PATH),
    }
    files = {
        path: sha256_file(repo_root / path)
        for path in M4_PATHS
    }
    return assets, {"files": files, "sha256": _json_sha256(files)}


def _seed_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_configs(request) -> tuple[Any, Any, dict[str, Any]]:
    if elf3_task.TASK_ID != TASK_ID:
        raise RuntimeError("ELF3 task identity mismatch")
    specification = gym.spec(TASK_ID)
    if specification.kwargs.get("rsl_rl_cfg_entry_point") != AGENT_ENTRY_POINT:
        raise RuntimeError("ELF3 agent entry point mismatch")
    env_cfg = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = request.num_envs
    env_cfg.seed = request.seed
    env_cfg.sim.device = request.device
    env_cfg.log_dir = str(request.run_dir)
    agent_cfg.seed = request.seed
    agent_cfg.device = request.device
    if request.command == "train":
        agent_cfg.max_iterations = request.iterations
    env_mapping = _normalize_json(env_cfg.to_dict())
    agent_mapping = _normalize_json(agent_cfg.to_dict())
    configs = {
        "env": env_mapping,
        "agent": agent_mapping,
        "sha256": {
            "env": _json_sha256(env_mapping),
            "agent": _json_sha256(agent_mapping),
        },
    }
    return env_cfg, agent_cfg, configs


def _require_finite(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise RuntimeError(f"{name} is non-finite")
        return
    if isinstance(value, Mapping) or hasattr(value, "items"):
        for key, nested in value.items():
            _require_finite(f"{name}.{key}", nested)
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _require_finite(f"{name}[{index}]", nested)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"{name} is non-finite")


def _load_checkpoint(path: Path, *, require_optimizers: bool) -> tuple[int, dict[str, Any]]:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    iteration = validate_checkpoint_payload(
        payload,
        require_optimizers=require_optimizers,
    )
    _require_finite("checkpoint", payload)
    return iteration, payload


def _checkpoint_identity(path: Path, iteration: int) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "iteration": iteration,
    }


def _parent_identity(checkpoint: Path, iteration: int) -> tuple[dict[str, Any], Path]:
    manifest_path = checkpoint.parent / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("resume checkpoint must have an adjacent parent manifest")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(payload)
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest_sha256": sha256_file(manifest_path),
        "iteration": iteration,
    }, manifest_path


def _adapt_bool_dones(env: RslRlVecEnvWrapper) -> RslRlVecEnvWrapper:
    original_step = env.step

    def step(actions: torch.Tensor):
        observations, rewards, dones, extras = original_step(actions)
        dones = dones.bool()
        return observations, rewards, dones, extras

    env.step = step
    return env


def _create_runner(request, env_cfg, agent_cfg):
    raw_env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
    try:
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    except Exception:
        raw_env.close()
        raise
    env = _adapt_bool_dones(env)
    log_dir = str(request.run_dir) if request.command == "train" else None
    cfg = agent_cfg.to_dict()
    try:
        runner = HIMOnPolicyRunner(
            env,
            cfg,
            log_dir=log_dir,
            device=request.device,
        )
    except Exception:
        env.close()
        raise
    return env, runner


def _training_result(request, runner: HIMOnPolicyRunner, start_iteration: int) -> dict[str, Any]:
    if request.iterations is None:
        raise ValueError("training iterations are required")
    if runner.current_learning_iteration != start_iteration:
        raise RuntimeError("runner start iteration does not match the checkpoint")
    expected_final = final_iteration(start_iteration, request.iterations)
    runner.learn(request.iterations, init_at_random_ep_len=True)
    if runner.current_learning_iteration != expected_final:
        raise RuntimeError("training iteration continuity failed")
    metrics = runner.get_training_metrics()
    _require_finite("training metrics", metrics)
    required_metrics = {
        "value_function",
        "surrogate",
        "entropy",
        "estimator_velocity",
        "estimator_swap",
        "actor_symmetry",
        "critic_symmetry",
    }
    if set(metrics) != required_metrics:
        raise RuntimeError("training metrics are incomplete")
    learning_rates = {
        "policy": float(runner.alg.learning_rate),
        "estimator": float(runner.alg.estimator_learning_rate),
    }
    _require_finite("learning rates", learning_rates)
    checkpoint_path = request.run_dir / f"model_{expected_final}.pt"
    checkpoint_iteration, checkpoint_payload = _load_checkpoint(
        checkpoint_path,
        require_optimizers=True,
    )
    if checkpoint_iteration != expected_final:
        raise RuntimeError("final checkpoint iteration is wrong")
    del checkpoint_payload
    return {
        "status": "PASS",
        "start_iteration": start_iteration,
        "final_iteration": expected_final,
        "metrics": _normalize_json(metrics),
        "learning_rates": learning_rates,
        "finite": {
            "observations": True,
            "actions": True,
            "rewards": True,
            "losses": True,
            "learning_rates": True,
            "entropy": True,
            "estimator_metrics": True,
            "checkpoint_values": True,
        },
        "checkpoint": _checkpoint_identity(checkpoint_path, expected_final),
    }


def _play_result(
    request,
    env: RslRlVecEnvWrapper,
    runner: HIMOnPolicyRunner,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    if request.checkpoint is None:
        raise ValueError("play requires a checkpoint")
    runner.load(
        str(request.checkpoint),
        load_optimizer=False,
        map_location=request.device,
    )
    runner.eval_mode()
    policy = runner.get_inference_policy(device=request.device)
    observations = env.get_observations().to(request.device)
    action_hash = hashlib.sha256()
    max_abs_diff = 0.0
    action_shape: list[int] | None = None
    with torch.inference_mode():
        for _ in range(PLAY_STEPS):
            _require_finite("play observations", observations)
            first = policy(observations)
            second = policy(observations)
            _require_finite("play actions", first)
            _require_finite("repeated play actions", second)
            expected_shape = (request.num_envs, C.NUM_POLICY_ACTIONS)
            if tuple(first.shape) != expected_shape or tuple(second.shape) != expected_shape:
                raise RuntimeError("play action shape is invalid")
            difference = float((first - second).abs().max().item())
            max_abs_diff = max(max_abs_diff, difference)
            if not torch.equal(first, second):
                raise RuntimeError("inference actions are not bitwise deterministic")
            action_shape = list(first.shape)
            action_hash.update(first.detach().cpu().contiguous().numpy().tobytes())
            observations, rewards, _, _ = env.step(first.to(env.device))
            observations = observations.to(request.device)
            _require_finite("play rewards", rewards)
    if action_shape is None:
        raise RuntimeError("play produced no actions")
    return {
        "status": "PASS",
        "play": {
            "steps": PLAY_STEPS,
            "action_shape": action_shape,
            "finite": True,
            "deterministic": True,
            "max_abs_diff": max_abs_diff,
            "action_sha256": action_hash.hexdigest(),
            "sha256_before": checkpoint_sha256,
        },
    }


def _fresh_probe(code: str, artifact: Path, samples: Path) -> dict[str, Any]:
    completed = _run_process(
        (sys.executable, "-c", code, str(artifact), str(samples))
    )
    prefix = "M5_PARITY_JSON="
    matches = [
        line[len(prefix) :]
        for line in completed.stdout.splitlines()
        if line.startswith(prefix)
    ]
    if len(matches) != 1:
        raise RuntimeError("fresh parity runtime did not emit one result")
    payload = json.loads(matches[0])
    _require_finite("fresh parity result", payload)
    return payload


def _export_result(request, runner: HIMOnPolicyRunner) -> dict[str, Any]:
    if request.checkpoint is None:
        raise ValueError("export requires a checkpoint")
    runner.load(
        str(request.checkpoint),
        load_optimizer=False,
        map_location=request.device,
    )
    runner.eval_mode()
    policy = runner.alg.policy.cpu()
    history_dim = policy.num_one_step_obs * policy.actor_history_length
    generator = torch.Generator(device="cpu").manual_seed(request.seed)
    histories = (
        torch.randn(1, history_dim, generator=generator),
        torch.randn(4, history_dim, generator=generator),
    )
    inference_policy = runner.get_inference_policy(device="cpu")
    live_actions: list[torch.Tensor] = []
    with torch.inference_mode():
        for history in histories:
            observation = TensorDict(
                {"policy": history},
                batch_size=[history.shape[0]],
            )
            actions = inference_policy(observation).detach().cpu()
            expected_shape = (history.shape[0], C.NUM_POLICY_ACTIONS)
            if tuple(actions.shape) != expected_shape:
                raise RuntimeError("live export-parity action shape is invalid")
            _require_finite("live export-parity actions", actions)
            live_actions.append(actions)
    expected = tuple(live_actions)

    torchscript_path = request.run_dir / "policy.ts"
    onnx_path = request.run_dir / "policy.onnx"
    torchscript_exporter = export_him_policy_torchscript(policy, torchscript_path)
    onnx_exporter = export_him_policy_onnx(policy, onnx_path)
    torchscript_exporter.cpu().eval()
    onnx_exporter.cpu().eval()
    exporter_errors = {"torchscript": 0.0, "onnx": 0.0}
    with torch.inference_mode():
        for history, expected_actions in zip(histories, expected, strict=True):
            torchscript_actions = torchscript_exporter(history)
            onnx_actions = onnx_exporter(history)
            for name, actions in (
                ("torchscript", torchscript_actions),
                ("onnx", onnx_actions),
            ):
                if actions.shape != expected_actions.shape:
                    raise RuntimeError(f"{name} exporter action shape is invalid")
                _require_finite(f"{name} exporter actions", actions)
                exporter_errors[name] = max(
                    exporter_errors[name],
                    float((actions - expected_actions).abs().max().item()),
                )
    torchscript_error = verify_him_policy_torchscript(
        torchscript_exporter,
        torchscript_path,
        histories,
        max_abs_error=TORCHSCRIPT_MAX_ERROR,
    )
    onnx_error = verify_him_policy_onnx(
        onnx_exporter,
        onnx_path,
        histories,
        max_abs_error=ONNX_MAX_ERROR,
    )
    onnx.checker.check_model(onnx.load(onnx_path))
    samples_path = request.run_dir / "parity_samples.npz"
    with samples_path.open("xb") as stream:
        np.savez(
            stream,
            history_1=histories[0].numpy(),
            expected_1=expected[0].numpy(),
            history_4=histories[1].numpy(),
            expected_4=expected[1].numpy(),
        )
    fresh_torchscript = _fresh_probe(
        _FRESH_TORCHSCRIPT_PROBE,
        torchscript_path,
        samples_path,
    )
    fresh_onnx = _fresh_probe(_FRESH_ONNX_PROBE, onnx_path, samples_path)
    input_shapes = [list(history.shape) for history in histories]
    output_shapes = [list(actions.shape) for actions in expected]
    for name, probe in (
        ("TorchScript", fresh_torchscript),
        ("ONNX", fresh_onnx),
    ):
        if (
            probe.get("input_shapes") != input_shapes
            or probe.get("output_shapes") != output_shapes
        ):
            raise RuntimeError(f"fresh {name} parity shapes are invalid")
    torchscript_error = max(
        exporter_errors["torchscript"],
        float(torchscript_error),
        float(fresh_torchscript["max_abs_error"]),
    )
    onnx_error = max(
        exporter_errors["onnx"],
        float(onnx_error),
        float(fresh_onnx["max_abs_error"]),
    )
    if torchscript_error > TORCHSCRIPT_MAX_ERROR:
        raise RuntimeError("TorchScript parity exceeds its threshold")
    if onnx_error > ONNX_MAX_ERROR:
        raise RuntimeError("ONNX parity exceeds its threshold")
    providers = fresh_onnx.get("providers")
    if providers != ["CPUExecutionProvider"]:
        raise RuntimeError("ONNX fresh runtime used an unexpected provider")
    if fresh_onnx.get("checker_passed") is not True:
        raise RuntimeError("ONNX checker did not pass in the fresh runtime")
    return {
        "status": "PASS",
        "exports": {
            "oracle": {
                "device": "cpu",
                "method": "runner.get_inference_policy",
            },
            "torchscript": {
                "path": str(torchscript_path.resolve(strict=True)),
                "sha256": sha256_file(torchscript_path),
                "provider": "torch.jit.load",
                "fresh_runtime": True,
                "batches": [1, 4],
                "input_shapes": input_shapes,
                "output_shapes": output_shapes,
                "max_abs_error": torchscript_error,
            },
            "onnx": {
                "path": str(onnx_path.resolve(strict=True)),
                "sha256": sha256_file(onnx_path),
                "providers": providers,
                "checker_passed": True,
                "fresh_runtime": True,
                "batches": [1, 4],
                "input_shapes": input_shapes,
                "output_shapes": output_shapes,
                "max_abs_error": onnx_error,
            },
        },
    }


def _manifest(
    request,
    *,
    start_iteration: int,
    configs: dict[str, Any],
    git: dict[str, Any],
    assets: dict[str, Any],
    m4_sources: dict[str, Any],
    runtime: dict[str, Any],
    gpu: dict[str, Any],
    checkpoint: dict[str, Any] | None,
    parent: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "command": request.command,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "task_id": TASK_ID,
        "seed": request.seed,
        "device": request.device,
        "num_envs": request.num_envs,
        "iterations": request.iterations,
        "cli": request.to_dict(),
        "git": git,
        "configs": configs,
        "assets": assets,
        "m4_sources": m4_sources,
        "runtime": runtime,
        "gpu": gpu,
    }
    if request.command == "train":
        payload["start_iteration"] = start_iteration
    if checkpoint is not None:
        payload["checkpoint"] = checkpoint
    if parent is not None:
        payload["parent"] = parent
    validate_manifest(payload)
    return payload


def _record_cleanup_error(current: Exception | None, cleanup: Exception) -> Exception:
    if current is None:
        return cleanup
    return RuntimeError(f"{current}; cleanup also failed: {cleanup}")


def run(request) -> dict[str, Any]:
    """Execute one validated post-launch request and return its JSON result."""
    if request.command not in {"train", "play", "export"}:
        raise ValueError(f"unsupported command: {request.command}")
    started = time.perf_counter()
    repo_root = _repository_root()
    git = _git_identity(repo_root)
    _seed_runtime(request.seed)
    env_cfg, agent_cfg, configs = _load_configs(request)
    assets, m4_sources = _source_identity(repo_root)
    runtime = _runtime_identity()
    gpu = _gpu_identity(request.device)

    start_iteration = 0
    source_checkpoint: dict[str, Any] | None = None
    parent: dict[str, Any] | None = None
    parent_manifest_path: Path | None = None
    if request.checkpoint is not None:
        start_iteration, checkpoint_payload = _load_checkpoint(
            request.checkpoint,
            require_optimizers=request.command == "train" and request.resume,
        )
        del checkpoint_payload
        source_checkpoint = _checkpoint_identity(request.checkpoint, start_iteration)
        if request.command == "train" and request.resume:
            parent, parent_manifest_path = _parent_identity(
                request.checkpoint,
                start_iteration,
            )
    elif request.command != "train" or request.resume:
        raise ValueError("this command requires an explicit checkpoint")

    run_dir = create_run_directory(request.run_dir)
    if run_dir != request.run_dir:
        raise RuntimeError("run directory identity changed")
    manifest = _manifest(
        request,
        start_iteration=start_iteration,
        configs=configs,
        git=git,
        assets=assets,
        m4_sources=m4_sources,
        runtime=runtime,
        gpu=gpu,
        checkpoint=source_checkpoint if request.command in {"play", "export"} else None,
        parent=parent,
    )
    write_json_once(run_dir / "manifest.json", manifest)

    env: RslRlVecEnvWrapper | None = None
    runner: HIMOnPolicyRunner | None = None
    result: dict[str, Any] | None = None
    failure: Exception | None = None
    try:
        env, runner = _create_runner(request, env_cfg, agent_cfg)
        if request.command == "train":
            if request.resume:
                assert request.checkpoint is not None
                runner.load(
                    str(request.checkpoint),
                    load_optimizer=True,
                    map_location=request.device,
                )
            elif runner.current_learning_iteration != 0:
                raise RuntimeError("fresh training did not start at iteration zero")
            result = _training_result(request, runner, start_iteration)
        elif request.command == "play":
            assert source_checkpoint is not None
            result = _play_result(
                request,
                env,
                runner,
                source_checkpoint["sha256"],
            )
        elif request.command == "export":
            result = _export_result(request, runner)
        else:
            raise ValueError(f"unsupported command: {request.command}")
    except Exception as exc:
        failure = exc
    finally:
        if runner is not None and runner.writer is not None:
            try:
                runner.writer.flush()
                runner.writer.close()
            except Exception as exc:
                failure = _record_cleanup_error(failure, exc)
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                failure = _record_cleanup_error(failure, exc)

    if source_checkpoint is not None:
        try:
            current_hash = sha256_file(request.checkpoint)
            if current_hash != source_checkpoint["sha256"]:
                raise RuntimeError("source checkpoint was mutated")
            if result is not None and request.command == "play":
                result["play"]["sha256_after"] = current_hash
        except Exception as exc:
            failure = _record_cleanup_error(failure, exc)
    if parent_manifest_path is not None and parent is not None:
        try:
            if sha256_file(parent_manifest_path) != parent["manifest_sha256"]:
                raise RuntimeError("parent manifest was mutated")
        except Exception as exc:
            failure = _record_cleanup_error(failure, exc)

    if failure is not None:
        result = {
            "status": "FAIL",
            "failure_code": "M5_RUNTIME_FAILURE",
            "error_type": type(failure).__name__,
            "error": str(failure),
        }
    if result is None:
        raise RuntimeError("workflow produced no result")
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    write_json_once(run_dir / "result.json", result)
    return result
