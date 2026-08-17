#!/usr/bin/env python3
"""Strict offline verifier for ELF3 M5 convergence evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT / "isaaclab_ext/source/openhomie_isaaclab"
PACKAGE_ROOT = PACKAGE_PARENT / "openhomie_isaaclab"
URDF_PATH = PACKAGE_ROOT / "assets/elf3/elf3.urdf"
USD_PATH = PACKAGE_ROOT / "assets/elf3/elf3.usd"
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from openhomie_isaaclab.workflows.elf3_run import (  # noqa: E402
    CANONICAL_ITERATIONS,
    CANONICAL_NUM_ENVS,
    CANONICAL_STEPS_PER_ENV,
    EVALUATION_NUM_ENVS,
    EVALUATION_SCENARIOS,
    EVALUATION_STEPS,
    canonical_iterations,
    evaluate_behavior,
    evaluate_convergence,
    read_tensorboard_scalars,
    sha256_file,
    validate_manifest,
)


TRAIN_DIRECTORY = "canonical_train"
EXPORT_DIRECTORY = "exact_export"
SCENARIOS_DIRECTORY = "scenarios"
AGGREGATE_FILENAME = "aggregate_result.json"
SEEDS = (42, 43, 44)
SCENARIOS = ("stand", "forward", "turn", "crouch")
MIN_FREE_GPU_MIB = 12 * 1024
TS_MAX_ERROR = 1e-7
ONNX_MAX_ERROR = 1e-5
CANONICAL_TRANSITIONS = (
    CANONICAL_NUM_ENVS * CANONICAL_ITERATIONS * CANONICAL_STEPS_PER_ENV
)
EXPECTED_VERSIONS = {
    "isaaclab": "0.54.2",
    "isaaclab-rl": "0.4.7",
    "rsl-rl-lib": "3.1.2",
    "torch": "2.7.0+cu128",
    "onnx": "1.21.0",
    "onnxruntime": "1.28.0",
}
REQUIRED_TAGS = (
    "Train/mean_episode_length",
    "Train/mean_reward",
    "Loss/value_function",
    "Loss/surrogate",
    "Loss/entropy",
    "Loss/estimator_velocity",
    "Loss/estimator_swap",
    "Loss/actor_symmetry",
    "Loss/critic_symmetry",
    "Loss/learning_rate",
    "Perf/total_fps",
    "Episode_Termination/time_out",
)
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
FROZEN_C1_SOURCES = {
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/runner.py": (
        "daed23208a91a71efff1ffe7ecca7ea40623be0432f9bea74bc106b4bb31fbf4"
    ),
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/tasks/locomotion/elf3/elf3_homie_env.py": (
        "d7f54abb9b424e95d043df70ca350f32a61a43a7075ecc8859f2c87c7ed43342"
    ),
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/elf3_run.py": (
        "bde3dbe060b403befc752bbd3c450b1c3959bd48b3270be7af7b12794ef4dd28"
    ),
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/elf3_sim.py": (
        "c597d33c80580025d7fe28136262d75fa74301b9153dc548f2b2a71d2539dcf8"
    ),
}
FINITE_TRAINING_KEYS = frozenset(
    {
        "observations",
        "actions",
        "rewards",
        "losses",
        "learning_rates",
        "entropy",
        "estimator_metrics",
        "checkpoint_values",
    }
)
METRIC_KEYS = {
    "stand": {"height_mae", "tilt_rms"},
    "forward": {"velocity_mae", "height_mae"},
    "turn": {"yaw_rate_mae", "height_mae"},
    "crouch": {"height_mae", "planar_speed_rms"},
}
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _walk_finite(value: Any, name: str = "JSON") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} contains a non-string key")
            _walk_finite(nested, f"{name}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _walk_finite(nested, f"{name}[{index}]")


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _root_directory(value: str | os.PathLike[str]) -> Path:
    root = Path(value)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("evidence root must be an absolute non-symlink directory")
    resolved = root.resolve(strict=True)
    if resolved != root:
        raise ValueError("evidence root must be canonical")
    return root


def _child_directory(root: Path, relative: str) -> Path:
    directory = root / relative
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"missing non-symlink directory: {relative}")
    if directory.resolve(strict=True).parent != (root / relative).parent.resolve():
        raise ValueError(f"directory escapes evidence root: {relative}")
    return directory


def _relative_file(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or "\\" in value:
        raise TypeError(f"{name} path must be a POSIX string")
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"{name} path must be canonical and relative")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} traverses a symlink")
    if not candidate.is_file():
        raise ValueError(f"missing {name}: {value}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} escapes evidence root")
    return resolved


def _absolute_exact(value: Any, expected: Path, name: str) -> None:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    candidate = Path(value)
    if candidate.is_symlink() or candidate.resolve(strict=True) != expected:
        raise ValueError(f"{name} does not identify the canonical artifact")


def _load_json(path: Path, name: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} is not a regular file")

    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    _walk_finite(payload, name)
    return payload


def _file_reference(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
    }


def _check_reference(
    root: Path, reference: Any, expected: Path, name: str
) -> None:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"{name} reference must contain only path and sha256")
    path = _relative_file(root, reference["path"], name)
    if path != expected:
        raise ValueError(f"{name} reference points to the wrong artifact")
    digest = _hash(reference["sha256"], f"{name} reference hash")
    if sha256_file(path) != digest:
        raise ValueError(f"{name} reference hash does not match")


def _check_source_freeze() -> None:
    for relative, expected in FROZEN_C1_SOURCES.items():
        source = REPO_ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"missing frozen C1 source: {relative}")
        if sha256_file(source) != expected:
            raise ValueError(f"frozen C1 source changed: {relative}")


def _check_identity(payload: Any, *, minimum_gpu_mib: int) -> Mapping[str, Any]:
    validate_manifest(payload)
    if not isinstance(payload, Mapping):
        raise TypeError("manifest must be a mapping")
    git = payload["git"]
    if re.fullmatch(r"[0-9a-f]{40}", str(git.get("commit", ""))) is None:
        raise ValueError("manifest Git commit is invalid")

    configs = payload["configs"]
    hashes = configs.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("manifest config hashes are missing")
    for name in ("env", "agent"):
        if not isinstance(configs.get(name), Mapping):
            raise ValueError(f"manifest {name} config is missing")
        if hashes.get(name) != _json_sha256(configs[name]):
            raise ValueError(f"manifest {name} config hash does not match")

    assets = payload["assets"]
    expected_assets = {
        "urdf_sha256": sha256_file(URDF_PATH),
        "usd_sha256": sha256_file(USD_PATH),
    }
    if assets != expected_assets:
        raise ValueError("manifest asset hashes do not match the accepted assets")

    expected_m4 = {path: sha256_file(REPO_ROOT / path) for path in M4_PATHS}
    m4 = payload["m4_sources"]
    if m4.get("files") != expected_m4:
        raise ValueError("manifest M4 source hashes do not match the frozen stack")
    if m4.get("sha256") != _json_sha256(expected_m4):
        raise ValueError("manifest aggregate M4 hash does not match")

    runtime = payload["runtime"]
    if runtime.get("versions") != EXPECTED_VERSIONS:
        raise ValueError("manifest dependency versions do not match")
    if not isinstance(runtime.get("python"), str) or not runtime["python"]:
        raise ValueError("manifest Python version is missing")
    for name in ("isaaclab_path", "isaaclab_app_path", "isaaclab_rl_path"):
        value = runtime.get(name)
        if (
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or "IsaacLab-v2.3.2" not in Path(value).parts
        ):
            raise ValueError(f"manifest {name} is outside IsaacLab-v2.3.2")

    gpu = payload["gpu"]
    for name in ("name", "driver_version", "cuda_version"):
        if not isinstance(gpu.get(name), str) or not gpu[name]:
            raise ValueError(f"manifest GPU {name} is missing")
    total = _number(gpu.get("total_mib"), "gpu.total_mib")
    free = _number(gpu.get("free_mib"), "gpu.free_mib")
    if total <= 0 or free < minimum_gpu_mib or free > total:
        raise ValueError("manifest GPU memory evidence is insufficient")
    capability = gpu.get("capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in capability)
        or tuple(capability) < (12, 0)
    ):
        raise ValueError("manifest GPU capability is below sm_120")
    if not isinstance(gpu.get("arch_list"), list) or "sm_120" not in gpu["arch_list"]:
        raise ValueError("manifest Torch architecture lacks sm_120")
    if gpu.get("cuda_probe_passed") is not True:
        raise ValueError("manifest CUDA tensor probe did not pass")
    return payload


def _check_manifest(
    payload: Any,
    *,
    command: str,
    run_dir: Path,
    seed: int,
    num_envs: int,
    iterations: int | None,
    minimum_gpu_mib: int,
) -> Mapping[str, Any]:
    manifest = _check_identity(payload, minimum_gpu_mib=minimum_gpu_mib)
    expected = {
        "schema_version": 1,
        "command": command,
        "seed": seed,
        "device": "cuda:0",
        "num_envs": num_envs,
        "iterations": iterations,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"manifest {key} does not match the C2 contract")
    cli = manifest.get("cli")
    if not isinstance(cli, Mapping):
        raise ValueError("manifest CLI evidence is missing")
    for key, value in expected.items():
        if key == "schema_version":
            continue
        if cli.get(key) != value:
            raise ValueError(f"manifest CLI {key} does not match")
    if cli.get("headless") is not True or cli.get("resume") is not False:
        raise ValueError("manifest CLI must be fresh and headless")
    _absolute_exact(cli.get("run_dir"), run_dir, "manifest CLI run directory")
    return manifest


def _check_result(payload: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("status") != "PASS":
        raise ValueError(f"{name} result is not PASS")
    return payload


def _check_training(
    root: Path, train: Path
) -> tuple[Path, str, int, int, dict[str, list[float]], Path, dict[int, Path], str]:
    files = list(train.iterdir())
    events = [path for path in files if path.name.startswith("events.out.tfevents.")]
    if len(events) != 1 or events[0].is_symlink() or not events[0].is_file():
        raise ValueError("canonical training requires exactly one regular event file")
    event = events[0].resolve(strict=True)
    manifest_path = train / "manifest.json"
    result_path = train / "result.json"
    manifest_raw = _load_json(manifest_path, "training manifest")
    requested_envs = manifest_raw.get("num_envs") if isinstance(manifest_raw, Mapping) else None
    num_envs = _integer(requested_envs, "training num_envs", minimum=1)
    iterations = canonical_iterations(num_envs, CANONICAL_STEPS_PER_ENV)
    manifest = _check_manifest(
        manifest_raw,
        command="train",
        run_dir=train,
        seed=42,
        num_envs=num_envs,
        iterations=iterations,
        minimum_gpu_mib=MIN_FREE_GPU_MIB,
    )
    if num_envs > CANONICAL_NUM_ENVS:
        raise ValueError("training num_envs cannot exceed the canonical 4096")
    if manifest.get("start_iteration") != 0:
        raise ValueError("canonical training must start at iteration zero")
    agent = manifest["configs"]["agent"]
    if agent.get("num_steps_per_env") != CANONICAL_STEPS_PER_ENV:
        raise ValueError("num_steps_per_env must remain frozen at 50")
    if agent.get("max_iterations") != iterations:
        raise ValueError("agent max_iterations must equal requested iterations")
    save_interval = agent.get("save_interval")
    if save_interval != 200:
        raise ValueError("checkpoint save_interval must remain frozen at 200")

    result = _check_result(_load_json(result_path, "training result"), "training")
    if result.get("start_iteration") != 0 or result.get("final_iteration") != iterations:
        raise ValueError("canonical training iteration lineage is incomplete")
    finite = result.get("finite")
    if not isinstance(finite, Mapping) or set(finite) != FINITE_TRAINING_KEYS:
        raise ValueError("training finite evidence is incomplete")
    if any(finite[name] is not True for name in FINITE_TRAINING_KEYS):
        raise ValueError("training contains non-finite runtime evidence")
    checkpoint_info = result.get("checkpoint")
    if not isinstance(checkpoint_info, Mapping):
        raise ValueError("training checkpoint identity is missing")
    if checkpoint_info.get("iteration") != iterations:
        raise ValueError("training checkpoint iteration does not match")
    checkpoint_iterations = list(range(save_interval, iterations + 1, save_interval))
    if not checkpoint_iterations or checkpoint_iterations[-1] != iterations:
        checkpoint_iterations.append(iterations)
    checkpoints = {
        iteration: train / f"model_{iteration}.pt"
        for iteration in checkpoint_iterations
    }
    for iteration, path in checkpoints.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"checkpoint chain is missing iteration {iteration}")
    checkpoint = checkpoints[iterations]
    _absolute_exact(checkpoint_info.get("path"), checkpoint, "training checkpoint")
    checkpoint_hash = _hash(checkpoint_info.get("sha256"), "training checkpoint hash")
    if sha256_file(checkpoint) != checkpoint_hash:
        raise ValueError("training checkpoint hash does not match")

    expected_names = {
        "manifest.json",
        "result.json",
        event.name,
        *(path.name for path in checkpoints.values()),
    }
    if {path.name for path in files} != expected_names:
        raise ValueError("canonical training directory contains unexpected artifacts")

    values = read_tensorboard_scalars(train, REQUIRED_TAGS, iterations)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(os.fspath(train))
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", ()))
    if not set(REQUIRED_TAGS).issubset(available):
        raise ValueError("TensorBoard required scalar tag inventory is incomplete")
    expected_steps = list(range(1, iterations + 1))
    for tag in REQUIRED_TAGS:
        points = accumulator.Scalars(tag)
        if len(points) != iterations:
            raise ValueError(f"TensorBoard scalar {tag} point count is not exact")
        if [point.step for point in points] != expected_steps:
            raise ValueError(f"TensorBoard scalar {tag} steps are not exactly 1..N")
        if len(values[tag]) != iterations:
            raise ValueError(f"TensorBoard scalar {tag} is truncated")

    timeout_values = values["Episode_Termination/time_out"]
    if any(value < 0.0 or value > 1.0 for value in timeout_values):
        raise ValueError("timeout transition fractions must remain in [0, 1]")
    transitions = num_envs * iterations * CANONICAL_STEPS_PER_ENV
    if transitions < CANONICAL_TRANSITIONS:
        raise ValueError("canonical transition budget is incomplete")
    convergence_payload = {
        "mean_episode_length": values["Train/mean_episode_length"],
        "timeouts": values["Episode_Termination/time_out"],
        "scalars": {
            "reward": values["Train/mean_reward"],
            "policy_loss": values["Loss/surrogate"],
            "value_loss": values["Loss/value_function"],
            "estimator_loss": values["Loss/estimator_velocity"],
            "learning_rate": values["Loss/learning_rate"],
            "entropy": values["Loss/entropy"],
            "throughput": values["Perf/total_fps"],
        },
        "actual_transitions": transitions,
        "expected_transitions": transitions,
    }
    convergence = evaluate_convergence(convergence_payload)
    if convergence != {"passed": True, "reasons": []}:
        raise ValueError(f"convergence gate failed: {convergence['reasons']}")
    return (
        checkpoint,
        checkpoint_hash,
        iterations,
        transitions,
        values,
        event,
        checkpoints,
        str(manifest["git"]["commit"]),
    )


def _check_source_checkpoint(
    manifest: Mapping[str, Any],
    checkpoint: Path,
    checkpoint_hash: str,
    checkpoint_iteration: int,
    name: str,
) -> None:
    source = manifest.get("checkpoint")
    if not isinstance(source, Mapping):
        raise ValueError(f"{name} source checkpoint identity is missing")
    _absolute_exact(source.get("path"), checkpoint, f"{name} source checkpoint")
    if source.get("sha256") != checkpoint_hash:
        raise ValueError(f"{name} uses a foreign checkpoint hash")
    if source.get("iteration") != checkpoint_iteration:
        raise ValueError(f"{name} checkpoint iteration is wrong")


def _check_export(
    root: Path,
    export: Path,
    checkpoint: Path,
    checkpoint_hash: str,
    checkpoint_iteration: int,
    expected_commit: str,
) -> tuple[float, float, Path, Path, Path]:
    expected_files = {"manifest.json", "result.json", "policy.ts", "policy.onnx", "parity_samples.npz"}
    if {path.name for path in export.iterdir()} != expected_files:
        raise ValueError("exact export directory contains unexpected artifacts")
    manifest = _check_manifest(
        _load_json(export / "manifest.json", "export manifest"),
        command="export",
        run_dir=export,
        seed=42,
        num_envs=1,
        iterations=None,
        minimum_gpu_mib=4096,
    )
    if manifest["git"].get("commit") != expected_commit:
        raise ValueError("export manifest belongs to a foreign Git commit")
    _check_source_checkpoint(
        manifest, checkpoint, checkpoint_hash, checkpoint_iteration, "export"
    )
    _absolute_exact(manifest["cli"].get("checkpoint"), checkpoint, "export CLI checkpoint")
    if manifest["cli"].get("scenario") is not None or manifest["cli"].get("steps") is not None:
        raise ValueError("export cannot contain scenario CLI fields")
    result = _check_result(_load_json(export / "result.json", "export result"), "export")
    exports = result.get("exports")
    if not isinstance(exports, Mapping) or set(exports) != {"oracle", "torchscript", "onnx"}:
        raise ValueError("export evidence inventory is incomplete")
    if exports["oracle"] != {"device": "cpu", "method": "runner.get_inference_policy"}:
        raise ValueError("export oracle is not the live CPU inference policy")
    expected_input_shapes = [[1, 468], [4, 468]]
    expected_output_shapes = [[1, 12], [4, 12]]
    results: dict[str, float] = {}
    paths: dict[str, Path] = {}
    for kind, filename, limit in (
        ("torchscript", "policy.ts", TS_MAX_ERROR),
        ("onnx", "policy.onnx", ONNX_MAX_ERROR),
    ):
        evidence = exports.get(kind)
        if not isinstance(evidence, Mapping):
            raise ValueError(f"missing {kind} evidence")
        artifact = export / filename
        _absolute_exact(evidence.get("path"), artifact, f"{kind} artifact")
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"{kind} artifact is not a regular file")
        if sha256_file(artifact) != _hash(evidence.get("sha256"), f"{kind} hash"):
            raise ValueError(f"{kind} artifact hash does not match")
        if evidence.get("fresh_runtime") is not True or evidence.get("batches") != [1, 4]:
            raise ValueError(f"{kind} did not run both batches in a fresh runtime")
        if evidence.get("input_shapes") != expected_input_shapes:
            raise ValueError(f"{kind} input shapes are wrong")
        if evidence.get("output_shapes") != expected_output_shapes:
            raise ValueError(f"{kind} output shapes are wrong")
        error = _number(evidence.get("max_abs_error"), f"{kind}.max_abs_error")
        if error > limit:
            raise ValueError(f"{kind} parity exceeds its threshold")
        results[kind] = error
        paths[kind] = artifact.resolve(strict=True)
    if exports["torchscript"].get("provider") != "torch.jit.load":
        raise ValueError("TorchScript must use torch.jit.load")
    if exports["onnx"].get("checker_passed") is not True:
        raise ValueError("ONNX checker did not pass")
    if exports["onnx"].get("providers") != ["CPUExecutionProvider"]:
        raise ValueError("ONNX must use exactly CPUExecutionProvider")
    if sha256_file(checkpoint) != checkpoint_hash:
        raise ValueError("export mutated the canonical checkpoint")
    parity_samples = export / "parity_samples.npz"
    if parity_samples.is_symlink() or not parity_samples.is_file():
        raise ValueError("export parity samples are not a regular file")

    return results["torchscript"], results["onnx"], paths["torchscript"], paths["onnx"], parity_samples.resolve(strict=True)


def _check_scenario(
    run: Path,
    scenario: str,
    seed: int,
    checkpoint: Path,
    checkpoint_hash: str,
    checkpoint_iteration: int,
    expected_commit: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if {path.name for path in run.iterdir()} != {"manifest.json", "result.json"}:
        raise ValueError(f"scenario run {run.name} contains unexpected artifacts")
    manifest = _check_manifest(
        _load_json(run / "manifest.json", f"{run.name} manifest"),
        command="play",
        run_dir=run,
        seed=seed,
        num_envs=EVALUATION_NUM_ENVS,
        iterations=None,
        minimum_gpu_mib=4096,
    )
    if manifest["git"].get("commit") != expected_commit:
        raise ValueError(f"{run.name} belongs to a foreign Git commit")
    _check_source_checkpoint(
        manifest, checkpoint, checkpoint_hash, checkpoint_iteration, run.name
    )
    cli = manifest["cli"]
    _absolute_exact(cli.get("checkpoint"), checkpoint, f"{run.name} CLI checkpoint")
    if cli.get("scenario") != scenario or cli.get("steps") != EVALUATION_STEPS:
        raise ValueError(f"{run.name} scenario CLI does not match")
    result = _check_result(_load_json(run / "result.json", f"{run.name} result"), run.name)
    play = result.get("play")
    if not isinstance(play, Mapping):
        raise ValueError(f"{run.name} play evidence is missing")
    expected = {
        "scenario": scenario,
        "mode": EVALUATION_SCENARIOS[scenario].mode,
        "steps": EVALUATION_STEPS,
        "num_envs": EVALUATION_NUM_ENVS,
        "seed": seed,
        "finite": True,
    }
    for key, value in expected.items():
        if play.get(key) != value:
            raise ValueError(f"{run.name} {key} does not match")
    _absolute_exact(play.get("checkpoint_path"), checkpoint, f"{run.name} checkpoint")
    command = play.get("command")
    if not isinstance(command, list) or len(command) != 4:
        raise ValueError(f"{run.name} command must have four values")
    command_values = [_number(value, f"{run.name} command") for value in command]
    expected_command = EVALUATION_SCENARIOS[scenario].command
    for index in range(3):
        if abs(command_values[index] - expected_command[index]) > 1e-6:
            raise ValueError(f"{run.name} command component {index} is wrong")
    if scenario == "crouch" and abs(command_values[3] - 0.80) > 1e-6:
        raise ValueError("crouch command height must be 0.80")
    if scenario != "crouch" and command_values[3] <= 0:
        raise ValueError(f"{run.name} default height must be positive")

    credited = _integer(play.get("credited_env_steps"), f"{run.name} credited steps")
    maximum = EVALUATION_NUM_ENVS * EVALUATION_STEPS
    if credited > maximum:
        raise ValueError(f"{run.name} credited steps exceed the rollout")
    termination_steps = play.get("non_timeout_termination_steps")
    termination_reasons = play.get("non_timeout_termination_reasons")
    if not isinstance(termination_steps, list) or len(termination_steps) != EVALUATION_NUM_ENVS:
        raise ValueError(f"{run.name} termination steps are incomplete")
    if not isinstance(termination_reasons, list) or len(termination_reasons) != EVALUATION_NUM_ENVS:
        raise ValueError(f"{run.name} termination reasons are incomplete")
    reconstructed = 0
    for index, (step, reason) in enumerate(zip(termination_steps, termination_reasons)):
        if step is None:
            if reason is not None:
                raise ValueError(f"{run.name} survivor {index} has a termination reason")
            reconstructed += EVALUATION_STEPS
        else:
            value = _integer(step, f"{run.name} termination step", minimum=1)
            if value > EVALUATION_STEPS or not isinstance(reason, str) or not reason:
                raise ValueError(f"{run.name} termination record {index} is invalid")
            reconstructed += value
    if reconstructed != credited:
        raise ValueError(f"{run.name} credited steps do not match terminations")
    timeout_count = _integer(play.get("timeout_count"), f"{run.name} timeout count")
    if timeout_count > maximum:
        raise ValueError(f"{run.name} timeout count exceeds the rollout")
    survival = _number(play.get("survival"), f"{run.name} survival")
    if survival != credited / maximum:
        raise ValueError(f"{run.name} survival does not match credited steps")
    metrics = play.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != METRIC_KEYS[scenario]:
        raise ValueError(f"{run.name} metric inventory is wrong")
    row: dict[str, Any] = {
        "scenario": scenario,
        "seed": seed,
        "finite": True,
        "survival": survival,
    }
    for name, value in metrics.items():
        row[name] = _number(value, f"{run.name}.{name}")
    action_hash = _hash(play.get("action_sha256"), f"{run.name} action hash")
    trajectory_hash = _hash(
        play.get("trajectory_sha256"), f"{run.name} trajectory hash"
    )
    for name in ("sha256_before", "sha256_after"):
        if play.get(name) != checkpoint_hash:
            raise ValueError(f"{run.name} checkpoint hash changed")
    if sha256_file(checkpoint) != checkpoint_hash:
        raise ValueError(f"{run.name} mutated the canonical checkpoint")
    return row, {
        "manifest": _file_reference(run.parents[1], run / "manifest.json"),
        "result": _file_reference(run.parents[1], run / "result.json"),
        "action_sha256": action_hash,
        "trajectory_sha256": trajectory_hash,
    }


def _scenario_means(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    means: dict[str, dict[str, float]] = {}
    for scenario in SCENARIOS:
        selected = [row for row in rows if row["scenario"] == scenario]
        means[scenario] = {
            metric: sum(float(row[metric]) for row in selected) / len(selected)
            for metric in sorted(METRIC_KEYS[scenario])
        }
    return means


def _evaluate_evidence(
    evidence_root: str | os.PathLike[str],
    *,
    require_aggregate: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify one immutable canonical train/export/scenario evidence root."""
    root = _root_directory(evidence_root)
    _check_source_freeze()
    expected_top = {
        TRAIN_DIRECTORY,
        EXPORT_DIRECTORY,
        SCENARIOS_DIRECTORY,
    }
    if require_aggregate:
        expected_top.add(AGGREGATE_FILENAME)
    if {path.name for path in root.iterdir()} != expected_top:
        raise ValueError("evidence root layout is not exact")
    train = _child_directory(root, TRAIN_DIRECTORY)
    export = _child_directory(root, EXPORT_DIRECTORY)
    scenarios_root = _child_directory(root, SCENARIOS_DIRECTORY)
    aggregate_path = root / AGGREGATE_FILENAME
    if require_aggregate and (aggregate_path.is_symlink() or not aggregate_path.is_file()):
        raise ValueError("aggregate result must be a regular file")

    (
        checkpoint,
        checkpoint_hash,
        checkpoint_iteration,
        transitions,
        values,
        event,
        checkpoints,
        train_commit,
    ) = _check_training(root, train)
    ts_error, onnx_error, ts_path, onnx_path, parity_samples = _check_export(
        root, export, checkpoint, checkpoint_hash, checkpoint_iteration, train_commit
    )
    expected_runs = {
        f"{scenario}_seed{seed}" for scenario in SCENARIOS for seed in SEEDS
    }
    if {path.name for path in scenarios_root.iterdir()} != expected_runs:
        raise ValueError("scenario matrix must contain exactly four scenarios by three seeds")
    rows: list[dict[str, Any]] = []
    scenario_refs: dict[str, Mapping[str, Any]] = {}
    default_heights: list[float] = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            key = f"{scenario}_seed{seed}"
            run = _child_directory(scenarios_root, key)
            row, refs = _check_scenario(
                run,
                scenario,
                seed,
                checkpoint,
                checkpoint_hash,
                checkpoint_iteration,
                train_commit,
            )
            rows.append(row)
            scenario_refs[key] = refs
            if scenario != "crouch":
                play = _load_json(run / "result.json", f"{key} result")["play"]
                default_heights.append(float(play["command"][3]))
    if not default_heights or any(value != default_heights[0] for value in default_heights):
        raise ValueError("stand/forward/turn must resolve one identical default height")
    behavior = evaluate_behavior({"rows": rows})
    if behavior != {"passed": True, "reasons": []}:
        raise ValueError(f"behavior gate failed: {behavior['reasons']}")

    first_mean = sum(values["Train/mean_episode_length"][:100]) / 100
    final_mean = sum(values["Train/mean_episode_length"][-100:]) / 100
    positive_timeouts = sum(
        value > 0 for value in values["Episode_Termination/time_out"][-100:]
    )
    acceptance = {
        "convergence": {
            "passed": True,
            "first_100_mean_episode_length": first_mean,
            "final_100_mean_episode_length": final_mean,
            "positive_timeout_points": positive_timeouts,
            "actual_transitions": transitions,
        },
        "checkpoint": {
            "passed": True,
            "iteration": checkpoint_iteration,
            "sha256": checkpoint_hash,
        },
        "export": {
            "passed": True,
            "torchscript_max_abs_error": ts_error,
            "onnx_max_abs_error": onnx_error,
        },
        "behavior": {
            "passed": True,
            "scenario_means": _scenario_means(rows),
        },
        "overall": {"passed": True},
    }
    expected_artifacts = {
        "canonical_train": {
            "manifest": _file_reference(root, train / "manifest.json"),
            "result": _file_reference(root, train / "result.json"),
            "event": _file_reference(root, event),
            "checkpoints": {
                str(iteration): _file_reference(root, path) for iteration, path in checkpoints.items()
            },
        },
        "exact_export": {
            "manifest": _file_reference(root, export / "manifest.json"),
            "result": _file_reference(root, export / "result.json"),
            "torchscript": _file_reference(root, ts_path),
            "onnx": _file_reference(root, onnx_path),
            "parity_samples": _file_reference(root, parity_samples),
        },
        "scenarios": scenario_refs,
    }
    expected_aggregate = {
        "schema_version": 1,
        "status": "PASS",
        "c1_sources": dict(FROZEN_C1_SOURCES),
        "artifacts": expected_artifacts,
        "acceptance": acceptance,
    }
    result = {
        "status": "PASS",
        "actual_transitions": transitions,
        "positive_timeout_points": positive_timeouts,
        "checkpoint_iteration": checkpoint_iteration,
        "checkpoint_sha256": checkpoint_hash,
        "scenario_runs": len(rows),
        "torchscript_max_abs_error": ts_error,
        "onnx_max_abs_error": onnx_error,
    }
    if not require_aggregate:
        return expected_aggregate, result

    aggregate = _load_json(aggregate_path, "aggregate result")
    if not isinstance(aggregate, Mapping):
        raise TypeError("aggregate result must be a mapping")
    if aggregate.get("schema_version") != 1 or aggregate.get("status") != "PASS":
        raise ValueError("aggregate result is not schema-1 PASS")
    if aggregate.get("c1_sources") != FROZEN_C1_SOURCES:
        raise ValueError("aggregate C1 source hashes do not match the frozen contract")
    artifacts = aggregate.get("artifacts")
    if artifacts != expected_artifacts:
        raise ValueError("aggregate artifact references are incomplete or mismatched")
    for group, entries in expected_artifacts.items():
        if group == "scenarios":
            for key, references in entries.items():
                for kind in ("manifest", "result"):
                    expected = root / references[kind]["path"]
                    _check_reference(root, artifacts[group][key][kind], expected, f"{key} {kind}")
        else:
            for kind, reference in entries.items():
                if kind == "checkpoints":
                    for iteration, checkpoint_ref in reference.items():
                        expected = root / checkpoint_ref["path"]
                        _check_reference(root, artifacts[group][kind][iteration], expected, f"{group} checkpoint {iteration}")
                else:
                    expected = root / reference["path"]
                    _check_reference(root, artifacts[group][kind], expected, f"{group} {kind}")
    if aggregate.get("acceptance") != acceptance:
        raise ValueError("aggregate acceptance claims do not match recomputed evidence")
    return expected_aggregate, result



def build_aggregate_payload(
    evidence_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate a raw three-directory evidence root and build its aggregate."""
    aggregate, _ = _evaluate_evidence(
        evidence_root, require_aggregate=False
    )
    return aggregate


def verify_convergence_evidence(
    evidence_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify one immutable evidence root including its aggregate result."""
    _, result = _evaluate_evidence(evidence_root, require_aggregate=True)
    return result

def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(allow_abbrev=False)
    parser.add_argument("--evidence-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    exit_code = 1
    try:
        args = _parser().parse_args(argv)
        evidence = verify_convergence_evidence(args.evidence_root)
        print(json.dumps(evidence, sort_keys=True, allow_nan=False), flush=True)
        exit_code = 0
    except Exception:
        traceback.print_exc()
    sentinel = "M5_CONVERGENCE_PASS" if exit_code == 0 else "M5_CONVERGENCE_FAIL"
    print(sentinel, flush=True)
    print(f"M5_INTERNAL_EXIT_CODE={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
