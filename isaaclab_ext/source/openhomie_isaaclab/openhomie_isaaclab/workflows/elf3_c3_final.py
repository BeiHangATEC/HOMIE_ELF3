"""Pure-CPU contract for ELF3 C3 final evidence inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from openhomie_isaaclab.workflows.elf3_c3 import (
    LOCAL_ITERATIONS,
    NUM_ENVS,
    NUM_STEPS_PER_ENV,
    validate_c3_manifest,
    validate_c3_result,
)
from openhomie_isaaclab.workflows.elf3_run import (
    EVALUATION_NUM_ENVS,
    EVALUATION_SCENARIOS,
    EVALUATION_STEPS,
    evaluate_behavior,
    sha256_file,
    validate_manifest,
    write_json_once,
)


SCHEMA_VERSION = 1
KIND = "elf3_m5_c3_final_evidence_contract"
FINAL_STAGE = "V3"
FINAL_LOCAL_ITERATION = 2000
FINAL_GLOBAL_ITERATION = 10000
FINAL_GLOBAL_START = 8000
APPROVED_PLAN_SHA256 = (
    "14c30aebfb9095de014837997dcecf5acf9dbe43b9adff5db80cb306bfc1d778"
)
EXPORT_DIRECTORY = "exact_export"
SCENARIOS_DIRECTORY = "scenarios"
AGGREGATE_FILENAME = "aggregate_result.json"
SCENARIOS = ("stand", "forward", "turn", "crouch")
SEEDS = (42, 43, 44)
TS_MAX_ERROR = 1.0e-7
ONNX_MAX_ERROR = 1.0e-5
METRIC_KEYS = {
    "stand": {"height_mae", "tilt_rms"},
    "forward": {"velocity_mae", "height_mae"},
    "turn": {"yaw_rate_mae", "height_mae"},
    "crouch": {"height_mae", "planar_speed_rms"},
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


@dataclass(frozen=True)
class C3FinalRequest:
    evidence_root: Path
    checkpoint: Path
    source_manifest: Path
    source_result: Path
    plan: Path
    plan_sha256: str

    @property
    def aggregate(self) -> Path:
        return self.evidence_root / AGGREGATE_FILENAME


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
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
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


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, name: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular file")

    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    _walk_finite(payload, name)
    return payload


def _regular_absolute_file(value: str | os.PathLike[str], name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an absolute non-symlink regular file")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError(f"{name} must be canonical")
    return resolved


def _root_directory(value: str | os.PathLike[str]) -> Path:
    root = Path(value).expanduser()
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("evidence root must be an absolute non-symlink directory")
    resolved = root.resolve(strict=True)
    if resolved != root:
        raise ValueError("evidence root must be canonical")
    return resolved


def _child_directory(root: Path, name: str) -> Path:
    path = root / name
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"missing non-symlink directory: {name}")
    if path.resolve(strict=True).parent != root:
        raise ValueError(f"directory escapes evidence root: {name}")
    return path.resolve(strict=True)


def _absolute_exact(value: Any, expected: Path, name: str) -> None:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    candidate = Path(value)
    if candidate.is_symlink() or candidate.resolve(strict=True) != expected:
        raise ValueError(f"{name} does not identify the bound artifact")


def _file_reference(path: Path, *, root: Path | None = None) -> dict[str, str]:
    recorded = str(path) if root is None else path.relative_to(root).as_posix()
    return {"path": recorded, "sha256": sha256_file(path)}


def _check_reference(root: Path, reference: Any, expected: Path, name: str) -> None:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"{name} reference is malformed")
    raw = reference["path"]
    if not isinstance(raw, str) or "\\" in raw:
        raise TypeError(f"{name} reference path must be POSIX")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{name} reference path is not canonical")
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink() or candidate.resolve(strict=True) != expected:
        raise ValueError(f"{name} reference points to another artifact")
    if sha256_file(candidate) != _hash(reference["sha256"], f"{name} hash"):
        raise ValueError(f"{name} reference hash does not match")


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="Check the CPU-only ELF3 C3 final evidence contract"
    )
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-result", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    return parser


def parse_request(argv: Sequence[str] | None = None) -> C3FinalRequest:
    args = build_parser().parse_args(argv)
    request = C3FinalRequest(
        evidence_root=_root_directory(args.evidence_root),
        checkpoint=_regular_absolute_file(args.checkpoint, "checkpoint"),
        source_manifest=_regular_absolute_file(args.source_manifest, "source manifest"),
        source_result=_regular_absolute_file(args.source_result, "source result"),
        plan=_regular_absolute_file(args.plan, "approved plan"),
        plan_sha256=_hash(args.plan_sha256, "approved plan SHA-256"),
    )
    if request.plan_sha256 != APPROVED_PLAN_SHA256:
        raise ValueError("plan SHA-256 is not the approved C3 design")
    if sha256_file(request.plan) != request.plan_sha256:
        raise ValueError("approved plan SHA-256 does not match the file")
    return request


def _check_config_hashes(manifest: Mapping[str, Any], name: str) -> None:
    configs = manifest.get("configs")
    if not isinstance(configs, Mapping):
        raise TypeError(f"{name} configs must be a mapping")
    hashes = configs.get("sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != {"env", "agent"}:
        raise ValueError(f"{name} config hash inventory is incomplete")
    for kind in ("env", "agent"):
        if not isinstance(configs.get(kind), Mapping):
            raise TypeError(f"{name} {kind} config must be a mapping")
        if hashes[kind] != _json_hash(configs[kind]):
            raise ValueError(f"{name} {kind} config hash does not match")


def _check_source_binding(request: C3FinalRequest) -> dict[str, Any]:
    manifest = _load_json(request.source_manifest, "source C3 manifest")
    result = _load_json(request.source_result, "source C3 result")
    validate_c3_manifest(manifest)
    validate_c3_result(result, manifest)
    run = request.source_manifest.parent
    if request.source_result.parent != run or request.checkpoint.parent != run:
        raise ValueError("source manifest, result, and checkpoint must share one V3 run")
    if request.source_manifest.name != "manifest.json" or request.source_result.name != "result.json":
        raise ValueError("source evidence filenames are not canonical")
    if request.checkpoint.name != "model_2000.pt":
        raise ValueError("final C3 checkpoint must be named model_2000.pt")
    if manifest["stage"] != FINAL_STAGE or result.get("status") != "PASS":
        raise ValueError("source evidence must be a passing C3 V3 stage")
    lifecycle = manifest["lifecycle"]
    if (
        lifecycle.get("local_iterations") != {"start": 0, "final": FINAL_LOCAL_ITERATION}
        or lifecycle.get("global_iterations")
        != {"start": FINAL_GLOBAL_START, "final": FINAL_GLOBAL_ITERATION}
        or result.get("lifecycle") != lifecycle
    ):
        raise ValueError("source lifecycle is not V3 local-2000/global-10000")
    if manifest.get("iterations") != LOCAL_ITERATIONS or manifest.get("num_envs") != NUM_ENVS:
        raise ValueError("source V3 training budget is not canonical")
    _absolute_exact(manifest["cli"].get("run_dir"), run, "source run directory")
    _absolute_exact(manifest["plan"].get("path"), request.plan, "source plan")
    if manifest["plan"].get("sha256") != request.plan_sha256:
        raise ValueError("source V3 stage used a different C3 plan")
    checkpoint = result.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("source result checkpoint identity is missing")
    _absolute_exact(checkpoint.get("path"), request.checkpoint, "source checkpoint")
    checkpoint_hash = sha256_file(request.checkpoint)
    if checkpoint.get("sha256") != checkpoint_hash:
        raise ValueError("source checkpoint SHA-256 does not match")
    expected_checkpoint = {
        "iteration": FINAL_LOCAL_ITERATION,
        "stage": FINAL_STAGE,
        "local_iteration": FINAL_LOCAL_ITERATION,
        "global_iteration": FINAL_GLOBAL_ITERATION,
    }
    for key, expected in expected_checkpoint.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"source checkpoint {key} is inconsistent")
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("source result metrics are missing")
    expected_transitions = manifest["lifecycle"]["transitions"]
    if metrics.get("transitions") != expected_transitions:
        raise ValueError("source result transition budget is incomplete")
    finite = metrics.get("finite")
    if not isinstance(finite, Mapping) or not finite or any(value is not True for value in finite.values()):
        raise ValueError("source V3 training finite evidence is incomplete")
    _check_config_hashes(manifest, "source C3 manifest")
    commit = manifest.get("git", {}).get("commit")
    if not isinstance(commit, str) or _GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError("source Git commit is invalid")
    c3_sources = manifest.get("c3_sources")
    if not isinstance(c3_sources, Mapping) or c3_sources.get("sha256") != _json_hash(c3_sources.get("files")):
        raise ValueError("source C3 source aggregate hash does not match")
    return {
        "checkpoint": {
            **_file_reference(request.checkpoint),
            "stage": FINAL_STAGE,
            "local_iteration": FINAL_LOCAL_ITERATION,
            "global_iteration": FINAL_GLOBAL_ITERATION,
        },
        "manifest": _file_reference(request.source_manifest),
        "result": _file_reference(request.source_result),
        "plan": _file_reference(request.plan),
        "git_commit": commit,
        "transitions": expected_transitions,
    }


def _check_runtime_manifest(
    payload: Any,
    *,
    command: str,
    run: Path,
    seed: int,
    num_envs: int,
    checkpoint: Path,
    checkpoint_hash: str,
    source_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_manifest(payload)
    expected = {
        "schema_version": 1,
        "command": command,
        "seed": seed,
        "device": source_manifest["device"],
        "num_envs": num_envs,
        "iterations": None,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{run.name} manifest {key} does not match")
    cli = payload.get("cli")
    if not isinstance(cli, Mapping):
        raise TypeError(f"{run.name} CLI evidence is missing")
    for key, value in expected.items():
        if key != "schema_version" and cli.get(key) != value:
            raise ValueError(f"{run.name} CLI {key} does not match")
    if cli.get("headless") is not True or cli.get("resume") is not False:
        raise ValueError(f"{run.name} must be a fresh headless evidence run")
    _absolute_exact(cli.get("run_dir"), run, f"{run.name} run directory")
    _absolute_exact(cli.get("checkpoint"), checkpoint, f"{run.name} CLI checkpoint")
    source = payload.get("checkpoint")
    if not isinstance(source, Mapping):
        raise TypeError(f"{run.name} source checkpoint identity is missing")
    _absolute_exact(source.get("path"), checkpoint, f"{run.name} checkpoint")
    if source.get("sha256") != checkpoint_hash or source.get("iteration") != FINAL_LOCAL_ITERATION:
        raise ValueError(f"{run.name} uses a foreign checkpoint identity")
    if payload.get("git", {}).get("commit") != source_manifest.get("git", {}).get("commit"):
        raise ValueError(f"{run.name} belongs to a foreign Git commit")
    for name in ("assets", "m4_sources"):
        if payload.get(name) != source_manifest.get(name):
            raise ValueError(f"{run.name} {name} differ from the V3 source")
    _check_config_hashes(payload, f"{run.name} manifest")
    if payload["configs"]["env"].get("command_stage") != FINAL_STAGE:
        raise ValueError(f"{run.name} runtime command stage is not V3")
    return payload


def _check_export(
    root: Path,
    export: Path,
    request: C3FinalRequest,
    source_manifest: Mapping[str, Any],
    checkpoint_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_files = {
        "manifest.json", "result.json", "policy.ts", "policy.onnx", "parity_samples.npz"
    }
    if {path.name for path in export.iterdir()} != expected_files:
        raise ValueError("exact export inventory is not exact")
    manifest_path = export / "manifest.json"
    result_path = export / "result.json"
    manifest = _check_runtime_manifest(
        _load_json(manifest_path, "export manifest"),
        command="export",
        run=export,
        seed=42,
        num_envs=1,
        checkpoint=request.checkpoint,
        checkpoint_hash=checkpoint_hash,
        source_manifest=source_manifest,
    )
    cli = manifest["cli"]
    if cli.get("scenario") is not None or cli.get("steps") is not None:
        raise ValueError("export CLI contains scenario fields")
    result = _load_json(result_path, "export result")
    if not isinstance(result, Mapping) or result.get("status") != "PASS":
        raise ValueError("export result is not PASS")
    exports = result.get("exports")
    if not isinstance(exports, Mapping) or set(exports) != {"oracle", "torchscript", "onnx"}:
        raise ValueError("export evidence inventory is incomplete")
    if exports["oracle"] != {"device": "cpu", "method": "runner.get_inference_policy"}:
        raise ValueError("export oracle identity is wrong")
    errors: dict[str, float] = {}
    artifacts: dict[str, Path] = {}
    for kind, filename, limit in (
        ("torchscript", "policy.ts", TS_MAX_ERROR),
        ("onnx", "policy.onnx", ONNX_MAX_ERROR),
    ):
        evidence = exports.get(kind)
        if not isinstance(evidence, Mapping):
            raise TypeError(f"{kind} evidence is missing")
        artifact = (export / filename).resolve(strict=True)
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"{kind} artifact must be a regular file")
        _absolute_exact(evidence.get("path"), artifact, f"{kind} artifact")
        if evidence.get("sha256") != sha256_file(artifact):
            raise ValueError(f"{kind} artifact SHA-256 does not match")
        if (
            evidence.get("fresh_runtime") is not True
            or evidence.get("batches") != [1, 4]
            or evidence.get("input_shapes") != [[1, 468], [4, 468]]
            or evidence.get("output_shapes") != [[1, 12], [4, 12]]
        ):
            raise ValueError(f"{kind} fresh-runtime parity identity is wrong")
        error = _number(evidence.get("max_abs_error"), f"{kind}.max_abs_error")
        if error > limit:
            raise ValueError(f"{kind} parity exceeds its threshold")
        errors[kind] = error
        artifacts[kind] = artifact
    if exports["torchscript"].get("provider") != "torch.jit.load":
        raise ValueError("TorchScript provider is wrong")
    if (
        exports["onnx"].get("checker_passed") is not True
        or exports["onnx"].get("providers") != ["CPUExecutionProvider"]
    ):
        raise ValueError("ONNX fresh CPU runtime evidence is wrong")
    samples = export / "parity_samples.npz"
    if samples.is_symlink() or not samples.is_file():
        raise ValueError("parity samples must be a regular file")
    with np.load(samples, allow_pickle=False) as payload:
        expected_shapes = {
            "history_1": (1, 468), "expected_1": (1, 12),
            "history_4": (4, 468), "expected_4": (4, 12),
        }
        if set(payload.files) != set(expected_shapes):
            raise ValueError("parity sample inventory is incomplete")
        for name, shape in expected_shapes.items():
            array = payload[name]
            if array.shape != shape or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
                raise ValueError(f"parity sample {name} is invalid")
    if sha256_file(request.checkpoint) != checkpoint_hash:
        raise ValueError("export evidence collection mutated the V3 checkpoint")
    refs = {
        "manifest": _file_reference(manifest_path, root=root),
        "result": _file_reference(result_path, root=root),
        "torchscript": _file_reference(artifacts["torchscript"], root=root),
        "onnx": _file_reference(artifacts["onnx"], root=root),
        "parity_samples": _file_reference(samples.resolve(strict=True), root=root),
    }
    return errors, refs


def _check_scenario(
    root: Path,
    run: Path,
    scenario: str,
    seed: int,
    request: C3FinalRequest,
    source_manifest: Mapping[str, Any],
    checkpoint_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    if {path.name for path in run.iterdir()} != {"manifest.json", "result.json"}:
        raise ValueError(f"scenario {run.name} inventory is not exact")
    manifest_path = run / "manifest.json"
    result_path = run / "result.json"
    manifest = _check_runtime_manifest(
        _load_json(manifest_path, f"{run.name} manifest"),
        command="play",
        run=run,
        seed=seed,
        num_envs=EVALUATION_NUM_ENVS,
        checkpoint=request.checkpoint,
        checkpoint_hash=checkpoint_hash,
        source_manifest=source_manifest,
    )
    cli = manifest["cli"]
    if cli.get("scenario") != scenario or cli.get("steps") != EVALUATION_STEPS:
        raise ValueError(f"{run.name} scenario CLI does not match")
    result = _load_json(result_path, f"{run.name} result")
    if not isinstance(result, Mapping) or result.get("status") != "PASS":
        raise ValueError(f"{run.name} result is not PASS")
    play = result.get("play")
    if not isinstance(play, Mapping):
        raise TypeError(f"{run.name} play evidence is missing")
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
    _absolute_exact(play.get("checkpoint_path"), request.checkpoint, f"{run.name} checkpoint")
    command = play.get("command")
    if not isinstance(command, list) or len(command) != 4:
        raise ValueError(f"{run.name} command must have four values")
    command_values = [_number(value, f"{run.name} command") for value in command]
    expected_command = EVALUATION_SCENARIOS[scenario].command
    for index in range(3):
        if abs(command_values[index] - expected_command[index]) > 1.0e-6:
            raise ValueError(f"{run.name} command component {index} is wrong")
    if scenario == "crouch":
        if abs(command_values[3] - 0.80) > 1.0e-6:
            raise ValueError("crouch command height must be 0.80")
    elif command_values[3] <= 0.0:
        raise ValueError(f"{run.name} default height must be positive")
    maximum = EVALUATION_NUM_ENVS * EVALUATION_STEPS
    credited = _integer(play.get("credited_env_steps"), f"{run.name} credited steps")
    if credited > maximum:
        raise ValueError(f"{run.name} credited steps exceed rollout")
    steps = play.get("non_timeout_termination_steps")
    reasons = play.get("non_timeout_termination_reasons")
    if not isinstance(steps, list) or len(steps) != EVALUATION_NUM_ENVS:
        raise ValueError(f"{run.name} termination steps are incomplete")
    if not isinstance(reasons, list) or len(reasons) != EVALUATION_NUM_ENVS:
        raise ValueError(f"{run.name} termination reasons are incomplete")
    reconstructed = 0
    for index, (step, reason) in enumerate(zip(steps, reasons, strict=True)):
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
    if _integer(play.get("timeout_count"), f"{run.name} timeout count") > maximum:
        raise ValueError(f"{run.name} timeout count exceeds rollout")
    survival = _number(play.get("survival"), f"{run.name} survival")
    if survival != credited / maximum:
        raise ValueError(f"{run.name} survival does not match credited steps")
    metrics = play.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != METRIC_KEYS[scenario]:
        raise ValueError(f"{run.name} metric inventory is wrong")
    row = {"scenario": scenario, "seed": seed, "finite": True, "survival": survival}
    row.update({name: _number(value, f"{run.name}.{name}") for name, value in metrics.items()})
    action_hash = _hash(play.get("action_sha256"), f"{run.name} action hash")
    trajectory_hash = _hash(play.get("trajectory_sha256"), f"{run.name} trajectory hash")
    for name in ("sha256_before", "sha256_after"):
        if play.get(name) != checkpoint_hash:
            raise ValueError(f"{run.name} checkpoint hash changed")
    if sha256_file(request.checkpoint) != checkpoint_hash:
        raise ValueError(f"{run.name} mutated the V3 checkpoint")
    refs = {
        "manifest": _file_reference(manifest_path, root=root),
        "result": _file_reference(result_path, root=root),
        "action_sha256": action_hash,
        "trajectory_sha256": trajectory_hash,
    }
    return row, refs, command_values[3]


def _scenario_means(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        scenario: {
            metric: sum(float(row[metric]) for row in rows if row["scenario"] == scenario) / len(SEEDS)
            for metric in sorted(METRIC_KEYS[scenario])
        }
        for scenario in SCENARIOS
    }


def build_aggregate_payload(request: C3FinalRequest) -> dict[str, Any]:
    if not isinstance(request, C3FinalRequest):
        raise TypeError("C3 final check requires a parsed C3FinalRequest")
    allowed_top = {EXPORT_DIRECTORY, SCENARIOS_DIRECTORY}
    if request.aggregate.exists() or request.aggregate.is_symlink():
        allowed_top.add(AGGREGATE_FILENAME)
    if {path.name for path in request.evidence_root.iterdir()} != allowed_top:
        raise ValueError("C3 final evidence root inventory is not exact")
    source_binding = _check_source_binding(request)
    source_manifest = _load_json(request.source_manifest, "source C3 manifest")
    checkpoint_hash = source_binding["checkpoint"]["sha256"]
    export = _child_directory(request.evidence_root, EXPORT_DIRECTORY)
    export_errors, export_refs = _check_export(
        request.evidence_root, export, request, source_manifest, checkpoint_hash
    )
    scenarios_root = _child_directory(request.evidence_root, SCENARIOS_DIRECTORY)
    expected_runs = {
        f"{scenario}_seed{seed}" for scenario in SCENARIOS for seed in SEEDS
    }
    if {path.name for path in scenarios_root.iterdir()} != expected_runs:
        raise ValueError("scenario matrix must contain exactly four scenarios by three seeds")
    rows: list[dict[str, Any]] = []
    refs: dict[str, dict[str, Any]] = {}
    default_heights: list[float] = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            key = f"{scenario}_seed{seed}"
            row, reference, height = _check_scenario(
                request.evidence_root,
                _child_directory(scenarios_root, key),
                scenario,
                seed,
                request,
                source_manifest,
                checkpoint_hash,
            )
            rows.append(row)
            refs[key] = reference
            if scenario != "crouch":
                default_heights.append(height)
    if not default_heights or any(height != default_heights[0] for height in default_heights):
        raise ValueError("stand/forward/turn must use one identical default height")
    behavior = evaluate_behavior({"rows": rows})
    if behavior != {"passed": True, "reasons": []}:
        raise ValueError(f"behavior gate failed: {behavior['reasons']}")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "PENDING_CONVERGENCE",
        "contract": {
            "passed": True,
            "scope": ["v3_source_binding", "exact_export", "behavior_12_run_matrix"],
        },
        "source": source_binding,
        "artifacts": {"exact_export": export_refs, "scenarios": refs},
        "acceptance": {
            "checkpoint": {
                "passed": True,
                "stage": FINAL_STAGE,
                "local_iteration": FINAL_LOCAL_ITERATION,
                "global_iteration": FINAL_GLOBAL_ITERATION,
                "sha256": checkpoint_hash,
            },
            "export": {
                "passed": True,
                "torchscript_max_abs_error": export_errors["torchscript"],
                "onnx_max_abs_error": export_errors["onnx"],
            },
            "behavior": {
                "passed": True,
                "scenario_runs": len(rows),
                "scenario_means": _scenario_means(rows),
            },
            "overall": {
                "passed": False,
                "status": "PENDING_CONVERGENCE",
                "reason": "C3 convergence evidence basis and acceptance window are not yet approved",
            },
        },
    }


def verify_final_contract(request: C3FinalRequest) -> dict[str, Any]:
    aggregate = build_aggregate_payload(request)
    if request.aggregate.exists() or request.aggregate.is_symlink():
        recorded = _load_json(request.aggregate, "C3 final aggregate")
        if recorded != aggregate:
            raise ValueError("recorded aggregate differs from recomputed C3 final contract")
        artifacts = aggregate["artifacts"]
        for kind, reference in artifacts["exact_export"].items():
            _check_reference(
                request.evidence_root,
                reference,
                request.evidence_root / reference["path"],
                f"exact export {kind}",
            )
        for key, references in artifacts["scenarios"].items():
            for kind in ("manifest", "result"):
                reference = references[kind]
                _check_reference(
                    request.evidence_root,
                    reference,
                    request.evidence_root / reference["path"],
                    f"{key} {kind}",
                )
    return aggregate


def write_or_verify_aggregate(request: C3FinalRequest) -> dict[str, Any]:
    aggregate = verify_final_contract(request)
    if not request.aggregate.exists():
        write_json_once(request.aggregate, aggregate)
    return aggregate
