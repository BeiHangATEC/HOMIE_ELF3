#!/usr/bin/env python3
"""Independent CPU checker for ELF3 M5 C3 signed-response evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (42, 43, 44)
COMMANDS = np.asarray((0.0, 0.1, 0.2, 0.3, 0.4, 0.5), dtype=np.float64)
WINDOWS = ("full", "post_initial")
BOUNDARIES = {"forward": 0.3, "yaw": 0.2}
BEHAVIOR_LIMITS = {"forward": 0.20, "yaw": 0.25}
IN_MIN = 0.70
IN_MAX = 1.30
OUT_MAX = 0.50
OUT_TO_IN_MAX = 0.60
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
STEPS = 1000
NUM_ENVS = 16
HEIGHT = 1.01
MODE = 0
REQUIRED_ARRAYS = {
    "step_index": ("int64", (STEPS,)),
    "command": ("float32", (STEPS, NUM_ENVS, 4)),
    "mode": ("int64", (STEPS, NUM_ENVS)),
    "root_lin_vel_b": ("float32", (STEPS, NUM_ENVS, 3)),
    "root_ang_vel_b": ("float32", (STEPS, NUM_ENVS, 3)),
    "roll_pitch": ("float32", (STEPS, NUM_ENVS, 2)),
    "tracking_height": ("float32", (STEPS, NUM_ENVS)),
    "action": ("float32", (STEPS, NUM_ENVS, 12)),
    "reward": ("float32", (STEPS, NUM_ENVS)),
    "active_before": ("bool", (STEPS, NUM_ENVS)),
    "done": ("bool", (STEPS, NUM_ENVS)),
    "timeout": ("bool", (STEPS, NUM_ENVS)),
}


def compute_window_metrics(
    actual: Any, target: Any, active: Any
) -> dict[str, int | float]:
    """Compute signed response and tracking errors over credited samples."""
    values = np.asarray(actual)
    mask = np.asarray(active)
    if values.shape != mask.shape or values.size == 0:
        raise ValueError("actual and active must have the same nonempty shape")
    if mask.dtype != np.bool_:
        raise TypeError("active must have bool dtype")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError("actual must contain only finite numeric values")
    try:
        expected = np.broadcast_to(np.asarray(target, dtype=np.float64), values.shape)
    except ValueError as exc:
        raise ValueError("target is not broadcastable to actual") from exc
    if not np.all(np.isfinite(expected)):
        raise ValueError("target must be finite")
    credited = int(np.count_nonzero(mask))
    if credited == 0:
        raise ValueError("window has no credited samples")
    selected = values.astype(np.float64, copy=False)[mask]
    errors = selected - expected[mask]
    return {
        "credited_env_steps": credited,
        "signed_mean": float(np.mean(selected, dtype=np.float64)),
        "mae": float(np.mean(np.abs(errors), dtype=np.float64)),
        "rmse": float(math.sqrt(float(np.mean(np.square(errors), dtype=np.float64)))),
    }


def _invalid(reason: str) -> dict[str, Any]:
    return {"status": "INVALID", "reasons": [reason], "windows": {}}


def _validate_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _gain(x: np.ndarray, y: np.ndarray) -> float:
    denominator = float(np.dot(x, x))
    if denominator <= 0.0:
        raise ValueError("gain fit has no support")
    return float(np.dot(x, y) / denominator)


def _window_result(
    responses: np.ndarray,
    mae: np.ndarray,
    commands: np.ndarray,
    boundary: float,
    limit: float,
) -> dict[str, Any]:
    corrected = responses - responses[:, :1]
    boundary_indices = np.flatnonzero(np.isclose(commands, boundary, rtol=0.0, atol=1.0e-12))
    if boundary_indices.size != 1:
        raise ValueError("support boundary is absent or ambiguous")
    boundary_index = int(boundary_indices[0])
    in_indices = np.flatnonzero((commands > 0.0) & (commands <= boundary + 1.0e-12))
    out_indices = np.flatnonzero(commands > boundary + 1.0e-12)
    if in_indices.size == 0 or out_indices.size == 0:
        raise ValueError("commands do not span both sides of the support boundary")

    seed_rows: list[dict[str, float | int | bool]] = []
    for seed_index, seed in enumerate(SEEDS):
        curve = corrected[seed_index]
        g_in = _gain(commands[in_indices], curve[in_indices])
        post_x = commands[out_indices] - boundary
        post_y = curve[out_indices] - curve[boundary_index]
        g_out = _gain(post_x, post_y)
        endpoint = float(curve[-1])
        seed_rows.append(
            {
                "seed": seed,
                "g_in": g_in,
                "g_out": g_out,
                "endpoint_response": endpoint,
                "veto": not (g_in > 0.0 and endpoint > 0.0 and g_out < g_in),
            }
        )

    mean_curve = np.mean(corrected, axis=0, dtype=np.float64)
    g_in = _gain(commands[in_indices], mean_curve[in_indices])
    g_out = _gain(
        commands[out_indices] - boundary,
        mean_curve[out_indices] - mean_curve[boundary_index],
    )
    command_mae = np.mean(mae, axis=0, dtype=np.float64)
    in_mae_indices = in_indices - 1
    return {
        "g_in": g_in,
        "g_out": g_out,
        "out_to_in": float(g_out / g_in) if g_in != 0.0 else math.inf,
        "endpoint_response": float(mean_curve[-1]),
        "endpoint_mae": float(command_mae[-1]),
        "in_support_mae": [float(command_mae[index]) for index in in_mae_indices],
        "in_support_passed": bool(np.all(command_mae[in_mae_indices] <= limit)),
        "seed_metrics": seed_rows,
    }


def _axis_evidence(axis: str, evidence: Any) -> tuple[dict[str, dict[str, Any]], float]:
    if axis not in BOUNDARIES:
        raise ValueError("axis must be forward or yaw")
    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a mapping")
    if evidence.get("axis") != axis:
        raise ValueError("evidence axis does not match the requested axis")
    if evidence.get("seeds") != list(SEEDS):
        raise ValueError("evidence must contain seeds 42, 43, and 44 in order")
    commands = np.asarray(evidence.get("commands"))
    if commands.shape != COMMANDS.shape or not np.issubdtype(commands.dtype, np.number):
        raise ValueError("commands have the wrong shape or dtype")
    commands = commands.astype(np.float64, copy=False)
    if not np.all(np.isfinite(commands)) or not np.array_equal(commands, COMMANDS):
        raise ValueError("commands must equal the canonical 0.0 through 0.5 grid")
    windows = evidence.get("windows")
    if not isinstance(windows, Mapping) or set(windows) != set(WINDOWS):
        raise ValueError("evidence must contain exactly the full and post_initial windows")

    results: dict[str, dict[str, Any]] = {}
    for name in WINDOWS:
        payload = windows[name]
        if not isinstance(payload, Mapping) or set(payload) != {"responses", "mae"}:
            raise ValueError(f"{name} must contain exactly responses and mae")
        responses = np.asarray(payload["responses"])
        mae = np.asarray(payload["mae"])
        if responses.shape != (3, 6) or mae.shape != (3, 5):
            raise ValueError(f"{name} response or MAE shape is invalid")
        if not np.issubdtype(responses.dtype, np.number) or not np.issubdtype(mae.dtype, np.number):
            raise TypeError(f"{name} response and MAE arrays must be numeric")
        responses = responses.astype(np.float64, copy=False)
        mae = mae.astype(np.float64, copy=False)
        if not np.all(np.isfinite(responses)) or not np.all(np.isfinite(mae)):
            raise ValueError(f"{name} contains non-finite evidence")
        if np.any(mae < 0.0):
            raise ValueError(f"{name} MAE values must be nonnegative")
        results[name] = _window_result(
            responses, mae, commands, BOUNDARIES[axis], BEHAVIOR_LIMITS[axis]
        )
    return results, BEHAVIOR_LIMITS[axis]


def _at_most(value: float, limit: float) -> bool:
    return value <= limit or math.isclose(value, limit, rel_tol=0.0, abs_tol=1.0e-12)


def _in_closed_interval(value: float, lower: float, upper: float) -> bool:
    return _at_most(lower, value) and _at_most(value, upper)


def classify_axis(axis: str, evidence: Any) -> dict[str, Any]:
    """Classify one axis, returning INVALID rather than trusting malformed data."""
    try:
        windows, behavior_limit = _axis_evidence(axis, evidence)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return _invalid(str(exc))

    result: dict[str, Any] = {"axis": axis, "windows": windows, "reasons": []}
    if any(_at_most(window["endpoint_mae"], behavior_limit) for window in windows.values()):
        result.update(status="GATE_PASSED")
        result["reasons"].append("the 0.5 command passes the frozen behavior gate")
        return result
    if any(not window["in_support_passed"] for window in windows.values()):
        result.update(status="IN_SUPPORT_DEFICIT")
        result["reasons"].append("an in-support command exceeds the frozen behavior gate")
        return result

    gain_passed = all(
        _in_closed_interval(window["g_in"], IN_MIN, IN_MAX)
        and _at_most(window["g_out"], OUT_MAX)
        and _at_most(window["out_to_in"], OUT_TO_IN_MAX)
        and window["endpoint_response"] > 0.0
        for window in windows.values()
    )
    if not gain_passed:
        result.update(status="NO_OOD_BREAK", vetoed_seeds=[])
        result["reasons"].append("aggregate gains do not satisfy the frozen OOD limits")
        return result

    vetoed = sorted(
        {
            int(seed_row["seed"])
            for window in windows.values()
            for seed_row in window["seed_metrics"]
            if seed_row["veto"]
        }
    )
    if vetoed:
        result.update(status="MIXED", vetoed_seeds=vetoed)
        result["reasons"].append("one or more seeds contradict the aggregate gain break")
        return result

    subtype = (
        "DEGRADATION"
        if any(window["g_out"] < 0.0 for window in windows.values())
        else "SATURATION"
    )
    result.update(status="OOD_SUPPORTED", subtype=subtype, vetoed_seeds=[])
    return result


def classify_grid(
    axis_results: Any, *, plan_sha256: str, checkpoint_sha256: str
) -> dict[str, Any]:
    """Authorize warm start only when both independently checked axes agree."""
    plan_hash = _validate_sha256(plan_sha256, "plan_sha256")
    checkpoint_hash = _validate_sha256(checkpoint_sha256, "checkpoint_sha256")
    if not isinstance(axis_results, Mapping) or set(axis_results) != set(BOUNDARIES):
        raise ValueError("axis_results must contain exactly forward and yaw")
    statuses: dict[str, str] = {}
    for axis in ("forward", "yaw"):
        result = axis_results[axis]
        if not isinstance(result, Mapping) or not isinstance(result.get("status"), str):
            raise TypeError(f"{axis} result has no status")
        statuses[axis] = result["status"]
    authorized = all(status == "OOD_SUPPORTED" for status in statuses.values())
    return {
        "status": "WARM_START_AUTHORIZED" if authorized else "STOPPED",
        "authorized": authorized,
        "plan_sha256": plan_hash,
        "checkpoint_sha256": checkpoint_hash,
        "axis_status": statuses,
        "axis_results": dict(axis_results),
    }


def build_authorization(decision: Any) -> dict[str, Any]:
    """Build the hash-bound promotion record for an authorized grid only."""
    if not isinstance(decision, Mapping):
        raise TypeError("decision must be a mapping")
    if decision.get("status") != "WARM_START_AUTHORIZED" or decision.get("authorized") is not True:
        raise ValueError("decision is not authorized")
    statuses = decision.get("axis_status")
    if statuses != {"forward": "OOD_SUPPORTED", "yaw": "OOD_SUPPORTED"}:
        raise ValueError("both axes must be authorized")
    return {
        "schema_version": 1,
        "kind": "elf3_m5_c3_warm_start_authorization",
        "status": "WARM_START_AUTHORIZED",
        "plan_sha256": _validate_sha256(decision.get("plan_sha256"), "plan_sha256"),
        "checkpoint_sha256": _validate_sha256(
            decision.get("checkpoint_sha256"), "checkpoint_sha256"
        ),
        "axis_status": dict(statuses),
    }


def _canonical_actions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        baseline = f"seed{seed}_walk_zero"
        rows.append(
            {
                "action_id": baseline,
                "seed": seed,
                "scenario": "walk_zero",
                "command": [0.0, 0.0, 0.0, HEIGHT],
                "mode": MODE,
                "steps": STEPS,
                "num_envs": NUM_ENVS,
                "baseline_action_id": None,
            }
        )
        for scenario, index in (("forward", 0), ("yaw", 2)):
            for value in (0.1, 0.2, 0.3, 0.4, 0.5):
                command = [0.0, 0.0, 0.0, HEIGHT]
                command[index] = value
                rows.append(
                    {
                        "action_id": (
                            f"seed{seed}_{scenario}_{value:.1f}".replace(".", "p")
                        ),
                        "seed": seed,
                        "scenario": scenario,
                        "command": command,
                        "mode": MODE,
                        "steps": STEPS,
                        "num_envs": NUM_ENVS,
                        "baseline_action_id": baseline,
                    }
                )
    return rows


def _regular_file(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _root_directory(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError("evidence root must be an absolute non-symlink directory")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError("evidence root must be canonical")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_strict_json(path: Path, name: str) -> Mapping[str, Any]:
    source = _regular_file(path, name)

    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains {value}")

    payload = json.loads(
        source.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return payload


def _derive_active_before(done: np.ndarray, timeout: np.ndarray) -> np.ndarray:
    if done.dtype != np.bool_ or timeout.dtype != np.bool_ or done.shape != timeout.shape:
        raise ValueError("done and timeout must be matching bool arrays")
    if np.any(timeout & ~done):
        raise ValueError("timeout must imply done")
    active = np.ones(done.shape[1], dtype=np.bool_)
    derived = np.empty_like(done)
    for step in range(done.shape[0]):
        derived[step] = active
        active &= ~(done[step] & ~timeout[step])
    return derived


def _load_trajectory(path: Path, action: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, int | bool]]:
    source = _regular_file(path, "trajectory")
    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != set(REQUIRED_ARRAYS):
                raise ValueError("trajectory array inventory is wrong")
            arrays = {name: archive[name] for name in REQUIRED_ARRAYS}
    except (OSError, ValueError) as exc:
        raise ValueError(f"trajectory cannot be loaded safely: {exc}") from exc
    for name, (dtype, shape) in REQUIRED_ARRAYS.items():
        value = arrays[name]
        if value.dtype != np.dtype(dtype) or value.shape != shape:
            raise ValueError(f"trajectory {name} has the wrong dtype or shape")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"trajectory {name} contains non-finite values")
    if not np.array_equal(arrays["step_index"], np.arange(STEPS, dtype=np.int64)):
        raise ValueError("trajectory step_index is not canonical")
    command = np.asarray(action["command"], dtype=np.float32)
    if not np.array_equal(
        arrays["command"], np.broadcast_to(command, arrays["command"].shape)
    ):
        raise ValueError("trajectory command is not fixed")
    if not np.all(arrays["mode"] == action["mode"]):
        raise ValueError("trajectory mode is not fixed")
    active = _derive_active_before(arrays["done"], arrays["timeout"])
    if not np.array_equal(active, arrays["active_before"]):
        raise ValueError("trajectory active credit mask is inconsistent")
    summary = {
        "finite": True,
        "credited_env_steps": int(np.count_nonzero(active)),
        "timeout_count": int(np.count_nonzero(arrays["timeout"] & active)),
        "non_timeout_termination_count": int(
            np.count_nonzero(arrays["done"] & ~arrays["timeout"] & active)
        ),
    }
    if summary != {
        "finite": True,
        "credited_env_steps": STEPS * NUM_ENVS,
        "timeout_count": NUM_ENVS,
        "non_timeout_termination_count": 0,
    }:
        raise ValueError("trajectory survival, credit, or timeout evidence is invalid")
    return arrays, summary


def _action_trajectory(
    root: Path, action: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, int | bool]]:
    directory = root / action["action_id"]
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("missing non-symlink action evidence directory")
    if {entry.name for entry in directory.iterdir()} != {
        "trajectory.npz",
        "result.json",
    }:
        raise ValueError("action evidence directory inventory is wrong")
    trajectory = _regular_file(directory / "trajectory.npz", "trajectory")
    result = _load_strict_json(directory / "result.json", "action result")
    if result.get("status") != "PASS" or result.get("action") != dict(action):
        raise ValueError("action result identity is invalid")
    record = result.get("trajectory")
    if not isinstance(record, Mapping):
        raise TypeError("action trajectory result is missing")
    arrays, summary = _load_trajectory(trajectory, action)
    if (
        record.get("path") != "trajectory.npz"
        or record.get("sha256") != _sha256_file(trajectory)
        or {key: record.get(key) for key in summary} != summary
    ):
        raise ValueError("action trajectory result does not bind the raw NPZ")
    return arrays, summary


def _validated_plan(plan_path: Path) -> Mapping[str, Any]:
    plan = _load_strict_json(plan_path, "signed-response plan")
    if (
        plan.get("schema_version") != 1
        or plan.get("kind") != "elf3_m5_c3_signed_response"
        or plan.get("seeds") != list(SEEDS)
        or plan.get("num_envs") != NUM_ENVS
        or plan.get("steps") != STEPS
        or plan.get("command_values") != [0.1, 0.2, 0.3, 0.4, 0.5]
        or plan.get("support_boundaries") != BOUNDARIES
        or plan.get("behavior_limits") != BEHAVIOR_LIMITS
        or plan.get("gain_limits")
        != {"in_min": IN_MIN, "in_max": IN_MAX, "out_max": OUT_MAX, "out_to_in_max": OUT_TO_IN_MAX}
        or plan.get("actions") != _canonical_actions()
    ):
        raise ValueError("signed-response plan is not the frozen canonical grid")
    source = plan.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("checkpoint_iteration") != 4000
        or source.get("stage") != "S0"
        or not isinstance(source.get("checkpoint_sha256"), str)
        or _SHA256.fullmatch(source["checkpoint_sha256"]) is None
        or not isinstance(source.get("manifest_sha256"), str)
        or _SHA256.fullmatch(source["manifest_sha256"]) is None
    ):
        raise ValueError("signed-response plan source identity is invalid")
    return plan


def _canonical_plan_hash(plan: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        plan, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_plan_source(plan: Mapping[str, Any]) -> None:
    source = plan["source"]
    checkpoint = _regular_file(
        Path(source["checkpoint_path"]), "signed-response source checkpoint"
    )
    manifest = _regular_file(
        Path(source["manifest_path"]), "signed-response source manifest"
    )
    if (
        _sha256_file(checkpoint) != source["checkpoint_sha256"]
        or _sha256_file(manifest) != source["manifest_sha256"]
    ):
        raise ValueError("signed-response source evidence was mutated")


def _window_evidence(values: np.ndarray, target: float, active: np.ndarray) -> dict[str, float]:
    return compute_window_metrics(values, target, active)


def build_axis_evidence(
    evidence_root: str | os.PathLike[str],
    plan_path: str | os.PathLike[str],
    axis: str,
) -> dict[str, Any]:
    """Independently rebuild one axis from the 33 raw trajectory bundles."""
    if axis not in BOUNDARIES:
        raise ValueError("axis must be forward or yaw")
    root = _root_directory(Path(evidence_root))
    plan_file = _regular_file(Path(plan_path), "signed-response plan")
    plan = _validated_plan(plan_file)
    _verify_plan_source(plan)
    root_manifest = _load_strict_json(root / "manifest.json", "raw grid manifest")
    root_result = _load_strict_json(root / "result.json", "raw grid result")
    plan_hash = _sha256_file(plan_file)
    if (
        root_manifest.get("kind") != "elf3_m5_c3_signed_response_raw"
        or root_manifest.get("plan")
        != {"path": str(plan_file), "sha256": plan_hash}
        or root_manifest.get("actions") != _canonical_actions()
        or root_result.get("status") != "PASS"
        or root_result.get("completed_actions")
        != [action["action_id"] for action in _canonical_actions()]
    ):
        raise ValueError("raw grid manifest or result does not bind the canonical plan")
    source = plan["source"]
    if root_manifest.get("source") != {
        "checkpoint_path": source["checkpoint_path"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "manifest_path": source["manifest_path"],
        "manifest_sha256": source["manifest_sha256"],
    }:
        raise ValueError("raw grid source identity differs from the signed plan")

    component = 0 if axis == "forward" else 2
    actions = _canonical_actions()
    by_id = {action["action_id"]: action for action in actions}
    windows = {"full": (0, STEPS), "post_initial": (100, STEPS)}
    output: dict[str, Any] = {
        "axis": axis,
        "seeds": list(SEEDS),
        "commands": COMMANDS.copy(),
        "windows": {},
    }
    per_window: dict[str, dict[str, list[list[float]]]] = {
        name: {"responses": [], "mae": []} for name in windows
    }
    for seed in SEEDS:
        baseline_id = f"seed{seed}_walk_zero"
        baseline_action = by_id[baseline_id]
        baseline_arrays, _ = _action_trajectory(root, baseline_action)
        for name, (start, stop) in windows.items():
            slice_active = baseline_arrays["active_before"][start:stop]
            baseline_values = (
                baseline_arrays["root_lin_vel_b"][start:stop, :, component]
                if axis == "forward"
                else baseline_arrays["root_ang_vel_b"][start:stop, :, component]
            )
            responses = [
                _window_evidence(baseline_values, 0.0, slice_active)["signed_mean"]
            ]
            errors: list[float] = []
            for value in (0.1, 0.2, 0.3, 0.4, 0.5):
                token = f"{value:.1f}".replace(".", "p")
                action_id = f"seed{seed}_{axis}_{token}"
                action = by_id[action_id]
                arrays, _ = _action_trajectory(root, action)
                active = arrays["active_before"][start:stop]
                measured = (
                    arrays["root_lin_vel_b"][start:stop, :, component]
                    if axis == "forward"
                    else arrays["root_ang_vel_b"][start:stop, :, component]
                )
                metric = _window_evidence(measured, value, active)
                responses.append(metric["signed_mean"])
                errors.append(metric["mae"])
            per_window[name]["responses"].append(responses)
            per_window[name]["mae"].append(errors)
    for name in windows:
        output["windows"][name] = {
            "responses": np.asarray(per_window[name]["responses"], dtype=np.float64),
            "mae": np.asarray(per_window[name]["mae"], dtype=np.float64),
        }
    return output


def _json_object(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON contains {value}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(nested) for nested in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", type=Path, help="forward aggregate evidence JSON")
    parser.add_argument("--yaw", type=Path, help="yaw aggregate evidence JSON")
    parser.add_argument("--evidence-root", type=Path, help="raw signed-response evidence root")
    parser.add_argument("--plan", type=Path, help="immutable signed-response plan")
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    aggregate_mode = args.forward is not None or args.yaw is not None
    raw_mode = args.evidence_root is not None or args.plan is not None
    if aggregate_mode == raw_mode:
        raise ValueError("select either aggregate JSON inputs or raw evidence inputs")
    if aggregate_mode:
        if args.forward is None or args.yaw is None:
            raise ValueError("both forward and yaw aggregate JSON inputs are required")
        axis_results = {
            "forward": classify_axis("forward", _json_object(args.forward)),
            "yaw": classify_axis("yaw", _json_object(args.yaw)),
        }
    else:
        assert args.evidence_root is not None and args.plan is not None
        plan = _validated_plan(_regular_file(args.plan, "signed-response plan"))
        if args.plan_sha256 != _canonical_plan_hash(plan):
            raise ValueError("explicit plan SHA-256 differs from the canonical plan")
        if args.checkpoint_sha256 != plan["source"]["checkpoint_sha256"]:
            raise ValueError("explicit checkpoint SHA-256 differs from the signed plan")
        axis_results = {
            axis: classify_axis(
                axis, build_axis_evidence(args.evidence_root, args.plan, axis)
            )
            for axis in ("forward", "yaw")
        }
    decision = classify_grid(
        axis_results,
        plan_sha256=args.plan_sha256,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    rendered = json.dumps(_json_ready(decision), sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    return 0 if decision["authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
