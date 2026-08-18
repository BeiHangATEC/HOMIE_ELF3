"""Pure-CPU contracts for the ELF3 M5 C3 signed-response evidence grid."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any

import numpy as np


SEEDS = (42, 43, 44)
COMMAND_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5)
STEPS = 1000
NUM_ENVS = 16
HEIGHT = 1.01
MODE = 0
OLD_PILOT_ROOT = "/home/user/wang-sm/OpenHomie_m5diag_s0_inrange_43e31b1_20260817"

FROZEN_C1_SHA256 = {
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/runner.py": (
        "daed23208a91a71efff1ffe7ecca7ea40623be0432f9bea74bc106b4bb31fbf4"
    ),
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/tasks/locomotion/"
    "elf3/elf3_homie_env.py": (
        "d7f54abb9b424e95d043df70ca350f32a61a43a7075ecc8859f2c87c7ed43342"
    ),
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/"
    "elf3_run.py": (
        "bde3dbe060b403befc752bbd3c450b1c3959bd48b3270be7af7b12794ef4dd28"
    ),
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/"
    "elf3_sim.py": (
        "c597d33c80580025d7fe28136262d75fa74301b9153dc548f2b2a71d2539dcf8"
    ),
}

REQUIRED_ARRAYS = {
    "step_index": {"dtype": "int64", "shape": [STEPS]},
    "command": {"dtype": "float32", "shape": [STEPS, NUM_ENVS, 4]},
    "mode": {"dtype": "int64", "shape": [STEPS, NUM_ENVS]},
    "root_lin_vel_b": {"dtype": "float32", "shape": [STEPS, NUM_ENVS, 3]},
    "root_ang_vel_b": {"dtype": "float32", "shape": [STEPS, NUM_ENVS, 3]},
    "roll_pitch": {"dtype": "float32", "shape": [STEPS, NUM_ENVS, 2]},
    "tracking_height": {"dtype": "float32", "shape": [STEPS, NUM_ENVS]},
    "action": {"dtype": "float32", "shape": [STEPS, NUM_ENVS, 12]},
    "reward": {"dtype": "float32", "shape": [STEPS, NUM_ENVS]},
    "active_before": {"dtype": "bool", "shape": [STEPS, NUM_ENVS]},
    "done": {"dtype": "bool", "shape": [STEPS, NUM_ENVS]},
    "timeout": {"dtype": "bool", "shape": [STEPS, NUM_ENVS]},
}

_TOP_LEVEL_KEYS = {
    "schema_version",
    "kind",
    "created_utc",
    "git_commit",
    "source",
    "seeds",
    "num_envs",
    "steps",
    "command_values",
    "windows",
    "support_boundaries",
    "behavior_limits",
    "gain_limits",
    "required_arrays",
    "actions",
    "excluded_evidence_roots",
    "frozen_c1_sha256",
}
_SOURCE_KEYS = {
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_iteration",
    "stage",
    "manifest_path",
    "manifest_sha256",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")


def _token(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def canonical_grid_actions() -> list[dict[str, Any]]:
    """Return the immutable 3-seed, 33-run signed-response inventory."""
    actions: list[dict[str, Any]] = []
    for seed in SEEDS:
        baseline = f"seed{seed}_walk_zero"
        actions.append(
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
        for scenario in ("forward", "yaw"):
            for value in COMMAND_VALUES:
                command = [0.0, 0.0, 0.0, HEIGHT]
                command[0 if scenario == "forward" else 2] = value
                actions.append(
                    {
                        "action_id": f"seed{seed}_{scenario}_{_token(value)}",
                        "seed": seed,
                        "scenario": scenario,
                        "command": command,
                        "mode": MODE,
                        "steps": STEPS,
                        "num_envs": NUM_ENVS,
                        "baseline_action_id": baseline,
                    }
                )
    return actions


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise KeyError(f"{name} keys differ; missing={missing}, extra={extra}")


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exact(value: Any, expected: Any, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} does not match the frozen C3 contract")


def validate_plan(plan: Any) -> None:
    """Validate a pre-registered plan and fail closed on any schema drift."""
    payload = _mapping(plan, "plan")
    _exact_keys(payload, _TOP_LEVEL_KEYS, "plan")
    _exact(payload["schema_version"], 1, "schema_version")
    _exact(payload["kind"], "elf3_m5_c3_signed_response", "kind")

    created_utc = payload["created_utc"]
    if not isinstance(created_utc, str) or _UTC_TIMESTAMP.fullmatch(created_utc) is None:
        raise ValueError("created_utc must be a second-resolution UTC timestamp")
    git_commit = payload["git_commit"]
    if not isinstance(git_commit, str) or _GIT_COMMIT.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a lowercase 40-character hash")

    source = _mapping(payload["source"], "source")
    _exact_keys(source, _SOURCE_KEYS, "source")
    _exact(source["checkpoint_iteration"], 4000, "checkpoint_iteration")
    _exact(source["stage"], "S0", "source stage")
    checkpoint_path = source["checkpoint_path"]
    if not isinstance(checkpoint_path, str) or PurePath(checkpoint_path).name != "model_4000.pt":
        raise ValueError("checkpoint_path must identify model_4000.pt")
    manifest_path = source["manifest_path"]
    if not isinstance(manifest_path, str) or PurePath(manifest_path).name != "manifest.json":
        raise ValueError("manifest_path must identify manifest.json")
    _sha256(source["checkpoint_sha256"], "checkpoint_sha256")
    _sha256(source["manifest_sha256"], "manifest_sha256")

    _exact(payload["seeds"], list(SEEDS), "seeds")
    _exact(payload["num_envs"], NUM_ENVS, "num_envs")
    _exact(payload["steps"], STEPS, "steps")
    _exact(payload["command_values"], list(COMMAND_VALUES), "command_values")
    _exact(
        payload["windows"],
        {
            "full": [0, 1000],
            "first100": [0, 100],
            "post_initial": [100, 1000],
            "blocks": [[start, start + 100] for start in range(0, 1000, 100)],
        },
        "windows",
    )
    _exact(payload["support_boundaries"], {"forward": 0.3, "yaw": 0.2}, "support_boundaries")
    _exact(payload["behavior_limits"], {"forward": 0.20, "yaw": 0.25}, "behavior_limits")
    _exact(
        payload["gain_limits"],
        {"in_min": 0.70, "in_max": 1.30, "out_max": 0.50, "out_to_in_max": 0.60},
        "gain_limits",
    )
    _exact(payload["required_arrays"], REQUIRED_ARRAYS, "required_arrays")
    _exact(payload["actions"], canonical_grid_actions(), "actions")
    _exact(payload["excluded_evidence_roots"], [OLD_PILOT_ROOT], "excluded_evidence_roots")
    _exact(payload["frozen_c1_sha256"], FROZEN_C1_SHA256, "frozen_c1_sha256")


def canonical_plan_bytes(plan: Any) -> bytes:
    """Validate and encode a plan using the one canonical JSON representation."""
    validate_plan(plan)
    return json.dumps(
        plan, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def plan_sha256(plan: Any) -> str:
    """Return the SHA-256 identity of a validated canonical plan."""
    return hashlib.sha256(canonical_plan_bytes(plan)).hexdigest()


def derive_active_before(done: np.ndarray, timeout: np.ndarray) -> np.ndarray:
    """Reconstruct the pre-action credit mask from post-action outcomes."""
    if not isinstance(done, np.ndarray) or not isinstance(timeout, np.ndarray):
        raise TypeError("done and timeout must be NumPy arrays")
    if done.dtype != np.bool_ or timeout.dtype != np.bool_:
        raise TypeError("done and timeout must have bool dtype")
    if done.ndim != 2 or timeout.shape != done.shape:
        raise ValueError("done and timeout must have the same two-dimensional shape")
    if np.any(timeout & ~done):
        raise ValueError("every timeout must also be marked done")

    active = np.ones(done.shape[1], dtype=np.bool_)
    result = np.empty_like(done)
    for step in range(done.shape[0]):
        result[step] = active
        active &= ~(done[step] & ~timeout[step])
    return result


def _validate_action(action: Any) -> Mapping[str, Any]:
    candidate = _mapping(action, "action")
    action_id = candidate.get("action_id")
    if not isinstance(action_id, str):
        raise TypeError("action_id must be a string")
    matches = [row for row in canonical_grid_actions() if row["action_id"] == action_id]
    if len(matches) != 1 or dict(candidate) != matches[0]:
        raise ValueError("action is not an exact member of the canonical grid")
    return candidate


def validate_trajectory_arrays(
    arrays: Any, action: Any
) -> dict[str, bool | int]:
    """Validate one raw trajectory bundle and return its credit summary."""
    action_row = _validate_action(action)
    payload = _mapping(arrays, "trajectory arrays")
    _exact_keys(payload, set(REQUIRED_ARRAYS), "trajectory arrays")

    for name, spec in REQUIRED_ARRAYS.items():
        value = payload[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
        if value.dtype != np.dtype(spec["dtype"]):
            raise TypeError(f"{name} dtype must be {spec['dtype']}")
        if list(value.shape) != spec["shape"]:
            raise ValueError(f"{name} shape must be {spec['shape']}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")

    if not np.array_equal(payload["step_index"], np.arange(STEPS, dtype=np.int64)):
        raise ValueError("step_index must be the exact range [0, 1000)")
    command = np.asarray(action_row["command"], dtype=np.float32)
    if not np.array_equal(payload["command"], np.broadcast_to(command, payload["command"].shape)):
        raise ValueError("command must remain fixed for every environment and step")
    if not np.all(payload["mode"] == action_row["mode"]):
        raise ValueError("mode must remain fixed for every environment and step")

    expected_active = derive_active_before(payload["done"], payload["timeout"])
    if not np.array_equal(payload["active_before"], expected_active):
        raise ValueError("active_before does not match done/timeout credit semantics")

    active = payload["active_before"]
    timeout = payload["timeout"] & active
    non_timeout = payload["done"] & ~payload["timeout"] & active
    return {
        "finite": True,
        "credited_env_steps": int(np.count_nonzero(active)),
        "timeout_count": int(np.count_nonzero(timeout)),
        "non_timeout_termination_count": int(np.count_nonzero(non_timeout)),
    }
