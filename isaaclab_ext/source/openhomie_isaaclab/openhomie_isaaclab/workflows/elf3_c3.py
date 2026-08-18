"""Fail-closed lifecycle for ELF3 C3 weights-only stage training."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from openhomie_isaaclab.workflows.elf3_run import (
    create_run_directory,
    sha256_file,
    validate_manifest as validate_m5_manifest,
    write_json_once,
)


TASK_ID = "OpenHomie-Elf3-Homie-Direct-v0"
WORKFLOW_ID = "elf3_c3_weights_only"
SCHEMA_VERSION = 1
STAGES = ("V1", "V2", "V3")
LOCAL_ITERATIONS = 2000
NUM_ENVS = 4096
NUM_STEPS_PER_ENV = 50
BASE_LEARNING_RATE = 0.001
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DEVICE = re.compile(r"(?:cpu|cuda:[0-9]+)\Z")

C3_SOURCE_PATHS = (
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/elf3_c3.py",
    "isaaclab_ext/scripts/elf3_c3.py",
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/tasks/locomotion/elf3/elf3_stages.py",
)

_STAGE_SPECS = {
    "V1": {
        "name": "V1",
        "walk_height": 1.01,
        "lin_vel_x": [-0.40, 0.60],
        "lin_vel_y": [-0.20, 0.20],
        "ang_vel_yaw": [-0.40, 0.40],
    },
    "V2": {
        "name": "V2",
        "walk_height": 1.01,
        "lin_vel_x": [-0.60, 0.90],
        "lin_vel_y": [-0.35, 0.35],
        "ang_vel_yaw": [-0.60, 0.60],
    },
    "V3": {
        "name": "V3",
        "walk_height": 1.01,
        "lin_vel_x": [-0.80, 1.20],
        "lin_vel_y": [-0.50, 0.50],
        "ang_vel_yaw": [-0.80, 0.80],
    },
}
STAGE_SPECS = MappingProxyType(_STAGE_SPECS)


@dataclass(frozen=True)
class StageTransition:
    target_stage: str
    parent_stage: str
    parent_checkpoint_iteration: int
    global_start: int
    global_final: int


_TRANSITIONS = {
    "V1": StageTransition("V1", "S0", 4000, 4000, 6000),
    "V2": StageTransition("V2", "V1", 2000, 6000, 8000),
    "V3": StageTransition("V3", "V2", 2000, 8000, 10000),
}
TRANSITIONS = MappingProxyType(_TRANSITIONS)


@dataclass(frozen=True)
class C3Request:
    stage: str
    run_dir: Path
    checkpoint: Path
    checkpoint_sha256: str
    parent_manifest: Path
    parent_manifest_sha256: str
    plan: Path
    plan_sha256: str
    device: str
    seed: int
    num_envs: int
    headless: bool

    @property
    def command(self) -> str:
        return "train"

    @property
    def iterations(self) -> int:
        return LOCAL_ITERATIONS

    @property
    def resume(self) -> bool:
        return False

    @property
    def scenario(self) -> None:
        return None

    @property
    def steps(self) -> None:
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "run_dir": str(self.run_dir),
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "parent_manifest": str(self.parent_manifest),
            "parent_manifest_sha256": self.parent_manifest_sha256,
            "plan": str(self.plan),
            "plan_sha256": self.plan_sha256,
            "device": self.device,
            "seed": self.seed,
            "num_envs": self.num_envs,
            "headless": self.headless,
            "command": self.command,
            "iterations": self.iterations,
            "resume": self.resume,
        }


def transition_for(stage: str) -> StageTransition:
    try:
        return TRANSITIONS[stage]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported C3 target stage: {stage}") from exc


def lifecycle_identity(stage: str) -> dict[str, Any]:
    transition = transition_for(stage)
    per_iteration = NUM_ENVS * NUM_STEPS_PER_ENV
    return {
        "stage": stage,
        "local_iterations": {"start": 0, "final": LOCAL_ITERATIONS},
        "global_iterations": {
            "start": transition.global_start,
            "final": transition.global_final,
        },
        "transitions": {
            "per_iteration": per_iteration,
            "stage": per_iteration * LOCAL_ITERATIONS,
            "global_start": per_iteration * transition.global_start,
            "global_final": per_iteration * transition.global_final,
        },
    }


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _canonical_num_envs(value: str) -> int:
    parsed = int(value)
    if parsed != NUM_ENVS:
        raise argparse.ArgumentTypeError(f"num-envs must be exactly {NUM_ENVS}")
    return parsed


def _device(value: str) -> str:
    if _DEVICE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("device must be cpu or cuda:<index>")
    return value


def _sha256_argument(value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "SHA-256 must be 64 lowercase hex characters"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one ELF3 C3 velocity stage from weights only"
    )
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-sha256", required=True, type=_sha256_argument
    )
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument(
        "--parent-manifest-sha256", required=True, type=_sha256_argument
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True, type=_sha256_argument)
    parser.add_argument("--device", required=True, type=_device)
    parser.add_argument("--seed", required=True, type=_nonnegative)
    parser.add_argument(
        "--num-envs", required=True, type=_canonical_num_envs
    )
    parser.add_argument("--headless", action="store_true")
    return parser


def _exists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _resolve_regular_file(
    value: str | os.PathLike[str], label: str
) -> Path:
    raw = os.fspath(value)
    candidate = Path(raw).expanduser()
    if candidate.name.casefold() == "latest" or any(
        char in raw for char in "*?[]"
    ):
        raise ValueError(f"{label} must be one explicit regular file")
    if not _exists(candidate) or candidate.is_symlink():
        raise ValueError(f"{label} must be an existing non-symlink file")
    try:
        mode = candidate.stat().st_mode
    except OSError as exc:
        raise ValueError(f"{label} cannot be inspected") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular file")
    return candidate.resolve(strict=True)


def _resolve_new_directory(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if _exists(candidate):
        raise FileExistsError(f"run path already exists: {candidate}")
    resolved = candidate.parent.resolve(strict=True) / candidate.name
    if _exists(resolved):
        raise FileExistsError(f"run path aliases an existing path: {resolved}")
    return resolved


def _verify_hash(path: Path, expected: str, label: str) -> str:
    if _SHA256.fullmatch(expected) is None:
        raise ValueError(f"{label} expected SHA-256 is malformed")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 does not match")
    return actual


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def parse_request(argv: Sequence[str] | None = None) -> C3Request:
    args = build_parser().parse_args(argv)
    checkpoint = _resolve_regular_file(args.checkpoint, "checkpoint")
    parent_manifest = _resolve_regular_file(
        args.parent_manifest, "parent manifest"
    )
    plan = _resolve_regular_file(args.plan, "approved plan")
    if parent_manifest != checkpoint.parent / "manifest.json":
        raise ValueError(
            "parent manifest must be manifest.json beside the checkpoint"
        )
    transition = transition_for(args.stage)
    expected_name = f"model_{transition.parent_checkpoint_iteration}.pt"
    if checkpoint.name != expected_name:
        raise ValueError(f"parent checkpoint must be named {expected_name}")
    _verify_hash(checkpoint, args.checkpoint_sha256, "checkpoint")
    _verify_hash(
        parent_manifest, args.parent_manifest_sha256, "parent manifest"
    )
    _verify_hash(plan, args.plan_sha256, "approved plan")
    payload = _read_json(parent_manifest, "parent manifest")
    identity = parent_stage_identity(
        payload, target_stage=args.stage, checkpoint_iteration=None
    )
    if identity["kind"] == "c3":
        recorded_plan = payload["plan"]
        if (
            recorded_plan.get("sha256") != args.plan_sha256
            or Path(recorded_plan.get("path", "")) != plan
        ):
            raise ValueError("C3 parent was trained under a different plan")
    return C3Request(
        stage=args.stage,
        run_dir=_resolve_new_directory(args.run_dir),
        checkpoint=checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        parent_manifest=parent_manifest,
        parent_manifest_sha256=args.parent_manifest_sha256,
        plan=plan,
        plan_sha256=args.plan_sha256,
        device=args.device,
        seed=args.seed,
        num_envs=args.num_envs,
        headless=args.headless,
    )


def _require_exact_keys(
    value: Any, expected: set[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise KeyError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _json_safe(value: Any, name: str = "value") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} is non-finite")
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{name} keys must be strings")
        for key, nested in value.items():
            _json_safe(nested, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _json_safe(nested, f"{name}[{index}]")
        return
    raise TypeError(f"{name} is not JSON-safe")


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a path string")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path


def _manifest_load_contract() -> dict[str, Any]:
    return {
        "mode": "weights_only",
        "load_optimizer": False,
        "local_iteration_reset": 0,
        "optimizer_state_entries": {"policy": 0, "estimator": 0},
        "learning_rates": {
            "policy": BASE_LEARNING_RATE,
            "estimator": BASE_LEARNING_RATE,
        },
    }


def _validate_parent_record(parent: Any, stage: str) -> None:
    parent = _require_exact_keys(
        parent,
        {
            "kind",
            "stage",
            "local_iteration",
            "global_iteration",
            "checkpoint_path",
            "checkpoint_sha256",
            "manifest_path",
            "manifest_sha256",
        },
        "parent",
    )
    transition = transition_for(stage)
    expected_kind = "m5_s0_bootstrap" if stage == "V1" else "c3"
    expected_local = 4000 if stage == "V1" else LOCAL_ITERATIONS
    if (
        parent["kind"] != expected_kind
        or parent["stage"] != transition.parent_stage
        or parent["local_iteration"] != expected_local
        or parent["global_iteration"] != transition.global_start
    ):
        raise ValueError("parent lifecycle identity is inconsistent")
    _require_path(parent["checkpoint_path"], "parent checkpoint_path")
    _require_path(parent["manifest_path"], "parent manifest_path")
    _require_sha(parent["checkpoint_sha256"], "parent checkpoint_sha256")
    _require_sha(parent["manifest_sha256"], "parent manifest_sha256")


def validate_c3_manifest(payload: Any) -> None:
    payload = _require_exact_keys(
        payload,
        {
            "schema_version",
            "workflow",
            "created_utc",
            "task_id",
            "stage",
            "seed",
            "device",
            "num_envs",
            "iterations",
            "cli",
            "plan",
            "parent",
            "stage_spec",
            "lifecycle",
            "load_contract",
            "git",
            "configs",
            "assets",
            "m4_sources",
            "c3_sources",
            "runtime",
            "gpu",
        },
        "C3 manifest",
    )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported C3 manifest schema")
    if (
        payload["workflow"] != WORKFLOW_ID
        or payload["task_id"] != TASK_ID
    ):
        raise ValueError("invalid C3 manifest identity")
    stage = payload["stage"]
    transition_for(stage)
    if (
        not isinstance(payload["created_utc"], str)
        or not payload["created_utc"].endswith("Z")
    ):
        raise ValueError("created_utc must be a UTC string")
    _integer(payload["seed"], "seed")
    if (
        not isinstance(payload["device"], str)
        or _DEVICE.fullmatch(payload["device"]) is None
    ):
        raise ValueError("invalid C3 device")
    if (
        payload["num_envs"] != NUM_ENVS
        or payload["iterations"] != LOCAL_ITERATIONS
    ):
        raise ValueError("C3 manifest has a noncanonical training budget")
    if payload["lifecycle"] != lifecycle_identity(stage):
        raise ValueError("C3 lifecycle identity is inconsistent")
    if payload["load_contract"] != _manifest_load_contract():
        raise ValueError("C3 weights-only load contract is inconsistent")
    if payload["stage_spec"] != STAGE_SPECS[stage]:
        raise ValueError("C3 stage specification is inconsistent")

    plan = _require_exact_keys(
        payload["plan"], {"path", "sha256"}, "plan"
    )
    _require_path(plan["path"], "plan.path")
    _require_sha(plan["sha256"], "plan.sha256")
    _validate_parent_record(payload["parent"], stage)

    cli = _require_exact_keys(
        payload["cli"],
        {
            "stage",
            "run_dir",
            "checkpoint",
            "checkpoint_sha256",
            "parent_manifest",
            "parent_manifest_sha256",
            "plan",
            "plan_sha256",
            "device",
            "seed",
            "num_envs",
            "headless",
            "command",
            "iterations",
            "resume",
        },
        "cli",
    )
    if (
        cli["stage"] != stage
        or cli["device"] != payload["device"]
        or cli["seed"] != payload["seed"]
        or cli["num_envs"] != NUM_ENVS
        or cli["iterations"] != LOCAL_ITERATIONS
        or cli["command"] != "train"
        or cli["resume"] is not False
        or cli["checkpoint_sha256"]
        != payload["parent"]["checkpoint_sha256"]
        or cli["parent_manifest_sha256"]
        != payload["parent"]["manifest_sha256"]
        or cli["plan_sha256"] != plan["sha256"]
    ):
        raise ValueError("C3 CLI identity is inconsistent")
    for name in ("run_dir", "checkpoint", "parent_manifest", "plan"):
        _require_path(cli[name], f"cli.{name}")
    if cli["checkpoint"] != payload["parent"]["checkpoint_path"]:
        raise ValueError("CLI checkpoint path differs from parent")
    if cli["parent_manifest"] != payload["parent"]["manifest_path"]:
        raise ValueError("CLI manifest path differs from parent")
    if cli["plan"] != plan["path"]:
        raise ValueError("CLI plan path differs from manifest")

    for name in (
        "git",
        "configs",
        "assets",
        "m4_sources",
        "c3_sources",
        "runtime",
        "gpu",
    ):
        if not isinstance(payload[name], Mapping):
            raise TypeError(f"{name} must be a mapping")
    c3_sources = _require_exact_keys(
        payload["c3_sources"], {"files", "sha256"}, "c3_sources"
    )
    if not isinstance(c3_sources["files"], Mapping):
        raise TypeError("c3_sources.files must be a mapping")
    for path, digest in c3_sources["files"].items():
        if not isinstance(path, str) or Path(path).is_absolute():
            raise ValueError(
                "C3 source paths must be repository-relative"
            )
        _require_sha(digest, f"c3_sources.files.{path}")
    _require_sha(c3_sources["sha256"], "c3_sources.sha256")
    configs = payload["configs"]
    if configs.get("env", {}).get("command_stage") != stage:
        raise ValueError("resolved environment config has the wrong stage")
    _json_safe(payload, "C3 manifest")


def parent_stage_identity(
    payload: Any,
    *,
    target_stage: str,
    checkpoint_iteration: int | None,
) -> dict[str, Any]:
    transition = transition_for(target_stage)
    if checkpoint_iteration is not None:
        _integer(checkpoint_iteration, "checkpoint iteration")
        if checkpoint_iteration != transition.parent_checkpoint_iteration:
            raise ValueError("parent checkpoint iteration is inconsistent")
    if target_stage == "V1":
        validate_m5_manifest(payload)
        configs = payload.get("configs")
        if (
            payload.get("command") != "train"
            or payload.get("start_iteration") != 2000
            or payload.get("iterations") != 2000
            or not isinstance(configs, Mapping)
            or not isinstance(configs.get("env"), Mapping)
            or configs["env"].get("command_stage") != "S0"
        ):
            raise ValueError(
                "V1 requires the exact S0 local/global-4000 bootstrap"
            )
        return {
            "kind": "m5_s0_bootstrap",
            "stage": "S0",
            "local_iteration": 4000,
            "global_iteration": 4000,
        }

    validate_c3_manifest(payload)
    expected_parent = transition.parent_stage
    if payload["stage"] != expected_parent:
        raise ValueError("C3 parent stage is out of order")
    local_final = payload["lifecycle"]["local_iterations"]["final"]
    global_final = payload["lifecycle"]["global_iterations"]["final"]
    if (
        local_final != LOCAL_ITERATIONS
        or global_final != transition.global_start
    ):
        raise ValueError("C3 parent local/global identity is inconsistent")
    return {
        "kind": "c3",
        "stage": expected_parent,
        "local_iteration": local_final,
        "global_iteration": global_final,
    }


def build_manifest(
    request: C3Request,
    *,
    parent: Mapping[str, Any],
    provenance: Mapping[str, Any],
    stage_spec: Mapping[str, Any],
    created_utc: str | None = None,
) -> dict[str, Any]:
    required_provenance = {
        "git",
        "configs",
        "assets",
        "m4_sources",
        "c3_sources",
        "runtime",
        "gpu",
    }
    _require_exact_keys(
        provenance, required_provenance, "provenance"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW_ID,
        "created_utc": created_utc
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "task_id": TASK_ID,
        "stage": request.stage,
        "seed": request.seed,
        "device": request.device,
        "num_envs": request.num_envs,
        "iterations": request.iterations,
        "cli": request.to_dict(),
        "plan": {
            "path": str(request.plan),
            "sha256": request.plan_sha256,
        },
        "parent": dict(parent),
        "stage_spec": dict(stage_spec),
        "lifecycle": lifecycle_identity(request.stage),
        "load_contract": _manifest_load_contract(),
        **{
            name: provenance[name]
            for name in sorted(required_provenance)
        },
    }
    validate_c3_manifest(payload)
    return payload


def _optimizer_state_size(optimizer: Any, name: str) -> int:
    state = getattr(optimizer, "state", None)
    if not isinstance(state, Mapping):
        raise RuntimeError(f"{name} optimizer state is not a mapping")
    return len(state)


def _optimizer_group_lr(optimizer: Any, name: str) -> float:
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        raise RuntimeError(f"{name} optimizer has no parameter groups")
    rates = [
        _number(group.get("lr"), f"{name} optimizer LR")
        for group in groups
    ]
    if any(rate != BASE_LEARNING_RATE for rate in rates):
        raise RuntimeError(
            f"{name} optimizer LR is not the configured base LR"
        )
    return rates[0]


def _assert_fresh_runner(runner: Any) -> None:
    if runner.current_learning_iteration != 0:
        raise RuntimeError(
            "weights-only training requires a fresh runner"
        )
    policy_entries = _optimizer_state_size(
        runner.alg.optimizer, "policy"
    )
    estimator_entries = _optimizer_state_size(
        runner.alg.estimator_optimizer, "estimator"
    )
    if policy_entries or estimator_entries:
        raise RuntimeError("fresh runner optimizer states must be empty")
    policy_lr = _number(
        runner.alg.learning_rate, "policy learning rate"
    )
    estimator_lr = _number(
        runner.alg.estimator_learning_rate,
        "estimator learning rate",
    )
    if (
        policy_lr != BASE_LEARNING_RATE
        or estimator_lr != BASE_LEARNING_RATE
    ):
        raise RuntimeError(
            "fresh runner learning rates must equal 0.001"
        )
    _optimizer_group_lr(runner.alg.optimizer, "policy")
    _optimizer_group_lr(
        runner.alg.estimator_optimizer, "estimator"
    )


def initialize_weights_only(
    runner: Any,
    *,
    checkpoint: Path,
    device: str,
    expected_checkpoint_iteration: int,
) -> dict[str, Any]:
    _assert_fresh_runner(runner)
    runner.load(
        str(checkpoint),
        load_optimizer=False,
        map_location=device,
    )
    loaded_iteration = runner.current_learning_iteration
    runner.current_learning_iteration = 0
    if loaded_iteration != expected_checkpoint_iteration:
        raise RuntimeError(
            "loaded checkpoint iteration is inconsistent"
        )
    _assert_fresh_runner(runner)
    return {
        "mode": "weights_only",
        "load_optimizer": False,
        "checkpoint_iteration": loaded_iteration,
        "local_iteration_after_load": (
            runner.current_learning_iteration
        ),
        "optimizer_state_entries": {
            "policy": 0,
            "estimator": 0,
        },
        "learning_rates": {
            "policy": BASE_LEARNING_RATE,
            "estimator": BASE_LEARNING_RATE,
        },
    }


def _validate_actual_load_contract(
    contract: Any, manifest: Mapping[str, Any]
) -> None:
    contract = _require_exact_keys(
        contract,
        {
            "mode",
            "load_optimizer",
            "checkpoint_iteration",
            "local_iteration_after_load",
            "optimizer_state_entries",
            "learning_rates",
        },
        "actual load contract",
    )
    expected_iteration = (
        4000 if manifest["stage"] == "V1" else LOCAL_ITERATIONS
    )
    expected = {
        "mode": "weights_only",
        "load_optimizer": False,
        "checkpoint_iteration": expected_iteration,
        "local_iteration_after_load": 0,
        "optimizer_state_entries": {
            "policy": 0,
            "estimator": 0,
        },
        "learning_rates": {
            "policy": BASE_LEARNING_RATE,
            "estimator": BASE_LEARNING_RATE,
        },
    }
    if dict(contract) != expected:
        raise ValueError(
            "actual weights-only load contract is inconsistent"
        )


def _validate_parent_verification(
    verification: Any, manifest: Mapping[str, Any]
) -> None:
    verification = _require_exact_keys(
        verification,
        {
            "checkpoint_sha256_before",
            "checkpoint_sha256_after",
            "manifest_sha256_before",
            "manifest_sha256_after",
            "plan_sha256_before",
            "plan_sha256_after",
        },
        "parent verification",
    )
    parent = manifest["parent"]
    plan = manifest["plan"]
    expected = {
        "checkpoint_sha256_before": parent["checkpoint_sha256"],
        "checkpoint_sha256_after": parent["checkpoint_sha256"],
        "manifest_sha256_before": parent["manifest_sha256"],
        "manifest_sha256_after": parent["manifest_sha256"],
        "plan_sha256_before": plan["sha256"],
        "plan_sha256_after": plan["sha256"],
    }
    if dict(verification) != expected:
        raise ValueError("parent or plan identity changed")


def build_pass_result(
    request: C3Request,
    *,
    manifest: Mapping[str, Any],
    load_contract: Mapping[str, Any],
    metrics: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    parent_verification: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    validate_c3_manifest(manifest)
    checkpoint = _require_exact_keys(
        checkpoint,
        {"path", "sha256", "iteration"},
        "checkpoint",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW_ID,
        "status": "PASS",
        "stage": request.stage,
        "lifecycle": manifest["lifecycle"],
        "load_contract": dict(load_contract),
        "metrics": dict(metrics),
        "checkpoint": {
            "path": checkpoint["path"],
            "sha256": checkpoint["sha256"],
            "iteration": checkpoint["iteration"],
            "stage": request.stage,
            "local_iteration": LOCAL_ITERATIONS,
            "global_iteration": transition_for(
                request.stage
            ).global_final,
        },
        "parent_verification": dict(parent_verification),
        "elapsed_seconds": float(elapsed_seconds),
    }
    validate_c3_result(payload, manifest)
    return payload


def build_failure_result(
    request: C3Request,
    *,
    manifest: Mapping[str, Any],
    failure: Exception,
    elapsed_seconds: float,
) -> dict[str, Any]:
    validate_c3_manifest(manifest)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW_ID,
        "status": "FAIL",
        "stage": request.stage,
        "lifecycle": manifest["lifecycle"],
        "failure_code": "C3_RUNTIME_FAILURE",
        "error_type": type(failure).__name__,
        "error": str(failure),
        "elapsed_seconds": float(elapsed_seconds),
    }
    validate_c3_result(payload, manifest)
    return payload


def validate_c3_result(
    payload: Any, manifest: Mapping[str, Any]
) -> None:
    validate_c3_manifest(manifest)
    if not isinstance(payload, Mapping):
        raise TypeError("C3 result must be a mapping")
    status = payload.get("status")
    common = {
        "schema_version",
        "workflow",
        "status",
        "stage",
        "lifecycle",
        "elapsed_seconds",
    }
    if status == "PASS":
        expected = common | {
            "load_contract",
            "metrics",
            "checkpoint",
            "parent_verification",
        }
    elif status == "FAIL":
        expected = common | {
            "failure_code",
            "error_type",
            "error",
        }
    else:
        raise ValueError("C3 result status must be PASS or FAIL")
    _require_exact_keys(payload, expected, "C3 result")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["workflow"] != WORKFLOW_ID
        or payload["stage"] != manifest["stage"]
        or payload["lifecycle"] != manifest["lifecycle"]
    ):
        raise ValueError(
            "C3 result identity differs from its manifest"
        )
    if _number(
        payload["elapsed_seconds"], "elapsed_seconds"
    ) < 0:
        raise ValueError("elapsed_seconds must be nonnegative")
    if status == "PASS":
        _validate_actual_load_contract(
            payload["load_contract"], manifest
        )
        if not isinstance(payload["metrics"], Mapping):
            raise TypeError("metrics must be a mapping")
        checkpoint = _require_exact_keys(
            payload["checkpoint"],
            {
                "path",
                "sha256",
                "iteration",
                "stage",
                "local_iteration",
                "global_iteration",
            },
            "result checkpoint",
        )
        transition = transition_for(manifest["stage"])
        if (
            checkpoint["stage"] != manifest["stage"]
            or checkpoint["iteration"] != LOCAL_ITERATIONS
            or checkpoint["local_iteration"] != LOCAL_ITERATIONS
            or checkpoint["global_iteration"]
            != transition.global_final
        ):
            raise ValueError(
                "result checkpoint lifecycle identity is inconsistent"
            )
        _require_path(
            checkpoint["path"], "result checkpoint path"
        )
        _require_sha(
            checkpoint["sha256"], "result checkpoint sha256"
        )
        _validate_parent_verification(
            payload["parent_verification"], manifest
        )
    else:
        if (
            payload["failure_code"] != "C3_RUNTIME_FAILURE"
            or not isinstance(payload["error_type"], str)
            or not isinstance(payload["error"], str)
        ):
            raise ValueError("C3 failure result is malformed")
    _json_safe(payload, "C3 result")


def write_immutable_json(
    path: str | os.PathLike[str], payload: Any
) -> Path:
    return write_json_once(path, payload)


def _verify_bound_inputs(
    request: C3Request,
    *,
    checkpoint_iteration: int | None,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    checkpoint = _resolve_regular_file(
        request.checkpoint, "checkpoint"
    )
    parent_manifest = _resolve_regular_file(
        request.parent_manifest, "parent manifest"
    )
    plan = _resolve_regular_file(request.plan, "approved plan")
    if (
        checkpoint != request.checkpoint
        or parent_manifest != request.parent_manifest
        or plan != request.plan
    ):
        raise RuntimeError("bound input path identity changed")
    _verify_hash(
        checkpoint, request.checkpoint_sha256, "checkpoint"
    )
    _verify_hash(
        parent_manifest,
        request.parent_manifest_sha256,
        "parent manifest",
    )
    _verify_hash(plan, request.plan_sha256, "approved plan")
    payload = _read_json(parent_manifest, "parent manifest")
    identity = parent_stage_identity(
        payload,
        target_stage=request.stage,
        checkpoint_iteration=checkpoint_iteration,
    )
    if identity["kind"] == "c3":
        recorded_plan = payload["plan"]
        if (
            recorded_plan["path"] != str(plan)
            or recorded_plan["sha256"] != request.plan_sha256
        ):
            raise ValueError(
                "C3 parent was trained under a different plan"
            )
    return identity, payload


def _c3_source_identity(
    repo_root: Path, simulator: Any
) -> dict[str, Any]:
    files: dict[str, str] = {}
    for relative in C3_SOURCE_PATHS:
        path = repo_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"C3 source is not a regular file: {relative}"
            )
        files[relative] = sha256_file(path)
    return {
        "files": files,
        "sha256": simulator._json_sha256(files),
    }


def _load_stage_configs(
    request: C3Request, simulator: Any, stages: Any
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    env_cfg, agent_cfg, _ = simulator._load_configs(request)
    env_cfg.command_stage = request.stage
    stage_spec = stages.get_stage(request.stage).to_dict()
    if stage_spec != STAGE_SPECS[request.stage]:
        raise RuntimeError(
            "authoritative ELF3 stage definition changed"
        )
    env_mapping = simulator._normalize_json(env_cfg.to_dict())
    agent_mapping = simulator._normalize_json(agent_cfg.to_dict())
    if (
        env_mapping.get("command_stage") != request.stage
        or agent_mapping.get("max_iterations")
        != LOCAL_ITERATIONS
        or agent_mapping.get("num_steps_per_env")
        != NUM_STEPS_PER_ENV
    ):
        raise RuntimeError("resolved C3 configuration is inconsistent")
    algorithm = agent_mapping.get("algorithm", {})
    if (
        algorithm.get("learning_rate") != BASE_LEARNING_RATE
        or algorithm.get("estimator_learning_rate")
        != BASE_LEARNING_RATE
    ):
        raise RuntimeError(
            "resolved base learning rates must both equal 0.001"
        )
    configs = {
        "env": env_mapping,
        "agent": agent_mapping,
        "sha256": {
            "env": simulator._json_sha256(env_mapping),
            "agent": simulator._json_sha256(agent_mapping),
        },
    }
    return env_cfg, agent_cfg, configs, stage_spec


def _parent_record(
    request: C3Request, identity: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **dict(identity),
        "checkpoint_path": str(request.checkpoint),
        "checkpoint_sha256": request.checkpoint_sha256,
        "manifest_path": str(request.parent_manifest),
        "manifest_sha256": request.parent_manifest_sha256,
    }


def _parent_verification(request: C3Request) -> dict[str, str]:
    _verify_bound_inputs(request, checkpoint_iteration=None)
    checkpoint_after = sha256_file(request.checkpoint)
    manifest_after = sha256_file(request.parent_manifest)
    plan_after = sha256_file(request.plan)
    return {
        "checkpoint_sha256_before": request.checkpoint_sha256,
        "checkpoint_sha256_after": checkpoint_after,
        "manifest_sha256_before": request.parent_manifest_sha256,
        "manifest_sha256_after": manifest_after,
        "plan_sha256_before": request.plan_sha256,
        "plan_sha256_after": plan_after,
    }


def _record_failure(
    current: Exception | None, additional: Exception
) -> Exception:
    if current is None:
        return additional
    return RuntimeError(
        f"{current}; additional failure: {additional}"
    )


def run(request: C3Request) -> dict[str, Any]:
    """Run one post-AppLauncher C3 stage and write immutable evidence."""
    if not isinstance(request, C3Request):
        raise TypeError("C3 run requires a parsed C3Request")
    transition = transition_for(request.stage)
    started = time.perf_counter()

    # These imports are intentionally delayed until AppLauncher is live.
    from openhomie_isaaclab.tasks.locomotion.elf3 import elf3_stages
    from openhomie_isaaclab.workflows import elf3_sim

    _verify_bound_inputs(request, checkpoint_iteration=None)
    checkpoint_iteration, checkpoint_payload = (
        elf3_sim._load_checkpoint(
            request.checkpoint, require_optimizers=False
        )
    )
    del checkpoint_payload
    parent_identity, _ = _verify_bound_inputs(
        request, checkpoint_iteration=checkpoint_iteration
    )

    repo_root = elf3_sim._repository_root()
    git = elf3_sim._git_identity(repo_root)
    elf3_sim._seed_runtime(request.seed)
    env_cfg, agent_cfg, configs, stage_spec = (
        _load_stage_configs(
            request, elf3_sim, elf3_stages
        )
    )
    assets, m4_sources = elf3_sim._source_identity(repo_root)
    c3_sources = _c3_source_identity(repo_root, elf3_sim)
    runtime = elf3_sim._runtime_identity()
    gpu = elf3_sim._gpu_identity(request.device)
    provenance = {
        "git": git,
        "configs": configs,
        "assets": assets,
        "m4_sources": m4_sources,
        "c3_sources": c3_sources,
        "runtime": runtime,
        "gpu": gpu,
    }

    run_dir = create_run_directory(request.run_dir)
    if run_dir != request.run_dir:
        raise RuntimeError("run directory identity changed")
    manifest = build_manifest(
        request,
        parent=_parent_record(request, parent_identity),
        provenance=provenance,
        stage_spec=stage_spec,
    )
    write_immutable_json(run_dir / "manifest.json", manifest)

    env = None
    runner = None
    load_contract: dict[str, Any] | None = None
    training: dict[str, Any] | None = None
    verification: dict[str, str] | None = None
    failure: Exception | None = None
    try:
        env, runner = elf3_sim._create_runner(
            request, env_cfg, agent_cfg
        )
        if (
            runner.num_steps_per_env != NUM_STEPS_PER_ENV
            or env.num_envs != NUM_ENVS
        ):
            raise RuntimeError(
                "live runner has a noncanonical transition budget"
            )
        load_contract = initialize_weights_only(
            runner,
            checkpoint=request.checkpoint,
            device=request.device,
            expected_checkpoint_iteration=(
                transition.parent_checkpoint_iteration
            ),
        )
        training = elf3_sim._training_result(
            request, runner, 0
        )
        if runner.current_learning_iteration != LOCAL_ITERATIONS:
            raise RuntimeError(
                "C3 local training did not finish at iteration 2000"
            )
    except Exception as exc:
        failure = exc
    finally:
        if runner is not None and runner.writer is not None:
            try:
                runner.writer.flush()
                runner.writer.close()
            except Exception as exc:
                failure = _record_failure(failure, exc)
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                failure = _record_failure(failure, exc)

    try:
        verification = _parent_verification(request)
    except Exception as exc:
        failure = _record_failure(failure, exc)

    elapsed = float(time.perf_counter() - started)
    result: dict[str, Any]
    if failure is None:
        try:
            if (
                load_contract is None
                or training is None
                or verification is None
            ):
                raise RuntimeError(
                    "C3 training did not produce complete evidence"
                )
            metrics = {
                "training": training["metrics"],
                "final_learning_rates": training[
                    "learning_rates"
                ],
                "finite": training["finite"],
                "transitions": manifest["lifecycle"][
                    "transitions"
                ],
            }
            result = build_pass_result(
                request,
                manifest=manifest,
                load_contract=load_contract,
                metrics=metrics,
                checkpoint=training["checkpoint"],
                parent_verification=verification,
                elapsed_seconds=elapsed,
            )
        except Exception as exc:
            failure = exc
    if failure is not None:
        result = build_failure_result(
            request,
            manifest=manifest,
            failure=failure,
            elapsed_seconds=elapsed,
        )
    validate_c3_result(result, manifest)
    write_immutable_json(run_dir / "result.json", result)
    return result
