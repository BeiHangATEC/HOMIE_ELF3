#!/usr/bin/env python3
"""Create, collect, and independently check the ELF3 C3 signed-response grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT / "isaaclab_ext/source/openhomie_isaaclab"
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from openhomie_isaaclab.workflows.elf3_c3_signed_response import (  # noqa: E402
    FROZEN_C1_SHA256,
    canonical_grid_actions,
    plan_sha256,
    validate_plan,
)
from openhomie_isaaclab.workflows.elf3_run import (  # noqa: E402
    sha256_file,
    validate_manifest,
    write_json_once,
)


WORKER = REPO_ROOT / "isaaclab_ext/scripts/elf3_m5_c3_signed_response.py"
CHECKER = REPO_ROOT / "isaaclab_ext/scripts/check_elf3_m5_c3_signed_response.py"
DEVICE = "cuda:0"


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _exists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _regular_file(value: str | os.PathLike[str], name: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _new_directory(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if _exists(path):
        raise FileExistsError(f"evidence root already exists: {path}")
    parent = path.parent.resolve(strict=True)
    resolved = parent / path.name
    if not path.name or _exists(resolved):
        raise FileExistsError(f"evidence root already exists: {resolved}")
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


def _git_commit() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("cannot determine the repository Git commit")
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise RuntimeError("repository Git commit is malformed")
    return commit


def _verify_frozen_sources() -> None:
    for relative, expected in FROZEN_C1_SHA256.items():
        path = _regular_file(REPO_ROOT / relative, "frozen C1 source")
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen C1 source changed: {relative}")


def _build_plan(checkpoint: Path, manifest: Path) -> dict[str, Any]:
    checkpoint_hash = sha256_file(checkpoint)
    manifest_hash = sha256_file(manifest)
    payload = _load_json(manifest, "S0 source manifest")
    validate_manifest(payload)
    if (
        payload.get("command") != "train"
        or payload.get("start_iteration") != 2000
        or payload.get("iterations") != 2000
        or payload.get("configs", {}).get("env", {}).get("command_stage")
        != "S0"
    ):
        raise ValueError("source manifest is not the S0 2000-to-4000 continuation")
    plan = {
        "schema_version": 1,
        "kind": "elf3_m5_c3_signed_response",
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "git_commit": _git_commit(),
        "source": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_iteration": 4000,
            "stage": "S0",
            "manifest_path": str(manifest),
            "manifest_sha256": manifest_hash,
        },
        "seeds": [42, 43, 44],
        "num_envs": 16,
        "steps": 1000,
        "command_values": [0.1, 0.2, 0.3, 0.4, 0.5],
        "windows": {
            "full": [0, 1000],
            "first100": [0, 100],
            "post_initial": [100, 1000],
            "blocks": [[start, start + 100] for start in range(0, 1000, 100)],
        },
        "support_boundaries": {"forward": 0.3, "yaw": 0.2},
        "behavior_limits": {"forward": 0.20, "yaw": 0.25},
        "gain_limits": {
            "in_min": 0.70,
            "in_max": 1.30,
            "out_max": 0.50,
            "out_to_in_max": 0.60,
        },
        "required_arrays": {
            "step_index": {"dtype": "int64", "shape": [1000]},
            "command": {"dtype": "float32", "shape": [1000, 16, 4]},
            "mode": {"dtype": "int64", "shape": [1000, 16]},
            "root_lin_vel_b": {"dtype": "float32", "shape": [1000, 16, 3]},
            "root_ang_vel_b": {"dtype": "float32", "shape": [1000, 16, 3]},
            "roll_pitch": {"dtype": "float32", "shape": [1000, 16, 2]},
            "tracking_height": {"dtype": "float32", "shape": [1000, 16]},
            "action": {"dtype": "float32", "shape": [1000, 16, 12]},
            "reward": {"dtype": "float32", "shape": [1000, 16]},
            "active_before": {"dtype": "bool", "shape": [1000, 16]},
            "done": {"dtype": "bool", "shape": [1000, 16]},
            "timeout": {"dtype": "bool", "shape": [1000, 16]},
        },
        "actions": canonical_grid_actions(),
        "excluded_evidence_roots": [
            "/home/user/wang-sm/OpenHomie_m5diag_s0_inrange_43e31b1_20260817"
        ],
        "frozen_c1_sha256": dict(FROZEN_C1_SHA256),
    }
    validate_plan(plan)
    return plan


def parse_request(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--isaaclab-sh", required=True)
    parser.add_argument("--device", choices=(DEVICE,), default=DEVICE)
    args = parser.parse_args(argv)
    args.checkpoint = _regular_file(args.checkpoint, "checkpoint")
    if args.checkpoint.name != "model_4000.pt":
        raise ValueError("signed-response collection requires model_4000.pt")
    args.manifest = _regular_file(
        args.checkpoint.parent / "manifest.json", "source manifest"
    )
    args.evidence_root = _new_directory(args.evidence_root)
    args.isaaclab_sh = _regular_file(args.isaaclab_sh, "isaaclab.sh")
    source_run = args.checkpoint.parent.resolve(strict=True)
    if args.evidence_root.is_relative_to(source_run):
        raise ValueError("evidence root must be outside the S0 training run")
    return args


def _write_log(path: Path, completed: subprocess.CompletedProcess[str]) -> None:
    content = (completed.stdout or "")
    if completed.stderr:
        content += "\n" + completed.stderr
    if not content.endswith("\n"):
        content += "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _verify_frozen_sources()
    plan = _build_plan(args.checkpoint, args.manifest)
    args.evidence_root.mkdir(mode=0o755)
    plan_path = args.evidence_root / "plan.json"
    write_json_once(plan_path, plan)
    plan_hash = plan_sha256(plan)
    if sha256_file(plan_path) != hashlib.sha256(
        json.dumps(plan, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        + b"\n"
    ).hexdigest():
        raise RuntimeError("written signed-response plan is not immutable JSON")

    environment = os.environ.copy()
    inherited = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(PACKAGE_PARENT), inherited) if value
    )
    raw_root = args.evidence_root / "raw"
    worker_command = (
        str(args.isaaclab_sh),
        "-p",
        str(WORKER),
        "--plan",
        str(plan_path),
        "--output-root",
        str(raw_root),
        "--device",
        args.device,
    )
    completed = subprocess.run(
        worker_command,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    _write_log(args.evidence_root / "worker.log", completed)
    if completed.returncode != 0:
        return {
            "status": "FAIL",
            "failure_code": "C3_SIGNED_RESPONSE_WORKER_FAILED",
            "worker_returncode": completed.returncode,
            "plan_sha256": plan_hash,
        }

    grid_path = args.evidence_root / "grid_result.json"
    checker_command = (
        sys.executable,
        str(CHECKER),
        "--evidence-root",
        str(raw_root),
        "--plan",
        str(plan_path),
        "--plan-sha256",
        plan_hash,
        "--checkpoint-sha256",
        plan["source"]["checkpoint_sha256"],
        "--output",
        str(grid_path),
    )
    checked = subprocess.run(
        checker_command,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    _write_log(args.evidence_root / "checker.log", checked)
    if checked.returncode not in (0, 2) or not grid_path.is_file():
        return {
            "status": "FAIL",
            "failure_code": "C3_SIGNED_RESPONSE_CHECKER_FAILED",
            "checker_returncode": checked.returncode,
            "plan_sha256": plan_hash,
        }
    decision = _load_json(grid_path, "signed-response grid result")
    if decision.get("authorized") is True:
        authorization = {
            "schema_version": 1,
            "kind": "elf3_m5_c3_warm_start_authorization",
            "status": "WARM_START_AUTHORIZED",
            "plan_sha256": plan_hash,
            "checkpoint_sha256": plan["source"]["checkpoint_sha256"],
            "axis_status": decision.get("axis_status"),
        }
        write_json_once(args.evidence_root / "authorization.json", authorization)
    _verify_frozen_sources()
    return {
        "status": "PASS" if decision.get("authorized") is True else "STOPPED",
        "plan_sha256": plan_hash,
        "checkpoint_sha256": plan["source"]["checkpoint_sha256"],
        "grid_result": str(grid_path),
        "authorized": decision.get("authorized") is True,
        "axis_status": decision.get("axis_status"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_request(argv)
    result: Mapping[str, Any]
    try:
        result = run(args)
    except Exception as exc:
        result = {
            "status": "FAIL",
            "failure_code": "C3_SIGNED_RESPONSE_ORCHESTRATOR_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    write_json_once(args.evidence_root / "orchestration_result.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["status"] in {"PASS", "STOPPED"} else 1


if __name__ == "__main__":
    sys.exit(main())
