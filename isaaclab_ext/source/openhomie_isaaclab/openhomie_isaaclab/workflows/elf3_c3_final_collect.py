"""Collect final ELF3 C3 V3 behavior and exact-export evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openhomie_isaaclab.workflows.elf3_c3 import (
    LOCAL_ITERATIONS,
    NUM_ENVS as TRAIN_NUM_ENVS,
    STAGE_SPECS,
    lifecycle_identity,
    validate_c3_manifest,
    validate_c3_result,
)
from openhomie_isaaclab.workflows.elf3_c3_final import (
    APPROVED_PLAN_SHA256,
    C3FinalRequest,
    FINAL_GLOBAL_ITERATION,
    FINAL_LOCAL_ITERATION,
    FINAL_STAGE,
    SCENARIOS,
    SEEDS,
    write_or_verify_aggregate,
)
from openhomie_isaaclab.workflows.elf3_run import (
    EVALUATION_NUM_ENVS,
    EVALUATION_STEPS,
    RunRequest,
    create_run_directory,
    sha256_file,
    write_json_once,
)


DEVICE = "cuda:0"
EXPORT_DIRECTORY = "exact_export"
SCENARIOS_DIRECTORY = "scenarios"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


@dataclass(frozen=True)
class C3FinalCollectRequest:
    evidence_root: Path
    checkpoint: Path
    source_manifest: Path
    source_result: Path
    plan: Path
    plan_sha256: str
    device: str = DEVICE
    headless: bool = True


def _exists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _regular_file(value: str | os.PathLike[str], name: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    try:
        mode = candidate.stat().st_mode
    except OSError as exc:
        raise ValueError(f"{name} cannot be inspected") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    resolved = candidate.resolve(strict=True)
    if candidate != resolved:
        raise ValueError(f"{name} path must be canonical")
    return resolved


def _new_directory(value: str | os.PathLike[str], name: str) -> Path:
    candidate = Path(value).expanduser()
    if _exists(candidate):
        raise FileExistsError(f"{name} already exists: {candidate}")
    parent = candidate.parent.resolve(strict=True)
    resolved = parent / candidate.name
    if not candidate.name or _exists(resolved):
        raise FileExistsError(f"{name} already exists: {resolved}")
    return resolved


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    source = _regular_file(path, name)

    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains {value}")

    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return payload


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _absolute_exact(value: Any, expected: Path, name: str) -> None:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    candidate = Path(value)
    if candidate.is_symlink() or candidate.resolve(strict=True) != expected:
        raise ValueError(f"{name} does not identify the bound artifact")


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="Collect final V3 behavior and exact-export evidence"
    )
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-result", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--device", choices=(DEVICE,), default=DEVICE)
    return parser


def parse_request(
    argv: Sequence[str] | None = None,
) -> C3FinalCollectRequest:
    args = build_parser().parse_args(argv)
    checkpoint = _regular_file(args.checkpoint, "checkpoint")
    source_manifest = _regular_file(args.source_manifest, "source manifest")
    source_result = _regular_file(args.source_result, "source result")
    plan = _regular_file(args.plan, "approved plan")
    plan_sha256 = _require_hash(args.plan_sha256, "approved plan SHA-256")
    if plan_sha256 != APPROVED_PLAN_SHA256:
        raise ValueError("plan SHA-256 is not the approved C3 design")
    if sha256_file(plan) != plan_sha256:
        raise ValueError("approved plan SHA-256 does not match the file")
    if (
        checkpoint.parent != source_manifest.parent
        or checkpoint.parent != source_result.parent
        or checkpoint.name != "model_2000.pt"
        or source_manifest.name != "manifest.json"
        or source_result.name != "result.json"
    ):
        raise ValueError(
            "checkpoint, manifest, and result must be canonical siblings in one V3 run"
        )
    evidence_root = _new_directory(args.evidence_root, "evidence root")
    source_run = checkpoint.parent
    if evidence_root.is_relative_to(source_run):
        raise ValueError("evidence root must be outside the C3 V3 run")
    request = C3FinalCollectRequest(
        evidence_root=evidence_root,
        checkpoint=checkpoint,
        source_manifest=source_manifest,
        source_result=source_result,
        plan=plan,
        plan_sha256=plan_sha256,
        device=args.device,
    )
    validate_source_binding(request)
    return request


def _validate_config_hashes(manifest: Mapping[str, Any]) -> None:
    configs = manifest.get("configs")
    if not isinstance(configs, Mapping):
        raise TypeError("source configs must be a mapping")
    hashes = configs.get("sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != {"env", "agent"}:
        raise ValueError("source config hash inventory is incomplete")
    for name in ("env", "agent"):
        config = configs.get(name)
        if not isinstance(config, Mapping):
            raise TypeError(f"source {name} config must be a mapping")
        if hashes[name] != _json_sha256(config):
            raise ValueError(f"source {name} config SHA-256 does not match")


def validate_source_binding(
    request: C3FinalCollectRequest,
) -> dict[str, str]:
    """Prove that collection is bound to the accepted final V3 source."""
    if not isinstance(request, C3FinalCollectRequest):
        raise TypeError("source binding requires a C3FinalCollectRequest")
    resolved_inputs = {
        "checkpoint": _regular_file(request.checkpoint, "checkpoint"),
        "source_manifest": _regular_file(
            request.source_manifest, "source manifest"
        ),
        "source_result": _regular_file(request.source_result, "source result"),
        "plan": _regular_file(request.plan, "approved plan"),
    }
    if any(
        resolved_inputs[name] != getattr(request, name)
        for name in resolved_inputs
    ):
        raise ValueError("bound source paths must remain canonical")
    if (
        request.checkpoint.parent != request.source_manifest.parent
        or request.checkpoint.parent != request.source_result.parent
        or request.checkpoint.name != "model_2000.pt"
        or request.source_manifest.name != "manifest.json"
        or request.source_result.name != "result.json"
    ):
        raise ValueError("bound source files are not canonical V3 siblings")
    if (
        request.plan_sha256 != APPROVED_PLAN_SHA256
        or sha256_file(request.plan) != request.plan_sha256
    ):
        raise ValueError("bound plan is not the approved C3 design")
    manifest = _load_json(request.source_manifest, "source C3 manifest")
    result = _load_json(request.source_result, "source C3 result")
    validate_c3_manifest(manifest)
    validate_c3_result(result, manifest)
    if manifest["stage"] != FINAL_STAGE or result["status"] != "PASS":
        raise ValueError("source must be a passing C3 V3 training stage")
    lifecycle = lifecycle_identity(FINAL_STAGE)
    if manifest["lifecycle"] != lifecycle or result["lifecycle"] != lifecycle:
        raise ValueError("source lifecycle is not V3 local-2000/global-10000")
    if (
        manifest["iterations"] != LOCAL_ITERATIONS
        or manifest["num_envs"] != TRAIN_NUM_ENVS
    ):
        raise ValueError("source V3 training budget is not canonical")
    metrics = result.get("metrics")
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("transitions") != lifecycle["transitions"]
    ):
        raise ValueError("source V3 transition budget is incomplete")
    finite = metrics.get("finite")
    if (
        not isinstance(finite, Mapping)
        or not finite
        or any(value is not True for value in finite.values())
    ):
        raise ValueError("source V3 finite evidence is incomplete")
    _absolute_exact(
        manifest["cli"].get("run_dir"),
        request.checkpoint.parent,
        "source run",
    )
    _absolute_exact(manifest["plan"].get("path"), request.plan, "source plan")
    if manifest["plan"].get("sha256") != request.plan_sha256:
        raise ValueError("source V3 stage used a different approved plan")
    checkpoint = result.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("source result checkpoint identity is missing")
    _absolute_exact(checkpoint.get("path"), request.checkpoint, "source checkpoint")
    checkpoint_hash = sha256_file(request.checkpoint)
    expected_checkpoint = {
        "sha256": checkpoint_hash,
        "iteration": FINAL_LOCAL_ITERATION,
        "stage": FINAL_STAGE,
        "local_iteration": FINAL_LOCAL_ITERATION,
        "global_iteration": FINAL_GLOBAL_ITERATION,
    }
    for name, expected in expected_checkpoint.items():
        if checkpoint.get(name) != expected:
            raise ValueError(f"source checkpoint {name} is inconsistent")
    _validate_config_hashes(manifest)
    sources = manifest.get("c3_sources")
    if (
        not isinstance(sources, Mapping)
        or sources.get("sha256") != _json_sha256(sources.get("files"))
    ):
        raise ValueError("source C3 source aggregate hash does not match")
    return {
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": sha256_file(request.source_manifest),
        "result_sha256": sha256_file(request.source_result),
        "plan_sha256": request.plan_sha256,
    }


def _verify_bound_inputs(
    request: C3FinalCollectRequest, expected: Mapping[str, str]
) -> None:
    actual = validate_source_binding(request)
    if dict(actual) != dict(expected):
        raise RuntimeError("bound final V3 inputs changed during collection")


def _load_v3_configs(
    runtime_request: RunRequest, simulator: Any
) -> tuple[Any, Any, dict[str, Any]]:
    """Load, stage, then hash each runtime configuration independently."""
    from openhomie_isaaclab.tasks.locomotion.elf3 import elf3_stages

    env_cfg, agent_cfg, _ = simulator._load_configs(runtime_request)
    env_cfg.command_stage = FINAL_STAGE
    if elf3_stages.get_stage(FINAL_STAGE).to_dict() != STAGE_SPECS[FINAL_STAGE]:
        raise RuntimeError("authoritative V3 stage definition changed")
    env_mapping = simulator._normalize_json(env_cfg.to_dict())
    agent_mapping = simulator._normalize_json(agent_cfg.to_dict())
    if env_mapping.get("command_stage") != FINAL_STAGE:
        raise RuntimeError("runtime environment command stage is not V3")
    configs = {
        "env": env_mapping,
        "agent": agent_mapping,
        "sha256": {
            "env": simulator._json_sha256(env_mapping),
            "agent": simulator._json_sha256(agent_mapping),
        },
    }
    return env_cfg, agent_cfg, configs


def _record_failure(current: Exception | None, additional: Exception) -> Exception:
    if current is None:
        return additional
    return RuntimeError(f"{current}; additional failure: {additional}")


def _collect_one(
    request: C3FinalCollectRequest,
    runtime_request: RunRequest,
    provenance: Mapping[str, Any],
    expected_inputs: Mapping[str, str],
) -> dict[str, Any]:
    from openhomie_isaaclab.workflows import elf3_sim

    started = time.perf_counter()
    _verify_bound_inputs(request, expected_inputs)
    elf3_sim._seed_runtime(runtime_request.seed)
    env_cfg, agent_cfg, configs = _load_v3_configs(runtime_request, elf3_sim)
    run_dir = create_run_directory(runtime_request.run_dir)
    checkpoint_iteration, payload = elf3_sim._load_checkpoint(
        request.checkpoint, require_optimizers=False
    )
    del payload
    if checkpoint_iteration != FINAL_LOCAL_ITERATION:
        raise RuntimeError("final V3 checkpoint payload iteration is not local 2000")
    checkpoint = elf3_sim._checkpoint_identity(
        request.checkpoint, checkpoint_iteration
    )
    manifest = elf3_sim._manifest(
        runtime_request,
        start_iteration=checkpoint_iteration,
        configs=configs,
        git=provenance["git"],
        assets=provenance["assets"],
        m4_sources=provenance["m4_sources"],
        runtime=provenance["runtime"],
        gpu=provenance["gpu"],
        checkpoint=checkpoint,
        parent=None,
    )
    write_json_once(run_dir / "manifest.json", manifest)

    env = None
    runner = None
    result: dict[str, Any] | None = None
    failure: Exception | None = None
    try:
        env, runner = elf3_sim._create_runner(
            runtime_request, env_cfg, agent_cfg
        )
        if runtime_request.command == "play":
            runner.load(
                str(request.checkpoint),
                load_optimizer=False,
                map_location=request.device,
            )
            runner.eval_mode()
            policy = runner.get_inference_policy(device=request.device)
            observations = env.get_observations().to(request.device)
            result = elf3_sim._scenario_play_result(
                runtime_request,
                env,
                policy,
                observations,
                expected_inputs["checkpoint_sha256"],
            )
        elif runtime_request.command == "export":
            result = elf3_sim._export_result(runtime_request, runner)
        else:
            raise ValueError("final evidence collector supports play/export only")
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
        _verify_bound_inputs(request, expected_inputs)
        if result is not None and runtime_request.command == "play":
            result["play"]["sha256_after"] = expected_inputs[
                "checkpoint_sha256"
            ]
    except Exception as exc:
        failure = _record_failure(failure, exc)
    if failure is not None:
        result = {
            "status": "FAIL",
            "failure_code": "C3_FINAL_COLLECTION_RUNTIME_FAILURE",
            "error_type": type(failure).__name__,
            "error": str(failure),
        }
    if result is None:
        raise RuntimeError("final evidence run produced no result")
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    write_json_once(run_dir / "result.json", result)
    if failure is not None:
        raise failure
    return result


def _runtime_request(
    request: C3FinalCollectRequest,
    *,
    command: str,
    run_dir: Path,
    seed: int,
    num_envs: int,
    scenario: str | None = None,
    steps: int | None = None,
) -> RunRequest:
    return RunRequest(
        command=command,
        run_dir=run_dir,
        device=request.device,
        seed=seed,
        num_envs=num_envs,
        iterations=None,
        checkpoint=request.checkpoint,
        resume=False,
        headless=True,
        scenario=scenario,
        steps=steps,
    )


def collection_requests(
    request: C3FinalCollectRequest,
) -> tuple[RunRequest, ...]:
    """Return the canonical ordered export and 12-run scenario inventory."""
    if not isinstance(request, C3FinalCollectRequest):
        raise TypeError("inventory requires a C3FinalCollectRequest")
    requests = [
        _runtime_request(
            request,
            command="export",
            run_dir=request.evidence_root / EXPORT_DIRECTORY,
            seed=42,
            num_envs=1,
        )
    ]
    requests.extend(
        _runtime_request(
            request,
            command="play",
            run_dir=(
                request.evidence_root
                / SCENARIOS_DIRECTORY
                / f"{scenario}_seed{seed}"
            ),
            seed=seed,
            num_envs=EVALUATION_NUM_ENVS,
            scenario=scenario,
            steps=EVALUATION_STEPS,
        )
        for scenario in SCENARIOS
        for seed in SEEDS
    )
    return tuple(requests)


def run(request: C3FinalCollectRequest) -> dict[str, Any]:
    """Collect the exact 1-export plus 12-scenario final evidence matrix."""
    if not isinstance(request, C3FinalCollectRequest):
        raise TypeError("collector requires a parsed C3FinalCollectRequest")

    # Isaac imports are intentionally delayed until AppLauncher is live.
    from openhomie_isaaclab.workflows import elf3_sim

    started = time.perf_counter()
    expected_inputs = validate_source_binding(request)
    repo_root = elf3_sim._repository_root()
    git = elf3_sim._git_identity(repo_root)
    assets, m4_sources = elf3_sim._source_identity(repo_root)
    source_manifest = _load_json(request.source_manifest, "source C3 manifest")
    if git.get("commit") != source_manifest.get("git", {}).get("commit"):
        raise RuntimeError("collector Git commit differs from final V3 training")
    if assets != source_manifest.get("assets"):
        raise RuntimeError("collector assets differ from final V3 training")
    if m4_sources != source_manifest.get("m4_sources"):
        raise RuntimeError("collector M4 sources differ from final V3 training")
    provenance = {
        "git": git,
        "assets": assets,
        "m4_sources": m4_sources,
        "runtime": elf3_sim._runtime_identity(),
        "gpu": elf3_sim._gpu_identity(request.device),
    }

    root = create_run_directory(request.evidence_root)
    scenarios = root / SCENARIOS_DIRECTORY
    scenarios.mkdir(mode=0o755)
    completed: list[str] = []
    for runtime_request in collection_requests(request):
        _collect_one(
            request, runtime_request, provenance, expected_inputs
        )
        completed.append(runtime_request.run_dir.name)

    _verify_bound_inputs(request, expected_inputs)
    final_request = C3FinalRequest(
        evidence_root=root,
        checkpoint=request.checkpoint,
        source_manifest=request.source_manifest,
        source_result=request.source_result,
        plan=request.plan,
        plan_sha256=request.plan_sha256,
    )
    aggregate = write_or_verify_aggregate(final_request)
    if aggregate.get("contract", {}).get("passed") is not True:
        raise RuntimeError("final evidence checker did not pass its contract")
    return {
        "status": "PASS",
        "collection_status": "COMPLETE",
        "acceptance_status": aggregate["status"],
        "final_acceptance_passed": aggregate["acceptance"]["overall"]["passed"],
        "stage": FINAL_STAGE,
        "local_iteration": FINAL_LOCAL_ITERATION,
        "global_iteration": FINAL_GLOBAL_ITERATION,
        "checkpoint_sha256": expected_inputs["checkpoint_sha256"],
        "completed_runs": completed,
        "aggregate": str(final_request.aggregate),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
