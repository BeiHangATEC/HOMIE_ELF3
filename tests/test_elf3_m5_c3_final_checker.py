"""Contracts for the CPU-only ELF3 M5 C3 final evidence checker."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "isaaclab_ext/source/openhomie_isaaclab"
SCRIPT = ROOT / "isaaclab_ext/scripts/check_elf3_m5_c3_final.py"
PLAN = ROOT / "docs/plans/elf3-m5-c3-continual-training-design.md"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from openhomie_isaaclab.workflows import elf3_c3, elf3_c3_final


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _runtime_identity(env, agent, *, commit: str) -> dict:
    m4_files = {"runner.py": "1" * 64}
    return {
        "git": {"commit": commit, "dirty_paths": []},
        "configs": {
            "env": env,
            "agent": agent,
            "sha256": {"env": _json_sha(env), "agent": _json_sha(agent)},
        },
        "assets": {"urdf_sha256": "2" * 64, "usd_sha256": "3" * 64},
        "m4_sources": {"files": m4_files, "sha256": _json_sha(m4_files)},
        "runtime": {"python": "3.11"},
        "gpu": {"total_mib": 32000, "free_mib": 16000},
    }


def _source_evidence(parent: Path) -> tuple[Path, Path, Path]:
    run = parent / "v3"
    run.mkdir()
    checkpoint = run / "model_2000.pt"
    checkpoint.write_bytes(b"C3 V3 final checkpoint")
    parent_checkpoint = parent / "v2-model_2000.pt"
    parent_checkpoint.write_bytes(b"C3 V2 checkpoint")
    parent_manifest = parent / "v2-manifest.json"
    parent_manifest.write_text("{}\n", encoding="utf-8")
    request = elf3_c3.C3Request(
        stage="V3",
        run_dir=run.resolve(),
        checkpoint=parent_checkpoint.resolve(),
        checkpoint_sha256=_sha(parent_checkpoint),
        parent_manifest=parent_manifest.resolve(),
        parent_manifest_sha256=_sha(parent_manifest),
        plan=PLAN.resolve(),
        plan_sha256=_sha(PLAN),
        device="cuda:0",
        seed=42,
        num_envs=4096,
        headless=True,
    )
    env = {"command_stage": "V3"}
    agent = {"num_steps_per_env": 50, "max_iterations": 2000}
    identity = _runtime_identity(env, agent, commit="a" * 40)
    c3_files = {"elf3_c3.py": "4" * 64}
    provenance = {
        **identity,
        "c3_sources": {"files": c3_files, "sha256": _json_sha(c3_files)},
    }
    manifest = elf3_c3.build_manifest(
        request,
        parent={
            "kind": "c3",
            "stage": "V2",
            "local_iteration": 2000,
            "global_iteration": 8000,
            "checkpoint_path": str(parent_checkpoint.resolve()),
            "checkpoint_sha256": _sha(parent_checkpoint),
            "manifest_path": str(parent_manifest.resolve()),
            "manifest_sha256": _sha(parent_manifest),
        },
        provenance=provenance,
        stage_spec=dict(elf3_c3.STAGE_SPECS["V3"]),
        created_utc="2026-08-18T00:00:00Z",
    )
    manifest_path = run / "manifest.json"
    _write_json(manifest_path, manifest)
    load_contract = {
        "mode": "weights_only",
        "load_optimizer": False,
        "checkpoint_iteration": 2000,
        "local_iteration_after_load": 0,
        "optimizer_state_entries": {"policy": 0, "estimator": 0},
        "learning_rates": {"policy": 0.001, "estimator": 0.001},
    }
    parent_verification = {
        "checkpoint_sha256_before": _sha(parent_checkpoint),
        "checkpoint_sha256_after": _sha(parent_checkpoint),
        "manifest_sha256_before": _sha(parent_manifest),
        "manifest_sha256_after": _sha(parent_manifest),
        "plan_sha256_before": _sha(PLAN),
        "plan_sha256_after": _sha(PLAN),
    }
    result = elf3_c3.build_pass_result(
        request,
        manifest=manifest,
        load_contract=load_contract,
        metrics={
            "training": {"mean_reward": 1.0},
            "final_learning_rates": {"policy": 0.001, "estimator": 0.001},
            "finite": {"observations": True, "actions": True, "losses": True},
            "transitions": manifest["lifecycle"]["transitions"],
        },
        checkpoint={
            "path": str(checkpoint.resolve()),
            "sha256": _sha(checkpoint),
            "iteration": 2000,
        },
        parent_verification=parent_verification,
        elapsed_seconds=1.0,
    )
    result_path = run / "result.json"
    _write_json(result_path, result)
    return checkpoint.resolve(), manifest_path.resolve(), result_path.resolve()


def _runtime_manifest(
    run: Path,
    source_manifest: dict,
    checkpoint: Path,
    *,
    command: str,
    seed: int,
    num_envs: int,
) -> dict:
    env = {"scene": {"num_envs": num_envs}, "command_stage": "V3"}
    agent = {"seed": seed, "num_steps_per_env": 50, "max_iterations": 2000}
    identity = _runtime_identity(env, agent, commit=source_manifest["git"]["commit"])
    identity["assets"] = deepcopy(source_manifest["assets"])
    identity["m4_sources"] = deepcopy(source_manifest["m4_sources"])
    return {
        "schema_version": 1,
        "command": command,
        "created_utc": "2026-08-18T01:00:00Z",
        "task_id": "OpenHomie-Elf3-Homie-Direct-v0",
        "seed": seed,
        "device": "cuda:0",
        "num_envs": num_envs,
        "iterations": None,
        "cli": {
            "command": command,
            "run_dir": str(run.resolve()),
            "device": "cuda:0",
            "seed": seed,
            "num_envs": num_envs,
            "iterations": None,
            "headless": True,
            "resume": False,
            "checkpoint": str(checkpoint),
            "scenario": None,
            "steps": None,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha(checkpoint),
            "iteration": 2000,
        },
        **identity,
    }


def _scenario_metrics(scenario: str) -> dict[str, float]:
    return {
        "stand": {"height_mae": 0.05, "tilt_rms": 0.10},
        "forward": {"velocity_mae": 0.10, "height_mae": 0.05},
        "turn": {"yaw_rate_mae": 0.10, "height_mae": 0.05},
        "crouch": {"height_mae": 0.05, "planar_speed_rms": 0.10},
    }[scenario]


def _scenario_command(scenario: str) -> list[float]:
    command = list(elf3_c3_final.EVALUATION_SCENARIOS[scenario].command)
    command[3] = 0.80 if scenario == "crouch" else 1.01
    return command


def _evidence_tree(tmp_path: Path) -> tuple[elf3_c3_final.C3FinalRequest, Path]:
    checkpoint, source_manifest_path, source_result_path = _source_evidence(tmp_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    root = tmp_path / "final-evidence"
    root.mkdir()
    export = root / "exact_export"
    export.mkdir()
    export_manifest = _runtime_manifest(
        export, source_manifest, checkpoint, command="export", seed=42, num_envs=1
    )
    ts = export / "policy.ts"
    onnx = export / "policy.onnx"
    ts.write_bytes(b"torchscript")
    onnx.write_bytes(b"onnx")
    samples = export / "parity_samples.npz"
    np.savez(
        samples,
        history_1=np.zeros((1, 468), dtype=np.float32),
        expected_1=np.zeros((1, 12), dtype=np.float32),
        history_4=np.zeros((4, 468), dtype=np.float32),
        expected_4=np.zeros((4, 12), dtype=np.float32),
    )
    _write_json(export / "manifest.json", export_manifest)
    _write_json(
        export / "result.json",
        {
            "status": "PASS",
            "exports": {
                "oracle": {"device": "cpu", "method": "runner.get_inference_policy"},
                "torchscript": {
                    "path": str(ts.resolve()),
                    "sha256": _sha(ts),
                    "provider": "torch.jit.load",
                    "fresh_runtime": True,
                    "batches": [1, 4],
                    "input_shapes": [[1, 468], [4, 468]],
                    "output_shapes": [[1, 12], [4, 12]],
                    "max_abs_error": 1.0e-8,
                },
                "onnx": {
                    "path": str(onnx.resolve()),
                    "sha256": _sha(onnx),
                    "providers": ["CPUExecutionProvider"],
                    "checker_passed": True,
                    "fresh_runtime": True,
                    "batches": [1, 4],
                    "input_shapes": [[1, 468], [4, 468]],
                    "output_shapes": [[1, 12], [4, 12]],
                    "max_abs_error": 1.0e-6,
                },
            },
        },
    )
    scenarios = root / "scenarios"
    scenarios.mkdir()
    for scenario in elf3_c3_final.SCENARIOS:
        for seed in elf3_c3_final.SEEDS:
            key = f"{scenario}_seed{seed}"
            run = scenarios / key
            run.mkdir()
            manifest = _runtime_manifest(
                run, source_manifest, checkpoint, command="play", seed=seed, num_envs=16
            )
            manifest["cli"].update(scenario=scenario, steps=1000)
            _write_json(run / "manifest.json", manifest)
            _write_json(
                run / "result.json",
                {
                    "status": "PASS",
                    "play": {
                        "scenario": scenario,
                        "command": _scenario_command(scenario),
                        "mode": elf3_c3_final.EVALUATION_SCENARIOS[scenario].mode,
                        "steps": 1000,
                        "num_envs": 16,
                        "seed": seed,
                        "checkpoint_path": str(checkpoint),
                        "finite": True,
                        "credited_env_steps": 16000,
                        "non_timeout_termination_steps": [None] * 16,
                        "non_timeout_termination_reasons": [None] * 16,
                        "timeout_count": 16,
                        "survival": 1.0,
                        "metrics": _scenario_metrics(scenario),
                        "action_sha256": hashlib.sha256(f"actions:{key}".encode()).hexdigest(),
                        "trajectory_sha256": hashlib.sha256(f"trajectory:{key}".encode()).hexdigest(),
                        "sha256_before": _sha(checkpoint),
                        "sha256_after": _sha(checkpoint),
                    },
                },
            )
    request = elf3_c3_final.C3FinalRequest(
        evidence_root=root.resolve(),
        checkpoint=checkpoint,
        source_manifest=source_manifest_path,
        source_result=source_result_path,
        plan=PLAN.resolve(),
        plan_sha256=_sha(PLAN),
    )
    return request, root


def _mutate(path: Path, callback) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    callback(payload)
    _write_json(path, payload)


def test_complete_contract_is_pass_but_final_acceptance_is_pending(tmp_path):
    request, _ = _evidence_tree(tmp_path)
    aggregate = elf3_c3_final.write_or_verify_aggregate(request)
    assert aggregate["contract"]["passed"] is True
    assert aggregate["status"] == "PENDING_CONVERGENCE"
    assert aggregate["acceptance"]["overall"] == {
        "passed": False,
        "status": "PENDING_CONVERGENCE",
        "reason": "C3 convergence evidence basis and acceptance window are not yet approved",
    }
    assert aggregate["acceptance"]["checkpoint"]["global_iteration"] == 10000
    assert aggregate["acceptance"]["behavior"]["scenario_runs"] == 12
    assert request.aggregate.is_file()
    assert elf3_c3_final.verify_final_contract(request) == aggregate


def test_cli_writes_pending_aggregate_and_never_claims_final_pass(tmp_path):
    request, _ = _evidence_tree(tmp_path)
    command = [
        sys.executable,
        str(SCRIPT),
        "--evidence-root", str(request.evidence_root),
        "--checkpoint", str(request.checkpoint),
        "--source-manifest", str(request.source_manifest),
        "--source-result", str(request.source_result),
        "--plan", str(request.plan),
        "--plan-sha256", request.plan_sha256,
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("C3_FINAL_CONTRACT_PASS") == 1
    assert "C3_FINAL_ACCEPTANCE_PASS" not in completed.stdout
    payload = json.loads(request.aggregate.read_text(encoding="utf-8"))
    assert payload["status"] == "PENDING_CONVERGENCE"
    assert payload["acceptance"]["overall"]["passed"] is False


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("global_identity", "lifecycle"),
        ("missing_scenario", "scenario matrix"),
        ("foreign_checkpoint", "foreign checkpoint identity"),
        ("runtime_stage", "runtime command stage is not V3"),
        ("behavior", "behavior gate failed"),
        ("export_parity", "onnx parity exceeds"),
        ("parity_samples", "parity sample history_1 is invalid"),
    ),
)
def test_contract_rejects_identity_inventory_behavior_and_export_failures(
    tmp_path, mutation, reason
):
    request, root = _evidence_tree(tmp_path)
    if mutation == "global_identity":
        _mutate(
            request.source_manifest,
            lambda payload: payload["lifecycle"]["global_iterations"].update(final=9999),
        )
    elif mutation == "missing_scenario":
        run = root / "scenarios/crouch_seed44"
        (run / "manifest.json").unlink()
        (run / "result.json").unlink()
        run.rmdir()
    elif mutation == "foreign_checkpoint":
        _mutate(
            root / "scenarios/forward_seed43/manifest.json",
            lambda payload: payload["checkpoint"].update(sha256="f" * 64),
        )
    elif mutation == "runtime_stage":
        manifest = root / "scenarios/turn_seed44/manifest.json"
        _mutate(
            manifest,
            lambda payload: (
                payload["configs"]["env"].update(command_stage="S0"),
                payload["configs"]["sha256"].update(
                    env=_json_sha(payload["configs"]["env"])
                ),
            ),
        )
    elif mutation == "behavior":
        for seed in elf3_c3_final.SEEDS:
            _mutate(
                root / f"scenarios/stand_seed{seed}/result.json",
                lambda payload: payload["play"]["metrics"].update(tilt_rms=0.21),
            )
    elif mutation == "export_parity":
        _mutate(
            root / "exact_export/result.json",
            lambda payload: payload["exports"]["onnx"].update(max_abs_error=1.1e-5),
        )
    else:
        samples = root / "exact_export/parity_samples.npz"
        np.savez(
            samples,
            history_1=np.zeros((1, 467), dtype=np.float32),
            expected_1=np.zeros((1, 12), dtype=np.float32),
            history_4=np.zeros((4, 468), dtype=np.float32),
            expected_4=np.zeros((4, 12), dtype=np.float32),
        )
    with pytest.raises((KeyError, TypeError, ValueError), match=reason):
        elf3_c3_final.verify_final_contract(request)


def test_recorded_aggregate_cannot_upgrade_pending_to_final_pass(tmp_path):
    request, _ = _evidence_tree(tmp_path)
    elf3_c3_final.write_or_verify_aggregate(request)
    _mutate(
        request.aggregate,
        lambda payload: payload["acceptance"]["overall"].update(
            passed=True, status="PASS", reason=""
        ),
    )
    with pytest.raises(ValueError, match="recorded aggregate differs"):
        elf3_c3_final.verify_final_contract(request)


def test_contract_module_and_checker_remain_cpu_only():
    source = (PACKAGE / "openhomie_isaaclab/workflows/elf3_c3_final.py").read_text(
        encoding="utf-8"
    )
    script = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("isaaclab.app", "AppLauncher", "pxr", "gym.make", "cuda()"):
        assert forbidden not in source
        assert forbidden not in script
