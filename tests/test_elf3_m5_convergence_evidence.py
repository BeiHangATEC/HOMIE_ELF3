"""Contracts for M5 C2 convergence and deterministic behavior evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "isaaclab_ext/scripts/check_elf3_m5_convergence.py"
SCENARIOS = ("stand", "forward", "turn", "crouch")
SEEDS = (42, 43, 44)
ITERATIONS = 2000
NUM_ENVS = 4096
STEPS_PER_ENV = 50
BUDGET = NUM_ENVS * ITERATIONS * STEPS_PER_ENV


def _load_harness() -> ModuleType:
    assert HARNESS.is_file(), "missing planner-owned M5 C2 convergence harness"
    spec = importlib.util.spec_from_file_location("_elf3_m5_convergence", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload, *, allow_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=allow_nan) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _mutate_json(path: Path, mutate, *, allow_nan: bool = False) -> None:
    payload = _read_json(path)
    mutate(payload)
    _write_json(path, payload, allow_nan=allow_nan)


def _refresh_reference(root: Path, keys: tuple[str, ...], path: Path) -> None:
    aggregate_path = root / "aggregate_result.json"
    aggregate = _read_json(aggregate_path)
    reference = aggregate["artifacts"]
    for key in keys:
        reference = reference[key]
    reference.clear()
    reference.update(_file_ref(root, path))
    _write_json(aggregate_path, aggregate)



def _identity(harness: ModuleType, *, command: str, seed: int, num_envs: int) -> dict:
    env_cfg = {"scene": {"num_envs": num_envs}, "task": "ELF3"}
    agent_cfg = {
        "seed": seed,
        "num_steps_per_env": STEPS_PER_ENV,
        "max_iterations": ITERATIONS if command == "train" else 50000,
        "save_interval": 200,
    }
    m4_files = {
        path: _sha256(ROOT / path) for path in harness.M4_PATHS
    }
    return {
        "git": {"commit": "4" * 40, "dirty_paths": []},
        "configs": {
            "env": env_cfg,
            "agent": agent_cfg,
            "sha256": {
                "env": _json_sha256(env_cfg),
                "agent": _json_sha256(agent_cfg),
            },
        },
        "assets": {
            "urdf_sha256": _sha256(harness.URDF_PATH),
            "usd_sha256": _sha256(harness.USD_PATH),
        },
        "m4_sources": {
            "files": m4_files,
            "sha256": _json_sha256(m4_files),
        },
        "runtime": {
            "python": "3.11.13",
            "isaaclab_path": (
                "/opt/IsaacLab-v2.3.2/source/isaaclab/isaaclab/__init__.py"
            ),
            "isaaclab_app_path": (
                "/opt/IsaacLab-v2.3.2/source/isaaclab/isaaclab/app.py"
            ),
            "isaaclab_rl_path": (
                "/opt/IsaacLab-v2.3.2/source/isaaclab_rl/isaaclab_rl/__init__.py"
            ),
            "versions": dict(harness.EXPECTED_VERSIONS),
        },
        "gpu": {
            "name": "NVIDIA GeForce RTX 5090",
            "driver_version": "590.48.01",
            "cuda_version": "12.8",
            "total_mib": 32607,
            "free_mib": 16384,
            "capability": [12, 0],
            "arch_list": ["sm_120"],
            "cuda_probe_passed": True,
        },
    }


def _manifest(
    harness: ModuleType,
    *,
    command: str,
    run_dir: Path,
    seed: int,
    num_envs: int,
    iterations: int | None,
) -> dict:
    payload = {
        "schema_version": 1,
        "command": command,
        "created_utc": "2026-08-17T00:00:00Z",
        "task_id": "OpenHomie-Elf3-Homie-Direct-v0",
        "seed": seed,
        "device": "cuda:0",
        "num_envs": num_envs,
        "iterations": iterations,
        "cli": {
            "command": command,
            "run_dir": str(run_dir),
            "device": "cuda:0",
            "seed": seed,
            "num_envs": num_envs,
            "iterations": iterations,
            "headless": True,
            "resume": False,
            "checkpoint": None,
            "scenario": None,
            "steps": None,
        },
    }
    payload.update(_identity(harness, command=command, seed=seed, num_envs=num_envs))
    return payload


def _file_ref(root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}


def _write_events(path: Path, harness: ModuleType) -> Path:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=path, max_queue=1)
    try:
        for step in range(1, ITERATIONS + 1):
            episode_length = 50.0 if step <= 100 else 350.0
            timeout = 0.1 if step > ITERATIONS - 5 else 0.0
            for tag in harness.REQUIRED_TAGS:
                if tag == "Train/mean_episode_length":
                    value = episode_length
                elif tag == "Episode_Termination/time_out":
                    value = timeout
                else:
                    value = 1.0
                writer.add_scalar(tag, value, step)
            writer.add_scalar("Episode_Reward/tracking_lin_vel", 1.0, step)
    finally:
        writer.close()
    events = list(path.glob("events.out.tfevents.*"))
    assert len(events) == 1
    return events[0]


def _replace_events(
    root: Path,
    harness: ModuleType,
    *,
    point_count: int = ITERATIONS,
    missing_tag: str | None = None,
    duplicate_step: bool = False,
    timeout_out_of_range: bool = False,
) -> Path:
    train = root / "canonical_train"
    for event in train.glob("events.out.tfevents.*"):
        event.unlink()
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=train, max_queue=1)
    try:
        for step in range(1, point_count + 1):
            for tag in harness.REQUIRED_TAGS:
                if tag == missing_tag:
                    continue
                if tag == "Train/mean_episode_length":
                    value = 50.0 if step <= 100 else 350.0
                elif tag == "Episode_Termination/time_out":
                    value = 0.1 if step > point_count - 5 else 0.0
                    if timeout_out_of_range and step == 1:
                        value = 1.1
                else:
                    value = 1.0
                writer.add_scalar(tag, value, step)
        if duplicate_step:
            writer.add_scalar("Train/mean_reward", 1.0, point_count)
    finally:
        writer.close()
    events = list(train.glob("events.out.tfevents.*"))
    assert len(events) == 1
    return events[0]


def _scenario_metrics(scenario: str) -> dict[str, float]:
    return {
        "stand": {"height_mae": 0.05, "tilt_rms": 0.10},
        "forward": {"velocity_mae": 0.10, "height_mae": 0.05},
        "turn": {"yaw_rate_mae": 0.10, "height_mae": 0.05},
        "crouch": {"height_mae": 0.05, "planar_speed_rms": 0.10},
    }[scenario]


def _command(harness: ModuleType, scenario: str) -> list[float]:
    raw = list(harness.EVALUATION_SCENARIOS[scenario].command)
    raw[3] = 0.80 if scenario == "crouch" else 1.01
    return raw


def _complete_evidence_tree(parent: Path, harness: ModuleType) -> Path:
    root = parent / "m5-c2-evidence"
    root.mkdir()
    train = root / "canonical_train"
    train.mkdir()
    checkpoints = {}
    for iteration in range(200, ITERATIONS + 1, 200):
        path = train / f"model_{iteration}.pt"
        path.write_bytes(f"ELF3 checkpoint iteration {iteration}".encode())
        checkpoints[iteration] = path
    checkpoint = checkpoints[ITERATIONS]
    checkpoint_hash = _sha256(checkpoint)

    train_manifest = _manifest(
        harness,
        command="train",
        run_dir=train,
        seed=42,
        num_envs=NUM_ENVS,
        iterations=ITERATIONS,
    )
    train_manifest["start_iteration"] = 0
    _write_json(train / "manifest.json", train_manifest)
    _write_json(
        train / "result.json",
        {
            "status": "PASS",
            "start_iteration": 0,
            "final_iteration": ITERATIONS,
            "finite": {
                name: True
                for name in (
                    "observations",
                    "actions",
                    "rewards",
                    "losses",
                    "learning_rates",
                    "entropy",
                    "estimator_metrics",
                    "checkpoint_values",
                )
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_hash,
                "iteration": ITERATIONS,
            },
        },
    )
    event = _write_events(train, harness)

    export = root / "exact_export"
    export.mkdir()
    export_manifest = _manifest(
        harness,
        command="export",
        run_dir=export,
        seed=42,
        num_envs=1,
        iterations=None,
    )
    export_manifest["checkpoint"] = {
        "path": str(checkpoint),
        "sha256": checkpoint_hash,
        "iteration": ITERATIONS,
    }
    export_manifest["cli"]["checkpoint"] = str(checkpoint)
    ts = export / "policy.ts"
    onnx = export / "policy.onnx"
    parity_samples = export / "parity_samples.npz"
    ts.write_bytes(b"canonical torchscript export")
    onnx.write_bytes(b"canonical onnx export")
    parity_samples.write_bytes(b"canonical parity samples")
    _write_json(export / "manifest.json", export_manifest)
    _write_json(
        export / "result.json",
        {
            "status": "PASS",
            "exports": {
                "oracle": {
                    "device": "cpu",
                    "method": "runner.get_inference_policy",
                },
                "torchscript": {
                    "path": str(ts),
                    "sha256": _sha256(ts),
                    "provider": "torch.jit.load",
                    "fresh_runtime": True,
                    "batches": [1, 4],
                    "input_shapes": [[1, 468], [4, 468]],
                    "output_shapes": [[1, 12], [4, 12]],
                    "max_abs_error": 1e-8,
                },
                "onnx": {
                    "path": str(onnx),
                    "sha256": _sha256(onnx),
                    "providers": ["CPUExecutionProvider"],
                    "checker_passed": True,
                    "fresh_runtime": True,
                    "batches": [1, 4],
                    "input_shapes": [[1, 468], [4, 468]],
                    "output_shapes": [[1, 12], [4, 12]],
                    "max_abs_error": 1e-6,
                },
            },
        },
    )

    scenarios_root = root / "scenarios"
    scenarios_root.mkdir()
    for scenario in SCENARIOS:
        for seed in SEEDS:
            key = f"{scenario}_seed{seed}"
            run = scenarios_root / key
            run.mkdir()
            manifest = _manifest(
                harness,
                command="play",
                run_dir=run,
                seed=seed,
                num_envs=16,
                iterations=None,
            )
            manifest["checkpoint"] = {
                "path": str(checkpoint),
                "sha256": checkpoint_hash,
                "iteration": ITERATIONS,
            }
            manifest["cli"].update(
                {
                    "checkpoint": str(checkpoint),
                    "scenario": scenario,
                    "steps": 1000,
                }
            )
            _write_json(run / "manifest.json", manifest)
            action_hash = hashlib.sha256(f"action:{key}".encode()).hexdigest()
            trajectory_hash = hashlib.sha256(
                f"trajectory:{key}".encode()
            ).hexdigest()
            credited = 8 * 920 + 8 * 1000
            _write_json(
                run / "result.json",
                {
                    "status": "PASS",
                    "play": {
                        "scenario": scenario,
                        "command": _command(harness, scenario),
                        "mode": harness.EVALUATION_SCENARIOS[scenario].mode,
                        "steps": 1000,
                        "num_envs": 16,
                        "seed": seed,
                        "checkpoint_path": str(checkpoint),
                        "finite": True,
                        "credited_env_steps": credited,
                        "non_timeout_termination_steps": [920] * 8 + [None] * 8,
                        "non_timeout_termination_reasons": ["terminated"] * 8 + [None] * 8,
                        "timeout_count": 0,
                        "survival": credited / 16000,
                        "metrics": _scenario_metrics(scenario),
                        "action_sha256": action_hash,
                        "trajectory_sha256": trajectory_hash,
                        "sha256_before": checkpoint_hash,
                        "sha256_after": checkpoint_hash,
                    },
                },
            )

    aggregate = harness.build_aggregate_payload(root)
    _write_json(root / "aggregate_result.json", aggregate)
    return root


@pytest.fixture
def evidence_root(tmp_path):
    harness = _load_harness()
    return _complete_evidence_tree(tmp_path, harness)


def test_valid_complete_fixture_passes(evidence_root):
    result = _load_harness().verify_convergence_evidence(evidence_root)
    assert result["status"] == "PASS"
    assert result["actual_transitions"] == BUDGET
    assert result["positive_timeout_points"] == 5
    assert result["checkpoint_iteration"] == ITERATIONS
    assert result["scenario_runs"] == 12


def test_cli_requires_one_absolute_existing_evidence_root(tmp_path, evidence_root):
    command = [sys.executable, str(HARNESS), "--evidence-root"]
    for value in (str(tmp_path / "missing"), "relative"):
        completed = subprocess.run(
            [*command, value], check=False, text=True, capture_output=True
        )
        assert completed.returncode != 0
        assert "M5_CONVERGENCE_FAIL" in completed.stdout
        assert "M5_CONVERGENCE_PASS" not in completed.stdout
        assert "skip" not in completed.stdout.lower()
    completed = subprocess.run(
        [*command, str(evidence_root)], check=False, text=True, capture_output=True
    )
    assert completed.returncode == 0
    assert completed.stdout.count("M5_CONVERGENCE_PASS") == 1


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_result",
        "malformed_json",
        "nonfinite_json",
        "hash_mismatch",
        "foreign_checkpoint",
        "truncated_event",
        "missing_tag",
        "threshold",
        "budget",
        "checkpoint_iteration",
        "export_parity",
        "foreign_git",
        "duplicate_event",
        "duplicate_step",
        "timeout_range",
        "checkpoint_chain",
        "parity_samples",
        "matrix_missing",
        "aggregate_claim",
        "aggregate_reference",
    ),
)
def test_rejects_each_evidence_failure(evidence_root, mutation):
    root = evidence_root
    if mutation == "missing_result":
        (root / "scenarios/stand_seed42/result.json").unlink()
    elif mutation == "malformed_json":
        (root / "canonical_train/result.json").write_text("{", encoding="utf-8")
        _refresh_reference(root, ("canonical_train", "result"), root / "canonical_train/result.json")
    elif mutation == "nonfinite_json":
        _mutate_json(
            root / "scenarios/turn_seed43/result.json",
            lambda p: p["play"]["metrics"].update(yaw_rate_mae=float("nan")),
            allow_nan=True,
        )
        _refresh_reference(root, ("scenarios", "turn_seed43", "result"), root / "scenarios/turn_seed43/result.json")
    elif mutation == "hash_mismatch":
        _mutate_json(
            root / "aggregate_result.json",
            lambda p: p["artifacts"]["canonical_train"]["result"].update(
                sha256="0" * 64
            ),
        )
    elif mutation == "foreign_checkpoint":
        _mutate_json(
            root / "scenarios/forward_seed44/manifest.json",
            lambda p: p["checkpoint"].update(sha256="f" * 64),
        )
        _refresh_reference(root, ("scenarios", "forward_seed44", "manifest"), root / "scenarios/forward_seed44/manifest.json")
    elif mutation in {"truncated_event", "missing_tag"}:
        harness = _load_harness()
        event = _replace_events(
            root,
            harness,
            point_count=ITERATIONS - 1 if mutation == "truncated_event" else ITERATIONS,
            missing_tag="Loss/estimator_swap" if mutation == "missing_tag" else None,
        )
        _refresh_reference(root, ("canonical_train", "event"), event)
    elif mutation == "threshold":
        for seed in SEEDS:
            result = root / f"scenarios/stand_seed{seed}/result.json"
            _mutate_json(
                result, lambda p: p["play"]["metrics"].update(tilt_rms=0.21)
            )
            _refresh_reference(
                root, ("scenarios", f"stand_seed{seed}", "result"), result
            )
    elif mutation == "budget":
        _mutate_json(
            root / "canonical_train/manifest.json",
            lambda p: p.update(iterations=1999),
        )
        _refresh_reference(root, ("canonical_train", "manifest"), root / "canonical_train/manifest.json")
    elif mutation == "checkpoint_iteration":
        _mutate_json(
            root / "canonical_train/result.json",
            lambda p: p["checkpoint"].update(iteration=1999),
        )
        _refresh_reference(root, ("canonical_train", "result"), root / "canonical_train/result.json")
    elif mutation == "export_parity":
        _mutate_json(
            root / "exact_export/result.json",
            lambda p: p["exports"]["onnx"].update(max_abs_error=1.1e-5),
        )
        _refresh_reference(root, ("exact_export", "result"), root / "exact_export/result.json")
    elif mutation == "foreign_git":
        manifest = root / "scenarios/crouch_seed43/manifest.json"
        _mutate_json(
            manifest,
            lambda p: p["git"].update(commit="5" * 40),
        )
        _refresh_reference(
            root, ("scenarios", "crouch_seed43", "manifest"), manifest
        )
    elif mutation == "duplicate_event":
        event = next((root / "canonical_train").glob("events.out.tfevents.*"))
        shutil.copyfile(event, event.with_name(event.name + ".duplicate"))
    elif mutation == "duplicate_step":
        event = _replace_events(root, _load_harness(), duplicate_step=True)
        _refresh_reference(root, ("canonical_train", "event"), event)
    elif mutation == "timeout_range":
        event = _replace_events(
            root,
            _load_harness(),
            timeout_out_of_range=True,
        )
        _refresh_reference(root, ("canonical_train", "event"), event)
    elif mutation == "checkpoint_chain":
        (root / "canonical_train/model_1000.pt").unlink()
    elif mutation == "parity_samples":
        (root / "exact_export/parity_samples.npz").unlink()
    elif mutation == "matrix_missing":
        shutil.rmtree(root / "scenarios/crouch_seed44")
    elif mutation == "aggregate_claim":
        _mutate_json(
            root / "aggregate_result.json",
            lambda p: p["acceptance"]["overall"].update(passed=False),
        )
    else:
        _mutate_json(
            root / "aggregate_result.json",
            lambda p: p["artifacts"]["scenarios"].pop("turn_seed43"),
        )

    expected_reason = {
        "missing_result": "contains unexpected artifacts",
        "malformed_json": "Expecting property name",
        "nonfinite_json": "contains NaN",
        "hash_mismatch": "aggregate artifact references",
        "foreign_checkpoint": "foreign checkpoint hash",
        "truncated_event": "insufficient points",
        "missing_tag": "missing TensorBoard scalar",
        "threshold": "behavior gate failed",
        "budget": "manifest iterations",
        "checkpoint_iteration": "training checkpoint iteration",
        "export_parity": "onnx parity exceeds",
        "foreign_git": "foreign Git commit",
        "duplicate_event": "exactly one regular event file",
        "duplicate_step": "point count is not exact",
        "timeout_range": "timeout transition fractions",
        "checkpoint_chain": "checkpoint chain is missing",
        "parity_samples": "exact export directory contains unexpected artifacts",
        "matrix_missing": "scenario matrix",
        "aggregate_claim": "aggregate acceptance claims",
        "aggregate_reference": "aggregate artifact references",
    }[mutation]
    with pytest.raises((KeyError, TypeError, ValueError)) as error:
        _load_harness().verify_convergence_evidence(root)
    assert expected_reason in str(error.value)


def test_rejects_symlink_and_path_escape(evidence_root, tmp_path):
    harness = _load_harness()
    result = evidence_root / "scenarios/stand_seed42/result.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(result.read_bytes())
    result.unlink()
    result.symlink_to(outside)
    with pytest.raises(ValueError):
        harness.verify_convergence_evidence(evidence_root)

    result.unlink()
    result.write_bytes(outside.read_bytes())
    _mutate_json(
        evidence_root / "aggregate_result.json",
        lambda p: p["artifacts"]["scenarios"]["stand_seed42"]["result"].update(
            path="../outside.json"
        ),
    )
    with pytest.raises(ValueError):
        harness.verify_convergence_evidence(evidence_root)




def test_reduced_env_fallback_threads_dynamic_checkpoint_iteration(tmp_path):
    harness = _load_harness()
    assert harness.canonical_iterations(2048, STEPS_PER_ENV) == 4000
    checkpoint = tmp_path / "model_4000.pt"
    checkpoint.write_bytes(b"fallback checkpoint")
    digest = _sha256(checkpoint)
    manifest = {
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": digest,
            "iteration": 4000,
        }
    }
    harness._check_source_checkpoint(
        manifest, checkpoint.resolve(), digest, 4000, "fallback"
    )
    manifest["checkpoint"]["iteration"] = 2000
    with pytest.raises(ValueError, match="checkpoint iteration"):
        harness._check_source_checkpoint(
            manifest, checkpoint.resolve(), digest, 4000, "fallback"
        )
    source = HARNESS.read_text(encoding="utf-8")
    assert 'source.get("iteration") != checkpoint_iteration' in source
def test_checker_freezes_c1_sources_tags_layout_and_pure_helpers():
    harness = _load_harness()
    assert set(harness.FROZEN_C1_SOURCES) == {
        "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/him_rl/runner.py",
        "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/tasks/locomotion/elf3/elf3_homie_env.py",
        "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/elf3_run.py",
        "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/workflows/elf3_sim.py",
    }
    assert len(harness.REQUIRED_TAGS) == 12
    assert harness.REQUIRED_TAGS[-1] == "Episode_Termination/time_out"
    assert harness.TRAIN_DIRECTORY == "canonical_train"
    assert harness.EXPORT_DIRECTORY == "exact_export"
    assert harness.AGGREGATE_FILENAME == "aggregate_result.json"
    source = HARNESS.read_text(encoding="utf-8")
    assert "from openhomie_isaaclab.workflows.elf3_run import" in source
    assert "isaaclab.app" not in source and "pxr" not in source
