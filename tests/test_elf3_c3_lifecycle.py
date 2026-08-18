from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab"
WORKFLOW = PACKAGE / "workflows/elf3_c3.py"
SCRIPT = ROOT / "isaaclab_ext/scripts/elf3_c3.py"
PLAN = ROOT / "docs/plans/elf3-m5-c3-continual-training-design.md"


def api():
    from openhomie_isaaclab.workflows import elf3_c3

    return elf3_c3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def m5_parent_manifest(*, stage: str = "S0", start: int = 2000, iterations: int = 2000):
    return {
        "schema_version": 1,
        "command": "train",
        "created_utc": "2026-08-18T00:00:00Z",
        "task_id": "OpenHomie-Elf3-Homie-Direct-v0",
        "seed": 42,
        "device": "cuda:0",
        "num_envs": 4096,
        "iterations": iterations,
        "start_iteration": start,
        "cli": {},
        "git": {"commit": "a" * 40, "dirty_paths": []},
        "configs": {"env": {"command_stage": stage}, "agent": {}, "sha256": {}},
        "assets": {},
        "m4_sources": {"files": {}},
        "runtime": {},
        "gpu": {"total_mib": 1, "free_mib": 1},
    }


def make_source_files(tmp_path: Path, manifest_payload=None):
    checkpoint = tmp_path / "model_4000.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(manifest_payload or m5_parent_manifest()), encoding="utf-8"
    )
    plan = tmp_path / "plan.md"
    plan.write_text("approved C3 plan\n", encoding="utf-8")
    return checkpoint, manifest, plan


def valid_argv(tmp_path: Path, *, manifest_payload=None, stage="V1"):
    checkpoint, manifest, plan = make_source_files(tmp_path, manifest_payload)
    argv = [
        "--stage",
        stage,
        "--run-dir",
        str(tmp_path / "run"),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-sha256",
        sha256(checkpoint),
        "--parent-manifest",
        str(manifest),
        "--parent-manifest-sha256",
        sha256(manifest),
        "--plan",
        str(plan),
        "--plan-sha256",
        sha256(plan),
        "--device",
        "cuda:0",
        "--seed",
        "42",
        "--num-envs",
        "4096",
        "--headless",
    ]
    return argv, checkpoint, manifest, plan


def test_surface_is_pure_python_and_cli_launches_before_runtime_imports():
    assert WORKFLOW.is_file() and SCRIPT.is_file()
    for path in (WORKFLOW, SCRIPT):
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        assert "/home/" not in source and "wang-sm" not in source

    source = str(PACKAGE.parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (source, env.get("PYTHONPATH"))))
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import openhomie_isaaclab.workflows.elf3_c3; "
                "assert not ({'torch','pxr','isaaclab','isaaclab_rl'} & set(sys.modules))"
            ),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert probe.returncode == 0, probe.stderr

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    launches = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "AppLauncher")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "AppLauncher")
        )
    ]
    runtime_imports = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        if any(
            name.split(".")[0] in {"torch", "gymnasium", "isaaclab_rl"}
            or "elf3_sim" in name
            for name in names
        ):
            runtime_imports.append(node.lineno)
    assert launches and runtime_imports and min(launches) < min(runtime_imports)


def test_fixed_stage_chain_and_transition_counts():
    m = api()
    assert m.STAGES == ("V1", "V2", "V3")
    actual = [
        (
            item.target_stage,
            item.parent_stage,
            item.parent_checkpoint_iteration,
            item.global_start,
            item.global_final,
        )
        for item in (m.transition_for(stage) for stage in m.STAGES)
    ]
    assert actual == [
        ("V1", "S0", 4000, 4000, 6000),
        ("V2", "V1", 2000, 6000, 8000),
        ("V3", "V2", 2000, 8000, 10000),
    ]
    identity = m.lifecycle_identity("V3")
    assert identity == {
        "stage": "V3",
        "local_iterations": {"start": 0, "final": 2000},
        "global_iterations": {"start": 8000, "final": 10000},
        "transitions": {
            "per_iteration": 4096 * 50,
            "stage": 4096 * 50 * 2000,
            "global_start": 4096 * 50 * 8000,
            "global_final": 4096 * 50 * 10000,
        },
    }
    with pytest.raises(ValueError):
        m.transition_for("S0")


def test_parse_request_binds_explicit_parent_and_plan_hashes(tmp_path):
    m = api()
    argv, checkpoint, manifest, plan = valid_argv(tmp_path)
    request = m.parse_request(argv)
    assert request.stage == "V1"
    assert request.command == "train" and request.iterations == 2000
    assert request.num_envs == 4096 and request.headless is True
    assert request.run_dir == (tmp_path / "run").resolve()
    assert request.checkpoint == checkpoint.resolve()
    assert request.parent_manifest == manifest.resolve()
    assert request.plan == plan.resolve()
    assert request.checkpoint_sha256 == sha256(checkpoint)
    assert request.parent_manifest_sha256 == sha256(manifest)
    assert request.plan_sha256 == sha256(plan)


@pytest.mark.parametrize(
    ("flag", "replacement"),
    [
        ("--checkpoint-sha256", "0" * 64),
        ("--parent-manifest-sha256", "0" * 64),
        ("--plan-sha256", "0" * 64),
        ("--checkpoint-sha256", "ABC"),
        ("--parent-manifest-sha256", "f" * 63),
        ("--plan-sha256", "g" * 64),
    ],
)
def test_parse_request_rejects_wrong_or_malformed_hashes(tmp_path, flag, replacement):
    m = api()
    argv, _, _, _ = valid_argv(tmp_path)
    argv[argv.index(flag) + 1] = replacement
    with pytest.raises((SystemExit, TypeError, ValueError)):
        m.parse_request(argv)


def test_parse_request_rejects_existing_run_and_noncanonical_budget(tmp_path):
    m = api()
    argv, _, _, _ = valid_argv(tmp_path)
    (tmp_path / "run").mkdir()
    with pytest.raises((FileExistsError, ValueError)):
        m.parse_request(argv)
    (tmp_path / "run").rmdir()
    argv[argv.index("--num-envs") + 1] = "2048"
    with pytest.raises((SystemExit, ValueError)):
        m.parse_request(argv)


@pytest.mark.parametrize("target", ("checkpoint", "manifest", "plan"))
def test_parse_request_rejects_symlinked_inputs(tmp_path, target):
    m = api()
    argv, checkpoint, manifest, plan = valid_argv(tmp_path)
    paths = {"checkpoint": checkpoint, "manifest": manifest, "plan": plan}
    flags = {
        "checkpoint": "--checkpoint",
        "manifest": "--parent-manifest",
        "plan": "--plan",
    }
    original = paths[target]
    link = tmp_path / f"{target}.link"
    link.symlink_to(original)
    argv[argv.index(flags[target]) + 1] = str(link)
    with pytest.raises(ValueError):
        m.parse_request(argv)


def test_parser_only_accepts_v_stages(tmp_path):
    m = api()
    for stage in ("S0", "S5", "V0", "V4", "v1"):
        case = tmp_path / stage
        case.mkdir()
        argv, _, _, _ = valid_argv(case, stage=stage)
        with pytest.raises(SystemExit):
            m.parse_request(argv)


def test_s0_bootstrap_parent_identity_is_fail_closed():
    m = api()
    identity = m.parent_stage_identity(
        m5_parent_manifest(), target_stage="V1", checkpoint_iteration=4000
    )
    assert identity == {
        "kind": "m5_s0_bootstrap",
        "stage": "S0",
        "local_iteration": 4000,
        "global_iteration": 4000,
    }
    mutations = [
        m5_parent_manifest(stage="S1"),
        m5_parent_manifest(start=0),
        m5_parent_manifest(iterations=1999),
    ]
    for payload in mutations:
        with pytest.raises((KeyError, TypeError, ValueError)):
            m.parent_stage_identity(
                payload, target_stage="V1", checkpoint_iteration=4000
            )
    with pytest.raises(ValueError):
        m.parent_stage_identity(
            m5_parent_manifest(), target_stage="V1", checkpoint_iteration=2000
        )
    with pytest.raises(ValueError):
        m.parent_stage_identity(
            m5_parent_manifest(), target_stage="V2", checkpoint_iteration=4000
        )


class FakeOptimizer:
    def __init__(self, lr=0.001, state=None):
        self.state = {} if state is None else state
        self.param_groups = [{"lr": lr}]


class FakeRunner:
    def __init__(
        self,
        checkpoint_iteration=4000,
        policy_lr=0.001,
        estimator_lr=0.001,
        policy_state=None,
        estimator_state=None,
    ):
        self.current_learning_iteration = 0
        self.checkpoint_iteration = checkpoint_iteration
        self.calls = []
        self.alg = SimpleNamespace(
            optimizer=FakeOptimizer(policy_lr, policy_state),
            estimator_optimizer=FakeOptimizer(estimator_lr, estimator_state),
            learning_rate=policy_lr,
            estimator_learning_rate=estimator_lr,
        )

    def load(self, path, *, load_optimizer, map_location):
        self.calls.append((path, load_optimizer, map_location))
        self.current_learning_iteration = self.checkpoint_iteration


def test_weights_only_load_resets_local_iteration_and_proves_fresh_state(tmp_path):
    m = api()
    checkpoint = tmp_path / "model_4000.pt"
    checkpoint.write_bytes(b"checkpoint")
    runner = FakeRunner()
    contract = m.initialize_weights_only(
        runner,
        checkpoint=checkpoint,
        device="cuda:0",
        expected_checkpoint_iteration=4000,
    )
    assert runner.calls == [(str(checkpoint), False, "cuda:0")]
    assert runner.current_learning_iteration == 0
    assert contract == {
        "mode": "weights_only",
        "load_optimizer": False,
        "checkpoint_iteration": 4000,
        "local_iteration_after_load": 0,
        "optimizer_state_entries": {"policy": 0, "estimator": 0},
        "learning_rates": {"policy": 0.001, "estimator": 0.001},
    }


@pytest.mark.parametrize(
    "runner",
    [
        FakeRunner(checkpoint_iteration=3999),
        FakeRunner(policy_lr=0.0009),
        FakeRunner(estimator_lr=0.0009),
        FakeRunner(policy_state={"p": {}}),
        FakeRunner(estimator_state={"p": {}}),
    ],
)
def test_weights_only_load_rejects_iteration_optimizer_or_lr_drift(tmp_path, runner):
    m = api()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(RuntimeError):
        m.initialize_weights_only(
            runner,
            checkpoint=checkpoint,
            device="cuda:0",
            expected_checkpoint_iteration=4000,
        )


def manifest_fixture(tmp_path: Path):
    m = api()
    argv, checkpoint, parent_manifest, plan = valid_argv(tmp_path)
    request = m.parse_request(argv)
    parent = {
        **m.parent_stage_identity(
            m5_parent_manifest(), target_stage="V1", checkpoint_iteration=4000
        ),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "manifest_path": str(parent_manifest.resolve()),
        "manifest_sha256": sha256(parent_manifest),
    }
    provenance = {
        "git": {"commit": "a" * 40, "dirty_paths": []},
        "configs": {"env": {"command_stage": "V1"}, "agent": {}, "sha256": {}},
        "assets": {},
        "m4_sources": {"files": {}},
        "c3_sources": {"files": {}, "sha256": "b" * 64},
        "runtime": {},
        "gpu": {},
    }
    payload = m.build_manifest(
        request,
        parent=parent,
        provenance=provenance,
        stage_spec={
            "name": "V1",
            "walk_height": 1.01,
            "lin_vel_x": [-0.4, 0.6],
            "lin_vel_y": [-0.2, 0.2],
            "ang_vel_yaw": [-0.4, 0.4],
        },
        created_utc="2026-08-18T00:00:00Z",
    )
    return request, payload


def test_manifest_is_exact_self_consistent_and_rejects_mutations(tmp_path):
    m = api()
    _, payload = manifest_fixture(tmp_path)
    assert set(payload) == {
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
    }
    m.validate_c3_manifest(payload)
    mutations = []
    missing = deepcopy(payload)
    missing.pop("parent")
    mutations.append(missing)
    extra = deepcopy(payload)
    extra["unexpected"] = True
    mutations.append(extra)
    wrong_stage = deepcopy(payload)
    wrong_stage["stage"] = "V2"
    mutations.append(wrong_stage)
    wrong_global = deepcopy(payload)
    wrong_global["lifecycle"]["global_iterations"]["final"] = 6001
    mutations.append(wrong_global)
    wrong_parent = deepcopy(payload)
    wrong_parent["parent"]["global_iteration"] = 3999
    mutations.append(wrong_parent)
    wrong_lr = deepcopy(payload)
    wrong_lr["load_contract"]["learning_rates"]["policy"] = 0.0009
    mutations.append(wrong_lr)
    for mutation in mutations:
        with pytest.raises((KeyError, TypeError, ValueError)):
            m.validate_c3_manifest(mutation)


def test_c3_parent_requires_exact_previous_stage_local_and_global_identity(tmp_path):
    m = api()
    _, v1 = manifest_fixture(tmp_path)
    identity = m.parent_stage_identity(
        v1, target_stage="V2", checkpoint_iteration=2000
    )
    assert identity == {
        "kind": "c3",
        "stage": "V1",
        "local_iteration": 2000,
        "global_iteration": 6000,
    }
    for field, value in (("stage", "V2"), ("global", 6001), ("local", 1999)):
        bad = deepcopy(v1)
        if field == "stage":
            bad["stage"] = value
        elif field == "global":
            bad["lifecycle"]["global_iterations"]["final"] = value
        else:
            bad["lifecycle"]["local_iterations"]["final"] = value
        with pytest.raises((KeyError, TypeError, ValueError)):
            m.parent_stage_identity(
                bad, target_stage="V2", checkpoint_iteration=2000
            )


def test_pass_result_validator_binds_output_and_transition_identity(tmp_path):
    m = api()
    request, manifest = manifest_fixture(tmp_path)
    checkpoint = tmp_path / "run/model_2000.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"trained")
    load_contract = {
        "mode": "weights_only",
        "load_optimizer": False,
        "checkpoint_iteration": 4000,
        "local_iteration_after_load": 0,
        "optimizer_state_entries": {"policy": 0, "estimator": 0},
        "learning_rates": {"policy": 0.001, "estimator": 0.001},
    }
    verification = {
        "checkpoint_sha256_before": request.checkpoint_sha256,
        "checkpoint_sha256_after": request.checkpoint_sha256,
        "manifest_sha256_before": request.parent_manifest_sha256,
        "manifest_sha256_after": request.parent_manifest_sha256,
        "plan_sha256_before": request.plan_sha256,
        "plan_sha256_after": request.plan_sha256,
    }
    result = m.build_pass_result(
        request,
        manifest=manifest,
        load_contract=load_contract,
        metrics={"surrogate": 1.0},
        checkpoint={
            "path": str(checkpoint.resolve()),
            "sha256": sha256(checkpoint),
            "iteration": 2000,
        },
        parent_verification=verification,
        elapsed_seconds=1.0,
    )
    m.validate_c3_result(result, manifest)
    assert result["checkpoint"]["local_iteration"] == 2000
    assert result["checkpoint"]["global_iteration"] == 6000
    assert result["lifecycle"] == manifest["lifecycle"]
    for mutation in (
        {**result, "unexpected": True},
        {**result, "stage": "V2"},
        {
            **result,
            "checkpoint": {**result["checkpoint"], "global_iteration": 6001},
        },
        {
            **result,
            "parent_verification": {
                **result["parent_verification"],
                "plan_sha256_after": "0" * 64,
            },
        },
    ):
        with pytest.raises((KeyError, TypeError, ValueError)):
            m.validate_c3_result(mutation, manifest)


def test_manifest_and_result_are_written_once(tmp_path):
    m = api()
    target = tmp_path / "manifest.json"
    m.write_immutable_json(target, {"status": "PASS"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "PASS"}
    with pytest.raises(FileExistsError):
        m.write_immutable_json(target, {"status": "FAIL"})
