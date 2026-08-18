"""Contracts for the isolated ELF3 M5 C3 Isaac Sim video worker."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab"
WORKFLOW = PACKAGE / "workflows/elf3_c3_video.py"
SCRIPT = ROOT / "isaaclab_ext/scripts/elf3_m5_c3_video.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_tree(tmp_path: Path) -> dict[str, Path | str]:
    evidence_root = tmp_path / "c3-evidence"
    source_run = evidence_root / "v3"
    source_run.mkdir(parents=True)
    checkpoint = source_run / "model_2000.pt"
    checkpoint.write_bytes(b"accepted V3 local-2000 global-10000 checkpoint")
    checkpoint_hash = _sha256(checkpoint)
    manifest = source_run / "manifest.json"
    result = source_run / "result.json"
    plan = tmp_path / "elf3-c3-design.md"
    plan.write_text("approved execution plan\n", encoding="utf-8")
    plan_hash = _sha256(plan)
    lifecycle = {
        "local_iterations": {"start": 0, "final": 2000},
        "global_iterations": {"start": 8000, "final": 10000},
    }
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "workflow": "elf3_c3_weights_only",
            "stage": "V3",
            "seed": 42,
            "num_envs": 4096,
            "iterations": 2000,
            "lifecycle": lifecycle,
            "plan": {
                "path": str(plan.resolve()),
                "sha256": plan_hash,
            },
            "configs": {
                "sha256": {"env": "a" * 64, "agent": "b" * 64}
            },
        },
    )
    _write_json(
        result,
        {
            "schema_version": 1,
            "workflow": "elf3_c3_weights_only",
            "status": "PASS",
            "stage": "V3",
            "lifecycle": lifecycle,
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": checkpoint_hash,
                "iteration": 2000,
                "stage": "V3",
                "local_iteration": 2000,
                "global_iteration": 10000,
            },
        },
    )
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_hash,
        "manifest": manifest,
        "result": result,
        "plan": plan,
        "plan_sha256": plan_hash,
        "output_root": tmp_path / "c3-video-evidence",
        "evidence_root": evidence_root,
    }


def _argv(paths: dict[str, Path | str]) -> list[str]:
    return [
        "--checkpoint",
        str(paths["checkpoint"]),
        "--output-root",
        str(paths["output_root"]),
        "--plan",
        str(paths["plan"]),
        "--plan-sha256",
        str(paths["plan_sha256"]),
    ]


def _api():
    assert WORKFLOW.is_file(), "missing isolated C3 video workflow"
    return importlib.import_module("openhomie_isaaclab.workflows.elf3_c3_video")


def _valid_metadata() -> dict[str, object]:
    api = _api()
    indices = list(api.SAMPLE_FRAME_INDICES)
    return {
        "relative_path": "videos/elf3-forward-seed42-model_2000-step-0.mp4",
        "sha256": "c" * 64,
        "size_bytes": 123456,
        "frame_count": 1000,
        "fps": 50.0,
        "width": 1280,
        "height": 720,
        "duration_seconds": 20.0,
        "sampled_frames": [
            {
                "index": index,
                "mean_luma": 80.0 + position,
                "nonblack_fraction": 0.75,
            }
            for position, index in enumerate(indices)
        ],
        "motion": {
            "pair_count": len(indices) - 1,
            "moving_pair_count": len(indices) - 1,
            "max_mean_abs_difference": 4.0,
        },
    }


def test_surface_is_new_pure_importable_and_machine_path_free():
    assert SCRIPT.is_file(), "missing isolated C3 video CLI"
    for path in (WORKFLOW, SCRIPT):
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        assert "/home/" not in source and "wang-sm" not in source

    source_root = str(PACKAGE.parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, env.get("PYTHONPATH")))
    )
    code = (
        "import sys; "
        "import openhomie_isaaclab.workflows.elf3_c3_video; "
        "assert not ({'gymnasium','torch','isaaclab','isaaclab_rl','pxr'} "
        "& set(sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_cli_is_fixed_forward_seed42_16_envs_1000_steps(tmp_path):
    module = _api()
    paths = _source_tree(tmp_path)
    request = module.parse_request(_argv(paths))
    assert request.checkpoint == Path(paths["checkpoint"]).resolve()
    assert request.output_root == Path(paths["output_root"]).resolve()
    assert request.plan == Path(paths["plan"]).resolve()
    assert request.plan_sha256 == paths["plan_sha256"]
    assert request.source_manifest == Path(paths["manifest"]).resolve()
    assert request.source_result == Path(paths["result"]).resolve()
    assert request.device == "cuda:0"
    assert request.headless is True
    assert request.enable_cameras is True
    assert request.scenario == "forward"
    assert request.seed == 42
    assert request.num_envs == 16
    assert request.steps == 1000
    assert request.fps == 50
    assert request.width == 1280 and request.height == 720

    for extra in (
        ["--seed", "43"],
        ["--scenario", "turn"],
        ["--num-envs", "1"],
        ["--steps", "10"],
        ["--device", "cpu"],
        ["--no-headless"],
    ):
        with pytest.raises((SystemExit, TypeError, ValueError)):
            module.parse_request(_argv(paths) + extra)


def test_cli_rejects_aliases_existing_or_c2_contained_output_and_bad_plan_hash(
    tmp_path,
):
    module = _api()
    paths = _source_tree(tmp_path)

    bad_hash = _argv(paths)
    bad_hash[-1] = "0" * 64
    with pytest.raises(ValueError, match="plan"):
        module.parse_request(bad_hash)

    output = Path(paths["output_root"])
    output.mkdir()
    with pytest.raises((FileExistsError, ValueError), match="output"):
        module.parse_request(_argv(paths))
    output.rmdir()

    paths["output_root"] = Path(paths["checkpoint"]).parent / "videos"
    with pytest.raises(ValueError, match="source V3 run"):
        module.parse_request(_argv(paths))

    paths = _source_tree(tmp_path / "symlink-case")
    link = tmp_path / "checkpoint-link.pt"
    link.symlink_to(paths["checkpoint"])
    paths["checkpoint"] = link
    with pytest.raises(ValueError, match="checkpoint"):
        module.parse_request(_argv(paths))


def test_source_binding_is_hash_bound_and_rejects_mismatched_evidence(tmp_path):
    module = _api()
    paths = _source_tree(tmp_path)
    request = module.parse_request(_argv(paths))
    binding = module.build_source_binding(request)
    assert binding["checkpoint"] == {
        "path": str(Path(paths["checkpoint"]).resolve()),
        "sha256": paths["checkpoint_sha256"],
        "iteration": 2000,
    }
    assert binding["plan"]["sha256"] == paths["plan_sha256"]
    assert binding["manifest"]["sha256"] == _sha256(Path(paths["manifest"]))
    assert binding["result"]["sha256"] == _sha256(Path(paths["result"]))
    assert binding["training"] == {
        "stage": "V3",
        "seed": 42,
        "num_envs": 4096,
        "local_start_iteration": 0,
        "local_final_iteration": 2000,
        "global_start_iteration": 8000,
        "global_final_iteration": 10000,
        "config_sha256": {"env": "a" * 64, "agent": "b" * 64},
    }
    module.verify_source_binding(binding)

    Path(paths["plan"]).write_text("mutated plan\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="plan"):
        module.verify_source_binding(binding)

    paths = _source_tree(tmp_path / "wrong-result")
    result = json.loads(Path(paths["result"]).read_text(encoding="utf-8"))
    result["checkpoint"]["sha256"] = "f" * 64
    _write_json(Path(paths["result"]), result)
    request = module.parse_request(_argv(paths))
    with pytest.raises(ValueError, match="checkpoint"):
        module.build_source_binding(request)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(frame_count=999), "frame_count"),
        (lambda value: value.update(fps=49.0), "fps"),
        (lambda value: value.update(width=1279), "width"),
        (lambda value: value.update(height=721), "height"),
        (lambda value: value.update(duration_seconds=19.5), "duration"),
        (lambda value: value.update(size_bytes=0), "size"),
        (lambda value: value.update(sha256="A" * 64), "SHA"),
        (
            lambda value: value["sampled_frames"][0].update(mean_luma=0.0),
            "black",
        ),
        (
            lambda value: value["sampled_frames"][0].update(
                nonblack_fraction=0.0
            ),
            "black",
        ),
        (
            lambda value: value["motion"].update(
                moving_pair_count=0, max_mean_abs_difference=0.0
            ),
            "motion",
        ),
    ],
)
def test_video_metadata_validation_fails_closed(mutate, message):
    module = _api()
    metadata = _valid_metadata()
    mutate(metadata)
    with pytest.raises((KeyError, TypeError, ValueError), match=message):
        module.validate_video_metadata(metadata)


def test_video_metadata_accepts_exact_20_second_nonblack_moving_clip():
    module = _api()
    metadata = _valid_metadata()
    assert module.validate_video_metadata(metadata) == metadata


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def test_app_launch_and_render_wrapper_order_are_static_auditable():
    workflow_source = WORKFLOW.read_text(encoding="utf-8")
    workflow_tree = ast.parse(workflow_source, filename=str(WORKFLOW))
    script_tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))

    forbidden = {"gymnasium", "torch", "isaaclab", "isaaclab_rl", "cv2", "numpy"}
    for node in workflow_tree.body:
        if isinstance(node, ast.Import):
            assert not forbidden.intersection(
                alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden

    launch_calls = [
        node
        for node in ast.walk(script_tree)
        if isinstance(node, ast.Call) and _call_name(node) == "AppLauncher"
    ]
    assert len(launch_calls) == 1
    launch_keywords = {keyword.arg: keyword.value for keyword in launch_calls[0].keywords}
    assert isinstance(launch_keywords["headless"], ast.Constant)
    assert launch_keywords["headless"].value is True
    assert isinstance(launch_keywords["enable_cameras"], ast.Constant)
    assert launch_keywords["enable_cameras"].value is True
    assert "device" in launch_keywords

    calls = [node for node in ast.walk(workflow_tree) if isinstance(node, ast.Call)]
    gym_make = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "gym"
        and node.func.attr == "make"
    ]
    record_video = [node for node in calls if _call_name(node) == "RecordVideo"]
    wrappers = [node for node in calls if _call_name(node) == "RslRlVecEnvWrapper"]
    runners = [node for node in calls if _call_name(node) == "HIMOnPolicyRunner"]
    assert len(gym_make) == len(record_video) == len(wrappers) == len(runners) == 1
    assert gym_make[0].lineno < record_video[0].lineno < wrappers[0].lineno < runners[0].lineno

    recorded_runner = next(
        node
        for node in workflow_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_create_recorded_runner"
    )
    runner_calls = [
        node for node in ast.walk(recorded_runner) if isinstance(node, ast.Call)
    ]
    resets = [
        node
        for node in runner_calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "raw_env"
        and node.func.attr == "reset"
    ]
    warmups = [node for node in runner_calls if _call_name(node) == "_warm_up_renderer"]
    assert len(resets) == len(warmups) == 1
    assert resets[0].lineno < warmups[0].lineno < record_video[0].lineno

    run = next(
        node
        for node in workflow_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    run_calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
    play_calls = [node for node in run_calls if _call_name(node) == "_play_result"]
    tail_calls = [
        node
        for node in run_calls
        if _call_name(node) == "_append_video_encoder_tail_frame"
    ]
    assert len(play_calls) == len(tail_calls) == 1
    assert play_calls[0].lineno < tail_calls[0].lineno

    make_keywords = {keyword.arg: keyword.value for keyword in gym_make[0].keywords}
    assert isinstance(make_keywords["render_mode"], ast.Constant)
    assert make_keywords["render_mode"].value == "rgb_array"
    record_keywords = {keyword.arg for keyword in record_video[0].keywords}
    assert {
        "video_folder",
        "step_trigger",
        "video_length",
        "name_prefix",
        "fps",
        "disable_logger",
    } <= record_keywords

    literals = {
        node.value
        for node in ast.walk(workflow_tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {
        "asset_root",
        "robot",
        "rgb_array",
        "videos",
        "manifest.json",
        "result.json",
        "action_sha256",
        "trajectory_sha256",
        "sha256_before",
        "sha256_after",
    } <= literals

    stage_assignments = [
        node
        for node in ast.walk(workflow_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "command_stage"
            for target in node.targets
        )
    ]
    assert any(
        isinstance(node.value, ast.Name) and node.value.id == "C3_FINAL_STAGE"
        for node in stage_assignments
    )
