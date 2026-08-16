from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab"
SIM_MODULE = PACKAGE / "workflows/elf3_sim.py"
HARNESS = ROOT / "isaaclab_ext/scripts/check_elf3_m5_headless.py"
TASK_ID = "OpenHomie-Elf3-Homie-Direct-v0"
AGENT_ENTRY = (
    "openhomie_isaaclab.tasks.locomotion.elf3.agents."
    "him_ppo_cfg:Elf3HIMRunnerCfg"
)


def require_batch_b():
    assert SIM_MODULE.is_file(), (
        "missing M5 Batch B production: "
        "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/"
        "workflows/elf3_sim.py"
    )


def _source() -> str:
    return SIM_MODULE.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source(), filename=str(SIM_MODULE))


def _names(tree: ast.AST) -> set[str]:
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            values.add(node.id)
        elif isinstance(node, ast.Attribute):
            values.add(node.attr)
        elif isinstance(node, ast.alias):
            values.add(node.asname or node.name.rsplit(".", 1)[-1])
    return values


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _load_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_elf3_m5_headless_harness", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_training() -> dict[str, bool]:
    return {
        "observations": True,
        "actions": True,
        "rewards": True,
        "losses": True,
        "learning_rates": True,
        "entropy": True,
        "estimator_metrics": True,
        "checkpoint_values": True,
    }


def _identity_evidence(harness: ModuleType) -> dict:
    env_cfg = {"scene": {"num_envs": 16}}
    agent_cfg = {"seed": 42}
    m4_files = {
        path: _sha256(harness.REPO_ROOT / path) for path in harness.M4_PATHS
    }
    return {
        "git": {"commit": "1" * 40, "dirty_paths": []},
        "configs": {
            "env": env_cfg,
            "agent": agent_cfg,
            "sha256": {
                "env": harness._json_sha256(env_cfg),
                "agent": harness._json_sha256(agent_cfg),
            },
        },
        "assets": {
            "urdf_sha256": _sha256(harness.URDF_PATH),
            "usd_sha256": _sha256(harness.USD_PATH),
        },
        "m4_sources": {
            "files": m4_files,
            "sha256": harness._json_sha256(m4_files),
        },
        "runtime": {
            "python": "3.11",
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
            "cuda_version": "13.1",
            "total_mib": 32607,
            "free_mib": 8192,
            "capability": [12, 0],
            "arch_list": ["sm_120"],
            "cuda_probe_passed": True,
        },
    }


def _manifest(
    harness: ModuleType,
    command: str,
    num_envs: int,
    iterations: int | None,
) -> dict:
    payload = {
        "schema_version": 1,
        "command": command,
        "task_id": TASK_ID,
        "seed": 42,
        "device": "cuda:0",
        "num_envs": num_envs,
        "iterations": iterations,
    }
    payload.update(_identity_evidence(harness))
    return payload


def _complete_evidence_tree(parent: Path, harness: ModuleType) -> Path:
    root = parent.resolve() / "m5-headless"
    root.mkdir(parents=True)
    train_ckpt = root / "train/checkpoint.pt"
    resume_ckpt = root / "resume/checkpoint.pt"
    train_ckpt.parent.mkdir()
    resume_ckpt.parent.mkdir()
    train_ckpt.write_bytes(b"train checkpoint iteration 2")
    resume_ckpt.write_bytes(b"resume checkpoint iteration 3")
    train_hash = _sha256(train_ckpt)
    resume_hash = _sha256(resume_ckpt)

    train_manifest = _manifest(harness, "train", 16, 2)
    train_manifest["start_iteration"] = 0
    _write_json(root / "train/manifest.json", train_manifest)
    _write_json(
        root / "train/result.json",
        {
            "status": "PASS",
            "start_iteration": 0,
            "final_iteration": 2,
            "finite": _finite_training(),
            "checkpoint": {
                "path": str(train_ckpt),
                "sha256": train_hash,
                "iteration": 2,
            },
        },
    )

    resume_manifest = _manifest(harness, "train", 16, 1)
    resume_manifest.update(
        {
            "start_iteration": 2,
            "parent": {
                "checkpoint_path": str(train_ckpt),
                "checkpoint_sha256": train_hash,
                "manifest_sha256": _sha256(root / "train/manifest.json"),
                "iteration": 2,
            },
        }
    )
    _write_json(root / "resume/manifest.json", resume_manifest)
    _write_json(
        root / "resume/result.json",
        {
            "status": "PASS",
            "start_iteration": 2,
            "final_iteration": 3,
            "finite": _finite_training(),
            "checkpoint": {
                "path": str(resume_ckpt),
                "sha256": resume_hash,
                "iteration": 3,
            },
        },
    )

    checkpoint_identity = {
        "path": str(resume_ckpt),
        "sha256": resume_hash,
        "iteration": 3,
    }
    play_manifest = _manifest(harness, "play", 16, None)
    play_manifest["checkpoint"] = checkpoint_identity
    _write_json(root / "play/manifest.json", play_manifest)
    _write_json(
        root / "play/result.json",
        {
            "status": "PASS",
            "play": {
                "steps": 100,
                "action_shape": [16, 12],
                "finite": True,
                "deterministic": True,
                "max_abs_diff": 0.0,
                "action_sha256": "a" * 64,
                "sha256_before": resume_hash,
                "sha256_after": resume_hash,
            },
        },
    )

    export_manifest = _manifest(harness, "export", 1, None)
    export_manifest["checkpoint"] = checkpoint_identity
    _write_json(root / "export/manifest.json", export_manifest)
    ts_path = root / "export/policy.ts"
    onnx_path = root / "export/policy.onnx"
    ts_path.write_bytes(b"torchscript artifact")
    onnx_path.write_bytes(b"onnx artifact")
    _write_json(
        root / "export/result.json",
        {
            "status": "PASS",
            "exports": {
                "torchscript": {
                    "path": str(ts_path),
                    "sha256": _sha256(ts_path),
                    "provider": "torch.jit.load",
                    "fresh_runtime": True,
                    "batches": [1, 4],
                    "max_abs_error": 1e-8,
                },
                "onnx": {
                    "path": str(onnx_path),
                    "sha256": _sha256(onnx_path),
                    "providers": ["CPUExecutionProvider"],
                    "checker_passed": True,
                    "fresh_runtime": True,
                    "batches": [1, 4],
                    "max_abs_error": 1e-6,
                },
            },
        },
    )

    python = str(Path(sys.executable).resolve())
    script = str((HARNESS.parent / "elf3_him.py").resolve())
    commands = harness._expected_commands(
        root, python, script, str(train_ckpt), str(resume_ckpt)
    )
    records = []
    for name in ("train", "resume", "play", "export"):
        log = root / "logs" / f"{name}.log"
        log.parent.mkdir(exist_ok=True)
        log.write_text(
            "child output\nM5_HEADLESS_PASS\nM5_INTERNAL_EXIT_CODE=0\n"
            f"{harness.APP_CLOSE_MARKER}\n",
            encoding="utf-8",
        )
        records.append(
            {
                "name": name,
                "returncode": 0,
                "command": commands[name],
                "log_path": str(log),
            }
        )
    _write_json(root / "children.json", records)
    return root


def _mutate_json(path: Path, mutate) -> None:
    payload = _read_json(path)
    mutate(payload)
    _write_json(path, payload)


def test_batch_b_module_is_static_auditable_and_post_launch_only():
    require_batch_b()
    source = _source()
    tree = _tree()
    ast.parse(source, filename=str(SIM_MODULE))
    assert "/home/" not in source and "wang-sm" not in source
    assert "/home/user/wang-sm/HOMIE" not in source
    assert "AppLauncher" not in _names(tree) and "isaaclab.app" not in source
    assert "pxr" not in source
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert {"gymnasium", "torch", "isaaclab_rl"} <= imported_roots
    integers = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    }
    assert not integers.intersection({80, 83, 480})


def test_run_dispatches_only_train_play_and_export():
    require_batch_b()
    tree = _tree()
    functions = {
        node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "run" in functions
    run = functions["run"]
    assert run.args.args and run.args.args[0].arg in {"request", "req"}
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {"train", "play", "export", "PASS", "FAIL"} <= literals
    assert {"train", "play", "export"} <= set(re.findall(r"\b(?:train|play|export)\b", _source()))
    assert any(
        isinstance(node, ast.Raise) for node in ast.walk(run)
    ), "unknown commands must fail instead of falling through"


def test_registry_config_final_wrapper_and_bool_done_adapter_are_explicit():
    require_batch_b()
    source = _source()
    tree = _tree()
    names = _names(tree)
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert TASK_ID in literals and AGENT_ENTRY in source
    assert "RslRlVecEnvWrapper" in names and "gym" in names and "make" in names
    wrapper_lines = [
        call.lineno for call in _calls(tree) if _call_name(call) == "RslRlVecEnvWrapper"
    ]
    runner_lines = [
        call.lineno for call in _calls(tree) if _call_name(call) == "HIMOnPolicyRunner"
    ]
    assert wrapper_lines and runner_lines and min(wrapper_lines) < min(runner_lines)
    assert re.search(r"(?:dones?|terminated|truncated).{0,120}(?:\.bool\(|torch\.bool|dtype\s*=\s*torch\.bool)", source, re.DOTALL), (
        "Isaac Lab 2.3.2 returns long dones; the final RslRlVecEnvWrapper "
        "boundary must adapt them to bool before HIMPPO"
    )


def test_environment_and_agent_overrides_use_the_explicit_request():
    require_batch_b()
    source = _source()
    tree = _tree()
    names = _names(tree)
    assert {"num_envs", "seed", "device", "clip_actions"} <= names
    assert "scene" in names and "cfg" in names
    assert "rsl_rl_cfg_entry_point" in source
    assert any(
        _call_name(call) in {"load_cfg_from_registry", "parse_env_cfg"}
        for call in _calls(tree)
    )
    assigned_attributes = {
        target.attr
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Attribute)
    }
    assert {"num_envs", "seed", "device"} <= assigned_attributes


def test_him_runner_uses_registry_cfg_final_env_log_dir_and_device():
    require_batch_b()
    source = _source()
    tree = _tree()
    names = _names(tree)
    assert "HIMOnPolicyRunner" in names
    calls = [call for call in _calls(tree) if _call_name(call) == "HIMOnPolicyRunner"]
    assert calls
    assert any(
        {keyword.arg for keyword in call.keywords if keyword.arg}
        >= {"log_dir", "device"}
        for call in calls
    )
    assert "to_dict" in names or "asdict" in names
    assert "current_learning_iteration" in names
    assert "RslRlVecEnvWrapper" in source


def test_fresh_train_writes_finite_two_iteration_checkpoint_evidence():
    require_batch_b()
    source = _source()
    tree = _tree()
    names = _names(tree)
    required = {
        "create_run_directory",
        "write_json_once",
        "validate_manifest",
        "learn",
        "get_training_metrics",
        "sha256_file",
    }
    assert required <= names
    assert {"manifest.json", "result.json"} <= {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "final_iteration" in source and "start_iteration" in source
    assert "isfinite" in source or "_require_finite" in source
    assert re.search(r"checkpoint", source, re.IGNORECASE)


def test_resume_loads_optimizers_and_enforces_two_to_three_lineage():
    require_batch_b()
    source = _source()
    tree = _tree()
    names = _names(tree)
    assert {"validate_checkpoint_payload", "final_iteration", "load", "learn"} <= names
    load_calls = [call for call in _calls(tree) if _call_name(call) == "load"]
    assert any(
        any(
            keyword.arg in {"load_optimizer", "load_optimizers"}
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
        for call in load_calls
    )
    assert all(token in source for token in ("parent", "manifest_sha256", "checkpoint_sha256"))
    assert "current_learning_iteration" in names


def test_play_is_fixed_100_step_finite_bitwise_deterministic_and_immutable():
    require_batch_b()
    source = _source()
    tree = _tree()
    names = _names(tree)
    integers = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    }
    assert 100 in integers
    assert {"get_inference_policy", "inference_mode", "sha256_file"} <= names
    assert "NUM_POLICY_ACTIONS" in source
    assert "isfinite" in names
    assert "equal" in names or "array_equal" in names
    assert all(token in source for token in ("action_shape", "max_abs_diff", "sha256_before", "sha256_after"))
    assert "play_steps" not in source and "--play-steps" not in source


def test_export_uses_same_checkpoint_and_fixed_fresh_runtime_parity():
    require_batch_b()
    source = _source()
    tree = _tree()
    names = _names(tree)
    required = {
        "export_him_policy_torchscript",
        "export_him_policy_onnx",
        "verify_him_policy_torchscript",
        "verify_him_policy_onnx",
        "sha256_file",
    }
    assert required <= names
    floats = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, float)
    }
    assert 1e-7 in floats and 1e-5 in floats
    assert re.search(r"\b1\b.{0,120}\b4\b", source, re.DOTALL)
    assert all(
        token in source
        for token in (
            "policy.ts",
            "policy.onnx",
            "CPUExecutionProvider",
            "checker_passed",
            "fresh_runtime",
        )
    )


def test_harness_constants_cli_and_four_child_commands_are_frozen(tmp_path):
    require_batch_b()
    harness = _load_harness()
    expected = {
        "NUM_ENVS": 16,
        "SEED": 42,
        "TRAIN_ITERATIONS": 2,
        "RESUME_ITERATIONS": 1,
        "PLAY_STEPS": 100,
        "EXPORT_NUM_ENVS": 1,
        "MIN_FREE_GPU_MIB": 4096,
        "TS_MAX_ERROR": 1e-7,
        "ONNX_MAX_ERROR": 1e-5,
    }
    assert {name: getattr(harness, name) for name in expected} == expected
    for name in ("gpu_preflight", "run_child", "load_json", "verify_headless_evidence", "main"):
        assert callable(getattr(harness, name))
    harness_tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    assert "AppLauncher" not in _names(harness_tree)
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            any(alias.name.split(".")[0] in {"isaaclab", "pxr"} for alias in node.names)
            if isinstance(node, ast.Import)
            else bool(node.module and node.module.split(".")[0] in {"isaaclab", "pxr"})
        )
        for node in ast.walk(harness_tree)
    )
    root = (tmp_path / "new-root").resolve()
    with pytest.raises((ValueError, SystemExit)):
        harness._validate_cli(["--run-root", str(root), "--device", "cuda:0"])
    with pytest.raises(ValueError):
        harness._validate_cli(["--run-root", "relative", "--device", "cuda:0", "--headless"])
    with pytest.raises(ValueError):
        harness._validate_cli(["--run-root", str(root), "--device", "cpu", "--headless"])
    commands = harness._expected_commands(
        root,
        str(Path(sys.executable).resolve()),
        str((HARNESS.parent / "elf3_him.py").resolve()),
        str((root / "train/checkpoint.pt").resolve()),
        str((root / "resume/checkpoint.pt").resolve()),
    )
    assert tuple(commands) == ("train", "resume", "play", "export")
    assert "--play-steps" not in commands["play"]
    assert commands["play"][commands["play"].index("--num-envs") + 1] == "16"
    assert commands["export"][commands["export"].index("--num-envs") + 1] == "1"
    assert all("--headless" in command for command in commands.values())
    assert all(Path(command[0]).is_absolute() for command in commands.values())


def test_harness_preflight_log_boundary_and_failure_reporting(
    tmp_path, monkeypatch, capsys
):
    require_batch_b()
    harness = _load_harness()

    class FakeTensor:
        def square(self):
            return self

        def sum(self):
            return self

        def item(self):
            return 32.0

    fake_torch = ModuleType("torch")
    fake_torch.version = SimpleNamespace(cuda="12.8")
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda index: (12, 0),
        get_arch_list=lambda: ["sm_120"],
    )
    fake_torch.ones = lambda *args, **kwargs: FakeTensor()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def fake_run(argv, **kwargs):
        if argv[0] == "nvidia-smi":
            return SimpleNamespace(
                returncode=0,
                stdout="0, NVIDIA RTX 5090, 590.48.01, 32607, 8192\n",
                stderr="",
            )
        return SimpleNamespace(
            returncode=0,
            stdout="M5_HEADLESS_PASS\nM5_INTERNAL_EXIT_CODE=0\n",
        )

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    evidence = harness.gpu_preflight("cuda:0")
    assert evidence["capability"] == [12, 0] and evidence["free_mib"] == 8192
    assert evidence["driver_version"] == "590.48.01"
    assert evidence["cuda_version"] == "12.8"
    log = tmp_path / "child.log"
    record = harness.run_child([str(Path(sys.executable).resolve()), "probe.py"], log)
    assert record["returncode"] == 0
    text = log.read_text(encoding="utf-8")
    assert text.index("M5_HEADLESS_PASS") < text.index(harness.APP_CLOSE_MARKER)

    monkeypatch.setattr(harness, "_validate_cli", lambda argv: (_ for _ in ()).throw(ValueError("bad")))
    assert harness.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out.count("M5_HEADLESS_FAIL") == 1
    assert captured.out.count("M5_INTERNAL_EXIT_CODE=1") == 1
    assert "M5_HEADLESS_PASS" not in captured.out


def test_offline_evidence_accepts_complete_tree_and_rejects_all_mutations(tmp_path):
    require_batch_b()
    harness = _load_harness()
    passing = _complete_evidence_tree(tmp_path / "passing", harness)
    result = harness.verify_headless_evidence(passing)
    assert result["status"] == "PASS"
    assert result["train_final_iteration"] == 2
    assert result["resume_final_iteration"] == 3

    def remove(path: str):
        return lambda root: (root / path).unlink()

    def change(path: str, mutate):
        return lambda root: _mutate_json(root / path, mutate)

    mutations = [
        remove("train/manifest.json"),
        change("train/result.json", lambda p: p.update(status="FAIL")),
        remove("resume/result.json"),
        change("play/result.json", lambda p: p["play"].update(max_abs_diff=math.nan)),
        change("resume/manifest.json", lambda p: p["parent"].update(iteration=1)),
        change("resume/manifest.json", lambda p: p["parent"].update(checkpoint_sha256="0" * 64)),
        change(
            "train/manifest.json",
            lambda p: p["configs"]["sha256"].update(env="0" * 64),
        ),
        change(
            "train/manifest.json",
            lambda p: p["assets"].update(urdf_sha256="0" * 64),
        ),
        change(
            "train/manifest.json",
            lambda p: p["m4_sources"]["files"].update(
                {next(iter(p["m4_sources"]["files"])): "0" * 64}
            ),
        ),
        change(
            "train/manifest.json",
            lambda p: p["runtime"].update(
                isaaclab_path="/opt/other/isaaclab.py"
            ),
        ),
        change(
            "train/manifest.json",
            lambda p: p["runtime"]["versions"].update(torch="0.0"),
        ),
        change(
            "train/manifest.json",
            lambda p: p["gpu"].update(driver_version=""),
        ),
        remove("export/policy.onnx"),
        change("export/result.json", lambda p: p["exports"]["onnx"].update(max_abs_error=1e-4)),
        lambda root: (root / "resume/checkpoint.pt").write_bytes(b"mutated checkpoint"),
        lambda root: (root / "logs/play.log").write_text(
            f"M5_INTERNAL_EXIT_CODE=0\n{harness.APP_CLOSE_MARKER}\n",
            encoding="utf-8",
        ),
        lambda root: (root / "logs/export.log").write_text(
            f"{harness.APP_CLOSE_MARKER}\nM5_HEADLESS_PASS\nM5_INTERNAL_EXIT_CODE=0\n",
            encoding="utf-8",
        ),
        lambda root: (root / "logs/train.log").write_text(
            "SKIPPED\nM5_HEADLESS_PASS\nM5_INTERNAL_EXIT_CODE=0\n"
            f"{harness.APP_CLOSE_MARKER}\n",
            encoding="utf-8",
        ),
    ]
    for index, mutate in enumerate(mutations):
        root = _complete_evidence_tree(tmp_path / f"mutation-{index}", harness)
        mutate(root)
        with pytest.raises((KeyError, TypeError, ValueError, json.JSONDecodeError)):
            harness.verify_headless_evidence(root)
