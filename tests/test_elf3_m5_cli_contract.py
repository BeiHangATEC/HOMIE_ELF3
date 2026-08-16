from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab"
SCRIPT = ROOT / "isaaclab_ext/scripts/elf3_him.py"
WORKFLOWS_INIT = PACKAGE / "workflows/__init__.py"
RUN_MODULE = PACKAGE / "workflows/elf3_run.py"
TASK_INIT = PACKAGE / "tasks/locomotion/elf3/__init__.py"
TASK_ID = "OpenHomie-Elf3-Homie-Direct-v0"
AGENT_ENTRY = (
    "openhomie_isaaclab.tasks.locomotion.elf3.agents."
    "him_ppo_cfg:Elf3HIMRunnerCfg"
)
PRODUCTION = (SCRIPT, WORKFLOWS_INIT, RUN_MODULE)


def require_batch_a():
    missing = [
        str(path.relative_to(ROOT))
        for path in PRODUCTION
        if not path.is_file()
    ]
    task_text = TASK_INIT.read_text(encoding="utf-8")
    if AGENT_ENTRY not in task_text:
        missing.append("tasks/locomotion/elf3/__init__.py:rsl_rl_cfg_entry_point")
    assert not missing, f"missing M5 Batch A production: {missing}"


def test_batch_a_surface_parses_has_no_reference_leaks_and_orders_app_launch():
    require_batch_a()
    for path in (*PRODUCTION, TASK_INIT):
        text = path.read_text(encoding="utf-8")
        ast.parse(text, filename=str(path))
        assert "/home/" not in text and "wang-sm" not in text
        integers = {
            n.value
            for n in ast.walk(ast.parse(text))
            if isinstance(n, ast.Constant)
            and isinstance(n.value, int)
            and not isinstance(n.value, bool)
        }
        assert not integers.intersection({80, 83, 480}), path
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    literals = {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert {"train", "play", "export"} <= literals
    launches = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == "AppLauncher")
            or (
                isinstance(n.func, ast.Attribute)
                and n.func.attr == "AppLauncher"
            )
        )
    ]
    assert launches
    runtime_lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                a.name.split(".")[0]
                in {"gymnasium", "torch", "isaaclab_rl"}
                or "elf3_sim" in a.name
                for a in node.names
            ):
                runtime_lines.append(node.lineno)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (
                node.module.split(".")[0]
                in {"gymnasium", "torch", "isaaclab_rl"}
                or "elf3_sim" in node.module
            )
        ):
            runtime_lines.append(node.lineno)
    assert runtime_lines and min(launches) < min(runtime_lines)


def test_registry_preserves_env_and_resolves_agent_entry_point():
    require_batch_a()
    import openhomie_isaaclab.tasks.locomotion.elf3  # noqa: F401
    spec = gym.spec(TASK_ID)
    assert spec.entry_point == (
        "openhomie_isaaclab.tasks.locomotion.elf3."
        "elf3_homie_env:Elf3HomieEnv"
    )
    assert spec.kwargs["env_cfg_entry_point"] == (
        "openhomie_isaaclab.tasks.locomotion.elf3."
        "elf3_homie_env_cfg:Elf3HomieEnvCfg"
    )
    assert spec.kwargs["rsl_rl_cfg_entry_point"] == AGENT_ENTRY
    module, name = AGENT_ENTRY.split(":")
    assert getattr(importlib.import_module(module), name).__name__ == name


def test_pure_workflow_import_does_not_load_simulator_modules():
    require_batch_a()
    source = str(PACKAGE.parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source, env.get("PYTHONPATH")))
    )
    code = (
        "import sys; import openhomie_isaaclab.workflows.elf3_run; "
        "assert not ({'pxr','isaaclab','isaaclab_rl'} & set(sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            [
                "train",
                "--run-dir",
                "run",
                "--device",
                "cpu",
                "--seed",
                "7",
                "--num-envs",
                "16",
                "--iterations",
                "2",
            ],
            ("train", False, None, 2),
        ),
        (
            [
                "train",
                "--resume",
                "--checkpoint",
                "model.pt",
                "--run-dir",
                "run",
                "--device",
                "cuda:0",
                "--seed",
                "42",
                "--num-envs",
                "16",
                "--iterations",
                "1",
            ],
            ("train", True, "model.pt", 1),
        ),
        (
            [
                "play",
                "--checkpoint",
                "model.pt",
                "--run-dir",
                "run",
                "--device",
                "cpu",
                "--seed",
                "1",
                "--num-envs",
                "1",
            ],
            ("play", False, "model.pt", None),
        ),
        (
            [
                "export",
                "--checkpoint",
                "model.pt",
                "--run-dir",
                "run",
                "--device",
                "cpu",
                "--seed",
                "1",
                "--num-envs",
                "1",
            ],
            ("export", False, "model.pt", None),
        ),
    ],
)
def test_parse_request_accepts_explicit_valid_commands(tmp_path, argv, expected):
    require_batch_a()
    from openhomie_isaaclab.workflows.elf3_run import parse_request

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    values = [
        (
            str(tmp_path / "run")
            if v == "run"
            else str(checkpoint)
            if v == "model.pt"
            else v
        )
        for v in argv
    ]
    req = parse_request(values)
    assert (
        req.command,
        req.resume,
        None if req.checkpoint is None else req.checkpoint.name,
        req.iterations,
    ) == expected
    assert (
        req.run_dir == (tmp_path / "run").resolve()
        and req.seed >= 0
        and req.num_envs > 0
    )


@pytest.mark.parametrize(
    "argv",
    [
        [
            "train",
            "--run-dir",
            "run",
            "--device",
            "cpu",
            "--seed",
            "0",
            "--num-envs",
            "0",
            "--iterations",
            "1",
        ],
        [
            "train",
            "--run-dir",
            "run",
            "--device",
            "cpu",
            "--seed",
            "0",
            "--num-envs",
            "-1",
            "--iterations",
            "1",
        ],
        [
            "train",
            "--run-dir",
            "run",
            "--device",
            "cpu",
            "--seed",
            "0",
            "--num-envs",
            "1",
            "--iterations",
            "0",
        ],
        [
            "train",
            "--run-dir",
            "run",
            "--device",
            "cpu",
            "--seed",
            "0",
            "--num-envs",
            "1",
            "--iterations",
            "-1",
        ],
        [
            "train",
            "--run-dir",
            "run",
            "--device",
            "cpu",
            "--seed",
            "-1",
            "--num-envs",
            "1",
            "--iterations",
            "1",
        ],
        [
            "train",
            "--run-dir",
            "run",
            "--device",
            "cuda:x",
            "--seed",
            "0",
            "--num-envs",
            "1",
            "--iterations",
            "1",
        ],
        [
            "train",
            "--checkpoint",
            "model.pt",
            "--run-dir",
            "run",
            "--device",
            "cpu",
            "--seed",
            "0",
            "--num-envs",
            "1",
            "--iterations",
            "1",
        ],
        [
            "train",
            "--resume",
            "--run-dir",
            "run",
            "--device",
            "cpu",
            "--seed",
            "0",
            "--num-envs",
            "1",
            "--iterations",
            "1",
        ],
        [
            "play",
            "--run-dir",
            "run",
            "--device",
            "cpu",
            "--seed",
            "0",
            "--num-envs",
            "1",
        ],
        [
            "export",
            "--run-dir",
            "run",
            "--device",
            "cpu",
            "--seed",
            "0",
            "--num-envs",
            "1",
        ],
    ],
)
def test_parse_request_rejects_invalid_argument_matrix(tmp_path, argv):
    require_batch_a()
    from openhomie_isaaclab.workflows.elf3_run import parse_request

    values = [str(tmp_path / "run") if v == "run" else v for v in argv]
    with pytest.raises((ValueError, SystemExit)):
        parse_request(values)


def test_checkpoint_and_run_paths_are_explicit_and_exclusive(tmp_path):
    require_batch_a()
    from openhomie_isaaclab.workflows.elf3_run import parse_request, resolve_checkpoint

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"x")
    broken = tmp_path / "broken.pt"
    broken.symlink_to(tmp_path / "missing.pt")
    assert resolve_checkpoint(checkpoint) == checkpoint.resolve()
    for bad in ("latest", "*.pt", tmp_path, tmp_path / "missing.pt", broken):
        with pytest.raises(ValueError):
            resolve_checkpoint(bad)
    existing = tmp_path / "existing"
    existing.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(existing, target_is_directory=True)
    for run_dir in (existing, alias):
        values = [
            "play",
            "--checkpoint",
            str(checkpoint),
            "--run-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--seed",
            "0",
            "--num-envs",
            "1",
        ]
        with pytest.raises((FileExistsError, ValueError)):
            parse_request(values)
