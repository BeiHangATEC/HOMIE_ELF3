"""Red contracts for the approved M5 Batch C1 instrumentation surface."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab"
RUNNER = PACKAGE / "him_rl/runner.py"
ENV = PACKAGE / "tasks/locomotion/elf3/elf3_homie_env.py"
ENV_CFG = PACKAGE / "tasks/locomotion/elf3/elf3_homie_env_cfg.py"
REWARDS = PACKAGE / "tasks/locomotion/elf3/elf3_homie_rewards.py"
RUN = PACKAGE / "workflows/elf3_run.py"
SIM = PACKAGE / "workflows/elf3_sim.py"


def _source(path: Path) -> str:
    assert path.is_file(), f"missing production file: {path.relative_to(ROOT)}"
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
    return source


def _tree(path: Path) -> ast.Module:
    return ast.parse(_source(path), filename=str(path))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    assert len(matches) == 1, f"expected exactly one class {name}"
    return matches[0]


def _method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(matches) == 1, f"missing required method {class_node.name}.{name}"
    return matches[0]


def _segment(path: Path, node: ast.AST) -> str:
    segment = ast.get_source_segment(_source(path), node)
    assert segment is not None
    return segment


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _play_argv(tmp_path: Path) -> list[str]:
    checkpoint = tmp_path / "model_2000.pt"
    checkpoint.write_bytes(b"explicit checkpoint")
    return [
        "play",
        "--headless",
        "--device",
        "cuda:0",
        "--seed",
        "42",
        "--num-envs",
        "16",
        "--checkpoint",
        str(checkpoint),
        "--run-dir",
        str(tmp_path / "new-play-run"),
    ]


def test_timeout_scalar_is_one_finite_rollout_transition_fraction_per_iteration():
    runner_module = importlib.import_module("openhomie_isaaclab.him_rl.runner")
    assert getattr(runner_module, "TIMEOUT_SCALAR_TAG", None) == (
        "Episode_Termination/time_out"
    )
    assert getattr(runner_module, "TIMEOUT_SCALAR_UNIT", None) == (
        "rollout_transition_fraction"
    )
    fraction = getattr(runner_module, "_rollout_timeout_fraction", None)
    assert callable(fraction), "runner must expose the pure timeout fraction calculation"
    assert fraction(0, num_envs=4, num_steps_per_env=2) == pytest.approx(0.0)
    assert fraction(2, num_envs=4, num_steps_per_env=2) == pytest.approx(0.25)
    assert fraction(8, num_envs=4, num_steps_per_env=2) == pytest.approx(1.0)
    for invalid in (True, -1, 9):
        with pytest.raises((TypeError, ValueError)):
            fraction(invalid, num_envs=4, num_steps_per_env=2)

    source = _source(RUNNER)
    tree = _tree(RUNNER)
    runner = _class(tree, "HIMOnPolicyRunner")
    learn = _segment(RUNNER, _method(runner, "learn"))
    assert "time_outs" in learn
    assert "count_nonzero" in learn or ".sum(" in learn
    assert "_rollout_timeout_fraction" in learn
    assert "TIMEOUT_SCALAR_TAG" in source and "add_scalar" in source
    assert source.count("Episode_Termination/time_out") == 1

    process_calls = [
        node
        for node in ast.walk(_method(runner, "learn"))
        if isinstance(node, ast.Call) and _call_name(node) == "process_env_step"
    ]
    assert process_calls
    assert any(
        len(call.args) >= 3
        and isinstance(call.args[2], ast.Name)
        and call.args[2].id == "dones"
        for call in process_calls
    ), "timeout instrumentation must not replace the dones passed to HIMPPO"


def test_wrapped_long_dones_still_cross_one_explicit_boolean_boundary():
    tree = _tree(SIM)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_adapt_bool_dones" in functions
    adapter = _segment(SIM, functions["_adapt_bool_dones"])
    assert ".bool()" in adapter
    assert "time_outs" not in adapter


def test_evaluation_scenario_catalog_has_exact_commands_and_modes():
    run_module = importlib.import_module("openhomie_isaaclab.workflows.elf3_run")
    catalog = getattr(run_module, "EVALUATION_SCENARIOS", None)
    assert isinstance(catalog, dict)
    assert set(catalog) == {"stand", "forward", "turn", "crouch"}

    def fields(value):
        if isinstance(value, dict):
            return tuple(value["command"]), value["mode"]
        return tuple(value.command), value.mode

    assert {name: fields(spec) for name, spec in catalog.items()} == {
        "stand": ((0.0, 0.0, 0.0, None), 1),
        "forward": ((0.5, 0.0, 0.0, None), 0),
        "turn": ((0.0, 0.0, 0.5, None), 0),
        "crouch": ((0.0, 0.0, 0.0, 0.80), 2),
    }


def test_scenario_cli_is_play_only_exactly_1000_steps_and_keeps_batch_b_play(tmp_path):
    run_module = importlib.import_module("openhomie_isaaclab.workflows.elf3_run")
    ordinary = run_module.parse_request(_play_argv(tmp_path))
    assert getattr(ordinary, "scenario", "missing") is None
    assert getattr(ordinary, "steps", None) == 100

    scenario_argv = _play_argv(tmp_path)[:-1] + [
        str(tmp_path / "new-scenario-run"),
        "--scenario",
        "stand",
        "--steps",
        "1000",
    ]
    scenario = run_module.parse_request(scenario_argv)
    assert scenario.scenario == "stand"
    assert scenario.steps == 1000
    assert scenario.num_envs == 16

    invalid_variants = (
        _play_argv(tmp_path) + ["--scenario", "stand"],
        _play_argv(tmp_path) + ["--steps", "1000"],
        _play_argv(tmp_path) + ["--scenario", "stand", "--steps", "999"],
    )
    for argv in invalid_variants:
        with pytest.raises((SystemExit, TypeError, ValueError)):
            run_module.parse_request(argv)


def test_fixed_evaluation_command_is_opt_in_and_survives_all_resampling_paths():
    tree = _tree(ENV)
    environment = _class(tree, "Elf3HomieEnv")
    methods = {
        node.name: node
        for node in environment.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {
        "_allocate_state",
        "set_evaluation_command",
        "_resample_commands",
        "_maybe_resample_commands",
        "_reset_idx",
    }
    assert required <= set(methods)

    allocate = _segment(ENV, methods["_allocate_state"])
    setter = _segment(ENV, methods["set_evaluation_command"])
    resample = _segment(ENV, methods["_resample_commands"])
    periodic = _segment(ENV, methods["_maybe_resample_commands"])
    reset = _segment(ENV, methods["_reset_idx"])
    assert "evaluation" in allocate and "None" in allocate
    assert all(token in setter for token in ("_commands", "_modes", "[:, 4]"))
    assert "evaluation" in resample
    assert "evaluation" in periodic
    assert "_resample_commands" in reset

    allowed = {
        "_allocate_state",
        "set_evaluation_command",
        "_resample_commands",
        "_maybe_resample_commands",
        "get_evaluation_observables",
    }
    for name, method in methods.items():
        attributes = {
            node.attr
            for node in ast.walk(method)
            if isinstance(node, ast.Attribute) and "evaluation" in node.attr
        }
        if attributes:
            assert name in allowed, (
                f"evaluation-only state leaked into the default method {name}: {attributes}"
            )


def test_evaluation_observables_are_finite_shape_checked_and_use_reward_height():
    tree = _tree(ENV)
    environment = _class(tree, "Elf3HomieEnv")
    observables = _segment(ENV, _method(environment, "get_evaluation_observables"))
    literals = {
        node.value
        for node in ast.walk(_method(environment, "get_evaluation_observables"))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {
        "command",
        "root_lin_vel_b",
        "root_ang_vel_b",
        "roll_pitch",
        "tracking_height",
    } <= literals
    for token in (
        "root_link_lin_vel_b",
        "root_link_ang_vel_b",
        "root_link_quat_w",
        "root_link_pos_w",
        "body_pos_w",
        "_feet_ids",
        "ankle_sole_distance",
    ):
        assert token in observables
    assert "maximum" in observables or "max(" in observables
    assert "shape" in observables
    assert "_assert_finite" in observables or "isfinite" in observables


def test_scenario_play_installs_before_first_observation_and_records_c1_evidence():
    source = _source(SIM)
    tree = _tree(SIM)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    setter_lines = [
        call.lineno for call in calls if _call_name(call) == "set_evaluation_command"
    ]
    wrapper_lines = [
        call.lineno for call in calls if _call_name(call) == "RslRlVecEnvWrapper"
    ]
    runner_lines = [
        call.lineno for call in calls if _call_name(call) == "HIMOnPolicyRunner"
    ]
    assert setter_lines and wrapper_lines and runner_lines
    assert min(setter_lines) < min(wrapper_lines) < min(runner_lines)
    assert "request.scenario" in source
    assert "request.steps" in source
    assert "get_evaluation_observables" in source

    required_evidence = {
        "scenario",
        "command",
        "mode",
        "steps",
        "num_envs",
        "finite",
        "credited_env_steps",
        "non_timeout_termination_steps",
        "non_timeout_termination_reasons",
        "timeout_count",
        "survival",
        "metrics",
        "action_sha256",
        "trajectory_sha256",
        "sha256_before",
        "sha256_after",
    }
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert required_evidence <= literals


def test_evaluation_instrumentation_does_not_enter_reward_physics_or_cfg_paths():
    tree = _tree(ENV)
    environment = _class(tree, "Elf3HomieEnv")
    for name in ("_pre_physics_step", "_apply_action", "_get_rewards", "_get_dones"):
        assert "evaluation" not in _segment(ENV, _method(environment, name))

    rewards = _source(REWARDS)
    cfg = _source(ENV_CFG)
    tracking = _segment(
        REWARDS,
        next(
            node
            for node in _tree(REWARDS).body
            if isinstance(node, ast.FunctionDef) and node.name == "tracking_base_height"
        ),
    )
    assert "evaluation" not in rewards
    assert "evaluation" not in cfg
    assert "maximum" in tracking and "ankle_sole_distance" in tracking
