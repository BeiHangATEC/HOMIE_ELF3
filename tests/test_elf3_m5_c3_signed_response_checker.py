"""Red contracts for the independent M5 C3 signed-response CPU checker."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "isaaclab_ext/scripts/check_elf3_m5_c3_signed_response.py"
SEEDS = (42, 43, 44)
COMMANDS = np.asarray((0.0, 0.1, 0.2, 0.3, 0.4, 0.5), dtype=np.float64)
PLAN_SHA = "a" * 64
MODEL_SHA = "b" * 64


def checker() -> ModuleType:
    assert CHECKER.is_file(), (
        "missing independent C3 signed-response CPU checker: "
        "isaaclab_ext/scripts/check_elf3_m5_c3_signed_response.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_elf3_m5_c3_signed_response_checker", CHECKER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _piecewise(axis: str, in_gain: float, out_gain: float) -> np.ndarray:
    boundary = 0.3 if axis == "forward" else 0.2
    return np.asarray(
        [
            in_gain * value
            if value <= boundary
            else in_gain * boundary + out_gain * (value - boundary)
            for value in COMMANDS
        ],
        dtype=np.float64,
    )


def _evidence(
    axis: str,
    *,
    in_gain: float,
    out_gain: float,
    endpoint_mae: float | None = None,
) -> dict:
    limit = 0.20 if axis == "forward" else 0.25
    if endpoint_mae is None:
        endpoint_mae = limit + 0.05
    biases = np.asarray((-0.02, 0.0, 0.03), dtype=np.float64)
    curve = _piecewise(axis, in_gain, out_gain)
    responses = biases[:, None] + curve[None, :]
    mae = np.full((3, 5), min(limit, 0.10), dtype=np.float64)
    mae[:, -1] = endpoint_mae
    window = {"responses": responses, "mae": mae}
    return {
        "axis": axis,
        "seeds": list(SEEDS),
        "commands": COMMANDS.copy(),
        "windows": {
            "full": {name: value.copy() for name, value in window.items()},
            "post_initial": {
                name: value.copy() for name, value in window.items()
            },
        },
    }


def test_checker_is_cpu_only_and_does_not_reuse_the_production_classifier():
    module = checker()
    source = CHECKER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(CHECKER))
    imported = {
        (node.module or "") if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert module is not None
    assert not any(
        name == "torch"
        or name.startswith("torch.")
        or name == "isaaclab"
        or name.startswith("isaaclab.")
        or "elf3_c3_signed_response" in name
        for name in imported
    )


def test_cpu_window_metrics_preserve_signed_response_instead_of_only_mae():
    actual = np.asarray([[0.2, 0.2], [0.3, 0.3]], dtype=np.float32)
    active = np.ones_like(actual, dtype=np.bool_)
    positive = checker().compute_window_metrics(actual, 0.0, active)
    negative = checker().compute_window_metrics(-actual, 0.0, active)
    assert positive["credited_env_steps"] == 4
    assert positive["signed_mean"] == pytest.approx(0.25)
    assert negative["signed_mean"] == pytest.approx(-0.25)
    assert positive["mae"] == negative["mae"]
    assert positive["rmse"] == negative["rmse"]


def test_ideal_linear_response_with_a_failed_absolute_gate_is_not_called_ood():
    result = checker().classify_axis(
        "forward", _evidence("forward", in_gain=1.0, out_gain=1.0)
    )
    assert result["status"] == "NO_OOD_BREAK"
    assert result["windows"]["full"]["g_in"] == pytest.approx(1.0)
    assert result["windows"]["full"]["g_out"] == pytest.approx(1.0)


@pytest.mark.parametrize("axis", ["forward", "yaw"])
def test_piecewise_plateau_is_ood_supported_for_each_axis(axis):
    result = checker().classify_axis(
        axis, _evidence(axis, in_gain=1.0, out_gain=0.1)
    )
    assert result["status"] == "OOD_SUPPORTED"
    assert result["subtype"] == "SATURATION"
    for window in ("full", "post_initial"):
        assert result["windows"][window]["g_in"] == pytest.approx(1.0)
        assert result["windows"][window]["g_out"] == pytest.approx(0.1)


def test_negative_post_boundary_gain_is_reported_as_degradation_not_saturation():
    result = checker().classify_axis(
        "forward", _evidence("forward", in_gain=1.0, out_gain=-0.2)
    )
    assert result["status"] == "OOD_SUPPORTED"
    assert result["subtype"] == "DEGRADATION"
    assert result["windows"]["full"]["g_out"] == pytest.approx(-0.2)


def test_one_seed_without_a_gain_break_forces_mixed_even_when_the_mean_passes():
    evidence = _evidence("forward", in_gain=1.0, out_gain=0.1)
    ideal = _piecewise("forward", 1.0, 1.0)
    for window in evidence["windows"].values():
        baseline = window["responses"][2, 0]
        window["responses"][2] = baseline + ideal
    result = checker().classify_axis("forward", evidence)
    assert result["status"] == "MIXED"
    assert 44 in result["vetoed_seeds"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["windows"]["full"]["responses"].__setitem__(
            (0, 1), np.nan
        ),
        lambda p: p["windows"].pop("post_initial"),
        lambda p: p.update(seeds=[42, 43]),
        lambda p: p.update(commands=np.asarray((0.0, 0.1, 0.2))),
        lambda p: p["windows"]["full"].update(
            mae=np.zeros((3, 4), dtype=np.float64)
        ),
    ],
)
def test_malformed_or_nonfinite_axis_evidence_is_invalid_not_an_exception(mutate):
    evidence = _evidence("forward", in_gain=1.0, out_gain=0.1)
    mutate(evidence)
    result = checker().classify_axis("forward", evidence)
    assert result["status"] == "INVALID"
    assert result["reasons"]


def test_boundary_values_are_inclusive_but_epsilon_over_ratio_stops_ood():
    accepted = checker().classify_axis(
        "forward", _evidence("forward", in_gain=0.70, out_gain=0.42)
    )
    rejected = checker().classify_axis(
        "forward", _evidence("forward", in_gain=0.70, out_gain=0.420001)
    )
    assert accepted["status"] == "OOD_SUPPORTED"
    assert rejected["status"] == "NO_OOD_BREAK"


def test_passing_endpoint_gate_short_circuits_to_gate_passed():
    result = checker().classify_axis(
        "forward",
        _evidence("forward", in_gain=1.0, out_gain=0.1, endpoint_mae=0.20),
    )
    assert result["status"] == "GATE_PASSED"


def test_both_axes_must_support_ood_before_authorization_is_possible():
    module = checker()
    supported = {
        axis: module.classify_axis(
            axis, _evidence(axis, in_gain=1.0, out_gain=0.1)
        )
        for axis in ("forward", "yaw")
    }
    decision = module.classify_grid(
        supported, plan_sha256=PLAN_SHA, checkpoint_sha256=MODEL_SHA
    )
    assert decision["status"] == "WARM_START_AUTHORIZED"
    assert decision["authorized"] is True
    authorization = module.build_authorization(decision)
    assert authorization["plan_sha256"] == PLAN_SHA
    assert authorization["checkpoint_sha256"] == MODEL_SHA
    assert authorization["axis_status"] == {
        "forward": "OOD_SUPPORTED",
        "yaw": "OOD_SUPPORTED",
    }

    stopped = dict(supported)
    stopped["yaw"] = module.classify_axis(
        "yaw", _evidence("yaw", in_gain=1.0, out_gain=1.0)
    )
    decision = module.classify_grid(
        stopped, plan_sha256=PLAN_SHA, checkpoint_sha256=MODEL_SHA
    )
    assert decision["status"] == "STOPPED" and decision["authorized"] is False
    with pytest.raises(ValueError, match="authorized"):
        module.build_authorization(decision)
