"""Red contracts for the M5 C3 signed-response evidence producer."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/"
    "workflows/elf3_c3_signed_response.py"
)
MODULE_NAME = "openhomie_isaaclab.workflows.elf3_c3_signed_response"
SEEDS = (42, 43, 44)
VALUES = (0.1, 0.2, 0.3, 0.4, 0.5)
STEPS = 1000
NUM_ENVS = 16
MODEL_4000_SHA = "4" * 64
OLD_PILOT_ROOT = (
    "/home/user/wang-sm/OpenHomie_m5diag_s0_inrange_43e31b1_20260817"
)
FROZEN_C1 = {
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/"
    "him_rl/runner.py": (
        "daed23208a91a71efff1ffe7ecca7ea40623be0432f9bea74bc106b4bb31fbf4"
    ),
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/tasks/"
    "locomotion/elf3/elf3_homie_env.py": (
        "d7f54abb9b424e95d043df70ca350f32a61a43a7075ecc8859f2c87c7ed43342"
    ),
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/"
    "workflows/elf3_run.py": (
        "bde3dbe060b403befc752bbd3c450b1c3959bd48b3270be7af7b12794ef4dd28"
    ),
    "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/"
    "workflows/elf3_sim.py": (
        "c597d33c80580025d7fe28136262d75fa74301b9153dc548f2b2a71d2539dcf8"
    ),
}
REQUIRED_ARRAYS = {
    "step_index": {"dtype": "int64", "shape": [STEPS]},
    "command": {"dtype": "float32", "shape": [STEPS, NUM_ENVS, 4]},
    "mode": {"dtype": "int64", "shape": [STEPS, NUM_ENVS]},
    "root_lin_vel_b": {
        "dtype": "float32",
        "shape": [STEPS, NUM_ENVS, 3],
    },
    "root_ang_vel_b": {
        "dtype": "float32",
        "shape": [STEPS, NUM_ENVS, 3],
    },
    "roll_pitch": {"dtype": "float32", "shape": [STEPS, NUM_ENVS, 2]},
    "tracking_height": {
        "dtype": "float32",
        "shape": [STEPS, NUM_ENVS],
    },
    "action": {"dtype": "float32", "shape": [STEPS, NUM_ENVS, 12]},
    "reward": {"dtype": "float32", "shape": [STEPS, NUM_ENVS]},
    "active_before": {"dtype": "bool", "shape": [STEPS, NUM_ENVS]},
    "done": {"dtype": "bool", "shape": [STEPS, NUM_ENVS]},
    "timeout": {"dtype": "bool", "shape": [STEPS, NUM_ENVS]},
}


def api() -> ModuleType:
    assert MODULE_PATH.is_file(), (
        "missing C3 signed-response contract module: "
        "workflows/elf3_c3_signed_response.py"
    )
    return importlib.import_module(MODULE_NAME)


def _token(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def _expected_actions() -> list[dict]:
    actions: list[dict] = []
    for seed in SEEDS:
        baseline = f"seed{seed}_walk_zero"
        actions.append(
            {
                "action_id": baseline,
                "seed": seed,
                "scenario": "walk_zero",
                "command": [0.0, 0.0, 0.0, 1.01],
                "mode": 0,
                "steps": STEPS,
                "num_envs": NUM_ENVS,
                "baseline_action_id": None,
            }
        )
        for scenario in ("forward", "yaw"):
            for value in VALUES:
                command = [0.0, 0.0, 0.0, 1.01]
                command[0 if scenario == "forward" else 2] = value
                actions.append(
                    {
                        "action_id": (
                            f"seed{seed}_{scenario}_{_token(value)}"
                        ),
                        "seed": seed,
                        "scenario": scenario,
                        "command": command,
                        "mode": 0,
                        "steps": STEPS,
                        "num_envs": NUM_ENVS,
                        "baseline_action_id": baseline,
                    }
                )
    return actions


def _plan() -> dict:
    return {
        "schema_version": 1,
        "kind": "elf3_m5_c3_signed_response",
        "created_utc": "2026-08-18T00:00:00Z",
        "git_commit": "43e31b1c9286a2a75d29d1e4a59d31e90b8bb6fc",
        "source": {
            "checkpoint_path": "/evidence/s0/model_4000.pt",
            "checkpoint_sha256": MODEL_4000_SHA,
            "checkpoint_iteration": 4000,
            "stage": "S0",
            "manifest_path": "/evidence/s0/manifest.json",
            "manifest_sha256": "5" * 64,
        },
        "seeds": list(SEEDS),
        "num_envs": NUM_ENVS,
        "steps": STEPS,
        "command_values": list(VALUES),
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
        "required_arrays": copy.deepcopy(REQUIRED_ARRAYS),
        "actions": _expected_actions(),
        "excluded_evidence_roots": [OLD_PILOT_ROOT],
        "frozen_c1_sha256": dict(FROZEN_C1),
    }


def _arrays(action: dict) -> dict[str, np.ndarray]:
    command = np.asarray(action["command"], dtype=np.float32)
    arrays = {
        "step_index": np.arange(STEPS, dtype=np.int64),
        "command": np.broadcast_to(command, (STEPS, NUM_ENVS, 4)).copy(),
        "mode": np.full((STEPS, NUM_ENVS), action["mode"], dtype=np.int64),
        "root_lin_vel_b": np.zeros((STEPS, NUM_ENVS, 3), dtype=np.float32),
        "root_ang_vel_b": np.zeros((STEPS, NUM_ENVS, 3), dtype=np.float32),
        "roll_pitch": np.zeros((STEPS, NUM_ENVS, 2), dtype=np.float32),
        "tracking_height": np.full(
            (STEPS, NUM_ENVS), command[3], dtype=np.float32
        ),
        "action": np.zeros((STEPS, NUM_ENVS, 12), dtype=np.float32),
        "reward": np.zeros((STEPS, NUM_ENVS), dtype=np.float32),
        "active_before": np.ones((STEPS, NUM_ENVS), dtype=np.bool_),
        "done": np.zeros((STEPS, NUM_ENVS), dtype=np.bool_),
        "timeout": np.zeros((STEPS, NUM_ENVS), dtype=np.bool_),
    }
    arrays["done"][-1] = True
    arrays["timeout"][-1] = True
    return arrays


def test_contract_module_is_cpu_only_and_importable_without_isaac():
    module = api()
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MODULE_PATH))
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
        or name == "pxr"
        or name.startswith("pxr.")
        for name in imported
    )


def test_canonical_inventory_is_exactly_33_runs_with_one_shared_zero_per_seed():
    actions = api().canonical_grid_actions()
    assert actions == _expected_actions()
    assert len(actions) == 33
    assert len({row["action_id"] for row in actions}) == 33
    for seed in SEEDS:
        selected = [row for row in actions if row["seed"] == seed]
        zeros = [row for row in selected if row["scenario"] == "walk_zero"]
        assert len(selected) == 11 and len(zeros) == 1
        baseline = zeros[0]["action_id"]
        assert all(
            row["baseline_action_id"] == baseline
            for row in selected
            if row["scenario"] != "walk_zero"
        )


def test_plan_schema_is_exact_canonical_and_bound_to_model_4000():
    module = api()
    plan = _plan()
    assert module.validate_plan(plan) is None
    expected = json.dumps(
        plan, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert module.canonical_plan_bytes(plan) == expected
    assert module.plan_sha256(plan) == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(schema_version=2),
        lambda p: p.update(extra="not allowed"),
        lambda p: p["source"].update(checkpoint_iteration=2000),
        lambda p: p["source"].update(checkpoint_sha256="bad"),
        lambda p: p.update(seeds=[42, 43, 45]),
        lambda p: p["actions"].pop(),
        lambda p: p["actions"].append(copy.deepcopy(p["actions"][0])),
        lambda p: p["windows"].update(post_initial=[99, 1000]),
        lambda p: p["gain_limits"].update(out_max=0.500001),
        lambda p: p["required_arrays"]["mode"].update(dtype="float32"),
        lambda p: p.update(excluded_evidence_roots=[]),
        lambda p: p["frozen_c1_sha256"].update(
            {next(iter(FROZEN_C1)): "0" * 64}
        ),
    ],
)
def test_plan_rejects_schema_inventory_threshold_and_provenance_drift(mutate):
    plan = _plan()
    mutate(plan)
    with pytest.raises((KeyError, TypeError, ValueError)):
        api().validate_plan(plan)


def test_array_contract_and_timeout_credit_alignment_are_exact():
    module = api()
    action = _expected_actions()[0]
    arrays = _arrays(action)
    expected_active = module.derive_active_before(
        arrays["done"], arrays["timeout"]
    )
    np.testing.assert_array_equal(expected_active, arrays["active_before"])
    summary = module.validate_trajectory_arrays(arrays, action)
    assert summary == {
        "finite": True,
        "credited_env_steps": STEPS * NUM_ENVS,
        "timeout_count": NUM_ENVS,
        "non_timeout_termination_count": 0,
    }


def test_non_timeout_done_deactivates_on_the_next_snapshot_only():
    module = api()
    done = np.zeros((5, 2), dtype=np.bool_)
    timeout = np.zeros_like(done)
    done[1, 0] = True
    timeout[3, 1] = done[3, 1] = True
    expected = np.asarray(
        [[1, 1], [1, 1], [0, 1], [0, 1], [0, 1]], dtype=np.bool_
    )
    np.testing.assert_array_equal(module.derive_active_before(done, timeout), expected)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.pop("mode"),
        lambda a: a.update(extra=np.zeros(1, dtype=np.float32)),
        lambda a: a.update(step_index=np.arange(999, dtype=np.int64)),
        lambda a: a.update(mode=a["mode"].astype(np.float32)),
        lambda a: a["command"].__setitem__((500, 0, 0), 0.1),
        lambda a: a["mode"].__setitem__((500, 0), 1),
        lambda a: a["root_lin_vel_b"].__setitem__((0, 0, 0), np.nan),
        lambda a: a["active_before"].__setitem__((500, 0), False),
    ],
)
def test_array_contract_rejects_missing_extra_shape_dtype_fixed_state_and_credit_drift(
    mutate,
):
    action = _expected_actions()[0]
    arrays = _arrays(action)
    mutate(arrays)
    with pytest.raises((KeyError, TypeError, ValueError)):
        api().validate_trajectory_arrays(arrays, action)
