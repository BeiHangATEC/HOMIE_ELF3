from __future__ import annotations

import hashlib
import json
import math

import pytest

from test_elf3_m5_cli_contract import require_batch_a


def api():
    require_batch_a()
    from openhomie_isaaclab.workflows import elf3_run

    return elf3_run


def ckpt():
    return {
        "schema_version": 1,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "estimator_optimizer_state_dict": {},
        "learning_rate": 1e-3,
        "estimator_learning_rate": 1e-3,
        "estimator_lr_follows_policy": False,
        "iter": 7,
    }


def manifest():
    return {
        "schema_version": 1,
        "command": "train",
        "created_utc": "2026-08-17T00:00:00Z",
        "task_id": "OpenHomie-Elf3-Homie-Direct-v0",
        "seed": 42,
        "device": "cuda:0",
        "num_envs": 4096,
        "iterations": 2000,
        "cli": {},
        "git": {"commit": "a" * 40, "dirty_paths": []},
        "configs": {"env": {}, "agent": {}, "sha256": {}},
        "assets": {"urdf_sha256": "a" * 64, "usd_sha256": "b" * 64},
        "m4_sources": {"files": {}, "sha256": "c" * 64},
        "runtime": {
            "python": "3.11",
            "isaaclab_path": "IsaacLab-v2.3.2/source/isaaclab",
            "versions": {},
        },
        "gpu": {
            "name": "RTX 5090",
            "driver": "x",
            "total_mib": 32607,
            "free_mib": 31319,
        },
    }


def convergence():
    scalar_names = (
        "reward",
        "policy_loss",
        "value_loss",
        "estimator_loss",
        "learning_rate",
        "entropy",
        "throughput",
    )
    return {
        "mean_episode_length": [40.0] * 100 + [301.0] * 100,
        "timeouts": [0.0] * 195 + [1.0] * 5,
        "scalars": {k: [1.0] * 200 for k in scalar_names},
        "actual_transitions": 4096 * 2000 * 50,
        "expected_transitions": 4096 * 2000 * 50,
    }


def behavior():
    limits = {
        "stand": {"survival": 0.95, "height_mae": 0.08, "tilt_rms": 0.20},
        "forward": {
            "survival": 0.90,
            "velocity_mae": 0.20,
            "height_mae": 0.10,
        },
        "turn": {
            "survival": 0.90,
            "yaw_rate_mae": 0.25,
            "height_mae": 0.10,
        },
        "crouch": {
            "survival": 0.90,
            "height_mae": 0.08,
            "planar_speed_rms": 0.15,
        },
    }
    rows = []
    for seed in (42, 43, 44):
        for scenario, spec in limits.items():
            row = {"seed": seed, "scenario": scenario, "finite": True}
            row.update(
                {
                    k: v + 0.01 if k == "survival" else v - 0.01
                    for k, v in spec.items()
                }
            )
            rows.append(row)
    return {"rows": rows}


def _updated(payload, **changes):
    payload.update(changes)
    return payload


def _without(payload, key):
    payload.pop(key)
    return payload


def write_events(path, values):
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=path)
    try:
        for tag, points in values.items():
            for step, value in enumerate(points):
                writer.add_scalar(tag, value, step)
    finally:
        writer.close()


def test_hash_changes_with_content(tmp_path):
    m = api()
    p = tmp_path / "x"
    p.write_bytes(b"a")
    first = m.sha256_file(p)
    assert first == hashlib.sha256(b"a").hexdigest()
    p.write_bytes(b"b")
    assert m.sha256_file(p) != first


def test_exclusive_directory_rejects_file_directory_and_symlink(tmp_path):
    m = api()
    run = tmp_path / "run"
    assert m.create_run_directory(run) == run.resolve()
    file = tmp_path / "file"
    file.write_text("x")
    link = tmp_path / "link"
    link.symlink_to(run, target_is_directory=True)
    for path in (run, file, link):
        with pytest.raises((FileExistsError, ValueError)):
            m.create_run_directory(path)


def test_one_shot_json_is_safe_and_non_overwriting(tmp_path):
    m = api()
    p = tmp_path / "value.json"
    m.write_json_once(p, {"nested": (1, 2)})
    assert json.loads(p.read_text()) == {"nested": [1, 2]}
    with pytest.raises((FileExistsError, ValueError)):
        m.write_json_once(p, {})
    for i, value in enumerate((math.nan, math.inf, -math.inf)):
        bad = tmp_path / f"bad{i}.json"
        with pytest.raises(ValueError):
            m.write_json_once(bad, {"value": value})
        assert not bad.exists()


def test_atomic_json_fsyncs_and_replaces_in_destination_directory(
    tmp_path, monkeypatch
):
    module = api()
    fsync_calls = []
    replace_calls = []
    real_fsync = module.os.fsync
    real_replace = module.os.replace

    def record_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    def record_replace(source, destination):
        source = module.Path(source)
        destination = module.Path(destination)
        replace_calls.append((source, destination))
        assert source.parent == destination.parent
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "fsync", record_fsync)
    monkeypatch.setattr(module.os, "replace", record_replace)
    target = tmp_path / "manifest.json"
    module.write_json_once(target, {"schema_version": 1})
    assert fsync_calls
    assert replace_calls and replace_calls[-1][1] == target


def test_atomic_json_never_deletes_a_concurrent_writers_target(
    tmp_path, monkeypatch
):
    module = api()
    target = tmp_path / "manifest.json"
    real_open = module.os.open

    def lose_reservation(path, flags, mode=0o777):
        if module.Path(path) == target:
            target.write_bytes(b"concurrent-owner")
            raise FileExistsError("lost O_EXCL race")
        return real_open(path, flags, mode)

    monkeypatch.setattr(module.os, "open", lose_reservation)
    with pytest.raises(FileExistsError):
        module.write_json_once(target, {"schema_version": 1})
    assert target.read_bytes() == b"concurrent-owner"


def test_classify_absent_incomplete_pass_and_fail(tmp_path):
    m = api()
    assert m.classify_run(tmp_path / "none") == "ABSENT"
    for status in (None, "PASS", "FAIL"):
        run = tmp_path / str(status)
        run.mkdir()
        m.write_json_once(run / "manifest.json", manifest())
        if status:
            m.write_json_once(run / "result.json", {"status": status})
        assert m.classify_run(run) == ("INCOMPLETE" if status is None else status)


@pytest.mark.parametrize(
    ("mutation", "resume", "valid"),
    [
        (lambda p: p, True, True),
        (lambda p: p, False, True),
        (lambda p: None, False, False),
        (lambda p: _without(p, "model_state_dict"), False, False),
        (lambda p: _updated(p, schema_version=2), False, False),
        (lambda p: _updated(p, iter=True), False, False),
        (lambda p: _updated(p, iter=-1), False, False),
        (lambda p: _updated(p, learning_rate=math.nan), False, False),
        (lambda p: _updated(p, estimator_learning_rate=math.inf), False, False),
        (lambda p: _without(p, "optimizer_state_dict"), True, False),
        (lambda p: _without(p, "estimator_optimizer_state_dict"), True, False),
        (lambda p: _without(p, "optimizer_state_dict"), False, True),
        (lambda p: _without(p, "estimator_optimizer_state_dict"), False, True),
        (lambda p: _updated(p, model_state_dict=None), False, False),
        (lambda p: _updated(p, optimizer_state_dict=None), True, False),
        (
            lambda p: _updated(p, estimator_optimizer_state_dict=None),
            True,
            False,
        ),
        (lambda p: _updated(p, optimizer_state_dict=None), False, False),
        (
            lambda p: _updated(p, estimator_optimizer_state_dict=None),
            False,
            False,
        ),
    ],
)
def test_checkpoint_validation(mutation, resume, valid):
    m = api()
    payload = ckpt()
    candidate = mutation(payload)
    if valid:
        assert (
            m.validate_checkpoint_payload(candidate, require_optimizers=resume) == 7
        )
    else:
        with pytest.raises((KeyError, TypeError, ValueError)):
            m.validate_checkpoint_payload(
                candidate, require_optimizers=resume
            )


def test_iteration_and_budget_math():
    m = api()
    assert m.final_iteration(7, 3) == 10
    for args in ((True, 1), (0, True), (0, -1), (-1, 1)):
        with pytest.raises(ValueError):
            m.final_iteration(*args)
    assert (
        m.canonical_iterations(4096, 50) == 2000
        and m.canonical_iterations(2048, 50) == 4000
    )
    odd = m.canonical_iterations(3000, 50)
    assert (
        odd == math.ceil(4096 * 2000 / 3000)
        and 3000 * odd * 50 >= 4096 * 2000 * 50
    )
    for args in ((0, 50), (-1, 50), (1, 0), (1, -1), (True, 50)):
        with pytest.raises(ValueError):
            m.canonical_iterations(*args)


def test_manifest_requires_identity_and_rejects_nonfinite_or_machine_paths():
    m = api()
    m.validate_manifest(manifest())
    for key in tuple(manifest()):
        p = manifest()
        p.pop(key)
        with pytest.raises((KeyError, TypeError, ValueError)):
            m.validate_manifest(p)
    for bad in (math.nan, math.inf):
        p = manifest()
        p["gpu"]["free_mib"] = bad
        with pytest.raises((TypeError, ValueError)):
            m.validate_manifest(p)
    p = manifest()
    p["git"]["dirty_paths"] = ["/home/user/wang-sm/OpenHomie/x.py"]
    with pytest.raises((TypeError, ValueError)):
        m.validate_manifest(p)


def test_manifest_modes_and_path_ownership():
    m = api()
    m.validate_manifest(manifest())
    for command in ("play", "export"):
        p = manifest()
        p["command"] = command
        p["iterations"] = None
        m.validate_manifest(p)

    p = manifest()
    p["iterations"] = None
    with pytest.raises((TypeError, ValueError)):
        m.validate_manifest(p)
    for command in ("play", "export"):
        p = manifest()
        p["command"] = command
        with pytest.raises((TypeError, ValueError)):
            m.validate_manifest(p)

    p = manifest()
    p["runtime"]["isaaclab_path"] = (
        "/home/user/wang-sm/IsaacLab-v2.3.2/source/isaaclab"
    )
    m.validate_manifest(p)

    p = manifest()
    p["git"]["dirty_paths"] = ["/home/user/wang-sm/OpenHomie/x.py"]
    with pytest.raises((TypeError, ValueError)):
        m.validate_manifest(p)
    p = manifest()
    p["m4_sources"]["files"] = {
        "/home/user/wang-sm/OpenHomie/him_rl/runner.py": "a" * 64
    }
    with pytest.raises((TypeError, ValueError)):
        m.validate_manifest(p)


@pytest.mark.parametrize(
    "change",
    (
        "mean",
        "ratio",
        "timeouts",
        "nan",
        "inf",
        "missing",
        "short",
        "short_scalar",
        "budget",
    ),
)
def test_convergence_rejects_each_contract_failure(change):
    m = api()
    p = convergence()
    if change == "mean":
        p["mean_episode_length"][-100:] = [300.0] * 100
    elif change == "ratio":
        p["mean_episode_length"][:100] = [76.0] * 100
    elif change == "timeouts":
        p["timeouts"][-100:] = [0.0] * 96 + [1.0] * 4
    elif change in ("nan", "inf"):
        p["scalars"]["reward"] = [
            math.nan if change == "nan" else math.inf
        ]
    elif change == "missing":
        p["scalars"].pop("entropy")
    elif change == "short":
        p["mean_episode_length"] = [301.0] * 199
    elif change == "short_scalar":
        p["scalars"]["reward"] = [1.0]
    else:
        p["actual_transitions"] -= 1
    result = m.evaluate_convergence(p)
    assert result["passed"] is False and result["reasons"]


def test_convergence_accepts_approved_contract():
    assert api().evaluate_convergence(convergence()) == {
        "passed": True,
        "reasons": [],
    }


@pytest.mark.parametrize(
    ("scenario", "metric", "bad"),
    [
        ("stand", "survival", 0.94),
        ("stand", "height_mae", 0.081),
        ("stand", "tilt_rms", 0.201),
        ("forward", "survival", 0.89),
        ("forward", "velocity_mae", 0.201),
        ("forward", "height_mae", 0.101),
        ("turn", "survival", 0.89),
        ("turn", "yaw_rate_mae", 0.251),
        ("turn", "height_mae", 0.101),
        ("crouch", "survival", 0.89),
        ("crouch", "height_mae", 0.081),
        ("crouch", "planar_speed_rms", 0.151),
    ],
)
def test_behavior_rejects_every_threshold(scenario, metric, bad):
    p = behavior()
    for row in p["rows"]:
        if row["scenario"] == scenario:
            row[metric] = bad
    result = api().evaluate_behavior(p)
    assert result["passed"] is False and result["reasons"]


@pytest.mark.parametrize(
    "change", ("missing_seed", "missing_scenario", "finite", "nan")
)
def test_behavior_rejects_incomplete_or_nonfinite(change):
    p = behavior()
    if change == "missing_seed":
        p["rows"] = [r for r in p["rows"] if r["seed"] != 44]
    elif change == "missing_scenario":
        p["rows"] = [r for r in p["rows"] if r["scenario"] != "turn"]
    elif change == "finite":
        p["rows"][0]["finite"] = False
    else:
        p["rows"][0]["height_mae"] = math.nan
    result = api().evaluate_behavior(p)
    assert result["passed"] is False and result["reasons"]


def test_behavior_accepts_four_scenarios_and_three_seeds():
    assert api().evaluate_behavior(behavior()) == {
        "passed": True,
        "reasons": [],
    }


def test_tensorboard_reader_returns_ordered_finite_required_series(tmp_path):
    module = api()
    with pytest.raises(ValueError):
        module.read_tensorboard_scalars(
            tmp_path, ("reward", "loss"), minimum_points=0
        )
    write_events(tmp_path, {"reward": [1.0, 2.0], "loss": [3.0, 4.0]})
    assert module.read_tensorboard_scalars(
        tmp_path, ("reward", "loss"), minimum_points=2
    ) == {"reward": [1.0, 2.0], "loss": [3.0, 4.0]}


@pytest.mark.parametrize(
    "values",
    [
        {"reward": [1.0, 2.0]},
        {"reward": [1.0, 2.0], "loss": [3.0]},
        {"reward": [1.0, math.nan], "loss": [3.0, 4.0]},
        {"reward": [1.0, math.inf], "loss": [3.0, 4.0]},
    ],
)
def test_tensorboard_reader_rejects_missing_short_or_nonfinite_series(
    tmp_path, values
):
    module = api()
    write_events(tmp_path, values)
    with pytest.raises((KeyError, TypeError, ValueError)):
        module.read_tensorboard_scalars(
            tmp_path, ("reward", "loss"), minimum_points=2
        )
