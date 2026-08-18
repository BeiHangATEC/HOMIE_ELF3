"""CPU and static contracts for the ELF3 C3 final evidence collector."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "openhomie_isaaclab.workflows.elf3_c3_final_collect"
MODULE_PATH = (
    ROOT
    / "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/"
    "workflows/elf3_c3_final_collect.py"
)
CLI = ROOT / "isaaclab_ext/scripts/elf3_m5_c3_final_collect.py"
PLAN = ROOT / "docs/plans/elf3-m5-c3-continual-training-design.md"


def api():
    return importlib.import_module(MODULE_NAME)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_evidence(tmp_path: Path):
    from test_elf3_m5_c3_final_checker import _source_evidence

    tmp_path.mkdir(parents=True)
    checkpoint, manifest, result = _source_evidence(tmp_path)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    result_payload = json.loads(result.read_text(encoding="utf-8"))
    result_payload["metrics"]["transitions"] = manifest_payload["lifecycle"][
        "transitions"
    ]
    _write_json(result, result_payload)
    return checkpoint, manifest, result


def _argv(tmp_path: Path, source):
    checkpoint, manifest, result = source
    outside = tmp_path / "outside"
    outside.mkdir(exist_ok=True)
    return [
        "--evidence-root", str(outside / "final-evidence"),
        "--checkpoint", str(checkpoint),
        "--source-manifest", str(manifest),
        "--source-result", str(result),
        "--plan", str(PLAN.resolve()),
        "--plan-sha256", _sha(PLAN),
        "--device", "cuda:0",
    ]


def test_module_is_cpu_importable_and_cli_launches_one_app():
    assert api() is not None
    workflow_tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    top_imports = {
        alias.name
        for node in workflow_tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch" not in top_imports
    assert "isaaclab" not in top_imports
    source = CLI.read_text(encoding="utf-8")
    assert source.count("AppLauncher(") == 1
    assert "enable_cameras" not in source


def test_request_binds_exact_v3_checkpoint_manifest_result_and_approved_plan(tmp_path):
    source = _source_evidence(tmp_path / "training")
    request = api().parse_request(_argv(tmp_path, source))
    binding = api().validate_source_binding(request)
    assert request.checkpoint.name == "model_2000.pt"
    assert request.source_manifest.name == "manifest.json"
    assert request.source_result.name == "result.json"
    assert request.plan_sha256 == api().APPROVED_PLAN_SHA256
    assert binding["checkpoint_sha256"] == _sha(request.checkpoint)


@pytest.mark.parametrize(
    "mutation", ["stage", "global", "checkpoint", "plan", "config_hash"]
)
def test_source_binding_rejects_wrong_final_identity_or_input(mutation, tmp_path):
    checkpoint, manifest, result = _source_evidence(tmp_path / "training")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    result_payload = json.loads(result.read_text(encoding="utf-8"))
    argv = _argv(tmp_path, (checkpoint, manifest, result))
    if mutation == "stage":
        manifest_payload["stage"] = "V2"
        _write_json(manifest, manifest_payload)
    elif mutation == "global":
        manifest_payload["lifecycle"]["global_iterations"]["final"] = 9999
        _write_json(manifest, manifest_payload)
    elif mutation == "checkpoint":
        result_payload["checkpoint"]["sha256"] = "0" * 64
        _write_json(result, result_payload)
    elif mutation == "plan":
        manifest_payload["plan"]["sha256"] = "0" * 64
        _write_json(manifest, manifest_payload)
    elif mutation == "config_hash":
        manifest_payload["configs"]["sha256"]["env"] = "0" * 64
        _write_json(manifest, manifest_payload)
    with pytest.raises((KeyError, TypeError, ValueError)):
        api().parse_request(argv)


def test_request_rejects_noncanonical_siblings_and_nested_evidence_root(tmp_path):
    checkpoint, manifest, result = _source_evidence(tmp_path / "training")
    renamed = result.parent / "final-result.json"
    result.rename(renamed)
    argv = _argv(tmp_path, (checkpoint, manifest, renamed))
    with pytest.raises(ValueError, match="canonical siblings"):
        api().parse_request(argv)

    renamed.rename(result)
    argv = _argv(tmp_path, (checkpoint, manifest, result))
    root_index = argv.index("--evidence-root") + 1
    argv[root_index] = str(checkpoint.parent / "nested-evidence")
    with pytest.raises(ValueError, match="outside"):
        api().parse_request(argv)


def test_v3_stage_is_assigned_before_config_serialization_and_hashing(monkeypatch):
    module = api()
    events = []

    class FakeCfg:
        command_stage = "S0"

        def to_dict(self):
            events.append(("serialize", self.command_stage))
            return {"command_stage": self.command_stage}

    class FakeAgent:
        def to_dict(self):
            return {"seed": 42}

    fake_sim = SimpleNamespace(
        _load_configs=lambda request: (FakeCfg(), FakeAgent(), {"stale": True}),
        _normalize_json=lambda value: dict(value),
        _json_sha256=lambda value: hashlib.sha256(
            json.dumps(value, sort_keys=True).encode()
        ).hexdigest(),
    )
    fake_stages = SimpleNamespace(
        get_stage=lambda stage: SimpleNamespace(
            to_dict=lambda: dict(module.STAGE_SPECS[stage])
        )
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "openhomie_isaaclab.tasks.locomotion.elf3.elf3_stages",
        fake_stages,
    )
    import openhomie_isaaclab.tasks.locomotion.elf3 as elf3_package
    monkeypatch.setattr(elf3_package, "elf3_stages", fake_stages, raising=False)

    _, _, configs = module._load_v3_configs(SimpleNamespace(), fake_sim)
    assert events == [("serialize", "V3")]
    assert configs["env"] == {"command_stage": "V3"}
    assert configs["sha256"]["env"] == fake_sim._json_sha256(configs["env"])


def test_inventory_is_exact_export_plus_four_by_three_scenario_matrix(tmp_path):
    module = api()
    request = module.C3FinalCollectRequest(
        evidence_root=(tmp_path / "evidence").resolve(),
        checkpoint=(tmp_path / "model_2000.pt").resolve(),
        source_manifest=(tmp_path / "manifest.json").resolve(),
        source_result=(tmp_path / "result.json").resolve(),
        plan=PLAN.resolve(),
        plan_sha256=_sha(PLAN),
    )
    runs = module.collection_requests(request)
    assert len(runs) == 13
    export = runs[0]
    assert (
        export.command,
        export.seed,
        export.num_envs,
        export.scenario,
        export.steps,
        export.run_dir.name,
    ) == ("export", 42, 1, None, None, "exact_export")
    scenarios = runs[1:]
    expected = [
        (scenario, seed)
        for scenario in ("stand", "forward", "turn", "crouch")
        for seed in (42, 43, 44)
    ]
    assert [(run.scenario, run.seed) for run in scenarios] == expected
    assert all(
        run.command == "play"
        and run.num_envs == 16
        and run.steps == 1000
        and run.headless is True
        and run.checkpoint == request.checkpoint
        for run in scenarios
    )
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "write_or_verify_aggregate(final_request)" in source
    assert 'acceptance_status": aggregate["status"]' in source
