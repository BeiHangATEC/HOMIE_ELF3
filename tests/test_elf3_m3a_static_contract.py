"""Static red tests for the approved M3a production surface."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ELF3_TASK = (
    REPO_ROOT
    / "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/tasks/locomotion/elf3"
)
ROLLOUT = REPO_ROOT / "isaaclab_ext/scripts/random_rollout_elf3.py"
TASK_ID = "OpenHomie-Elf3-Homie-Direct-v0"


def _source(path: Path) -> str:
    assert path.is_file(), f"M3a production interface is missing: {path.relative_to(REPO_ROOT)}"
    source = path.read_text()
    ast.parse(source, filename=str(path))
    return source


def _dotted_name(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _ast_references(source: str) -> set[str]:
    references = set()
    for node in ast.walk(ast.parse(source)):
        target = node.func if isinstance(node, ast.Call) else node
        name = _dotted_name(target)
        if name is not None:
            references.add(name)
    return references


def _imported_modules(source: str) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_pure_environment_core_module_exists_and_parses():
    _source(ELF3_TASK / "elf3_homie_env_core.py")


def test_direct_environment_cfg_exists_and_parses():
    source = _source(ELF3_TASK / "elf3_homie_env_cfg.py")
    assert "Elf3HomieEnvCfg" in source
    assert "DirectRLEnvCfg" in source
    references = _ast_references(source)
    assert "C.NUM_POLICY_ACTIONS" in references
    assert "C.num_actor_obs" in references
    assert "C.num_critic_obs" in references


def test_direct_environment_exists_and_parses():
    source = _source(ELF3_TASK / "elf3_homie_env.py")
    assert "Elf3HomieEnv" in source
    assert "DirectRLEnv" in source


def test_registration_uses_the_exact_compatible_task_id():
    source = _source(ELF3_TASK / "__init__.py")
    assert source.count(TASK_ID) == 1
    assert "OpenHomie-ELF3-Homie-Direct-v0" not in source
    assert "Elf3HomieEnv" in source
    assert "Elf3HomieEnvCfg" in source


def test_random_rollout_entry_point_exists_parses_and_reports_before_close():
    source = _source(ROLLOUT)
    assert TASK_ID in source
    assert "openhomie_isaaclab.tasks.locomotion.elf3" in _imported_modules(source)
    assert "M3a random rollout: PASS" in source
    assert "random_rollout_elf3 exit code:" in source
    assert "simulation_app.close()" in source
    assert source.index("random_rollout_elf3 exit code:") < source.index(
        "simulation_app.close()"
    )


def test_m3a_files_do_not_embed_machine_specific_paths_or_m4_m6_imports():
    paths = [
        ELF3_TASK / "elf3_homie_env_core.py",
        ELF3_TASK / "elf3_homie_env_cfg.py",
        ELF3_TASK / "elf3_homie_env.py",
        ELF3_TASK / "__init__.py",
        ROLLOUT,
    ]
    combined = "\n".join(_source(path) for path in paths)
    assert "/home/" not in combined
    assert "/root/" not in combined
    for forbidden in ("him_rl", "rsl_rl", "mujoco", "sim2sim"):
        assert forbidden not in combined.lower()
