from __future__ import annotations

import ast
import importlib.metadata
import inspect
from pathlib import Path

import onnx
import onnxruntime

from openhomie_isaaclab import elf3_constants as C


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab"
HIM = PACKAGE / "him_rl"
AGENTS = PACKAGE / "tasks/locomotion/elf3/agents"
PRODUCTION = [
    HIM / "__init__.py", HIM / "estimator.py", HIM / "actor_critic.py",
    HIM / "symmetry.py", HIM / "storage.py", HIM / "ppo.py",
    HIM / "runner.py", HIM / "exporter.py", AGENTS / "__init__.py",
    AGENTS / "him_ppo_cfg.py",
]


def require_production():
    missing = [str(path.relative_to(ROOT)) for path in PRODUCTION if not path.is_file()]
    assert not missing, f"missing M4 production files: {missing}"


def test_m4_production_surface_exists_and_parses():
    require_production()
    for path in PRODUCTION:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_generic_him_modules_are_robot_agnostic_and_have_no_machine_paths():
    require_production()
    forbidden = tuple(C.JOINT_NAMES) + ("elf3_constants", "/home/", "wang-sm")
    for path in PRODUCTION[:8]:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), f"generic leak in {path.name}"


def test_production_excludes_reference_29dof_dimension_literals():
    require_production()
    reference_literals = {str(C.num_one_step_actor_obs() + 2),
                          str(C.num_one_step_critic_obs() + 2),
                          str(C.num_actor_obs() + 2 * C.NUM_ACTOR_HISTORY)}
    for path in PRODUCTION:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        integers = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                    and isinstance(node.value, int) and not isinstance(node.value, bool)}
        assert not integers.intersection(map(int, reference_literals)), path


def test_agent_cfg_derives_canonical_dimensions_and_public_class_names():
    require_production()
    from openhomie_isaaclab.tasks.locomotion.elf3.agents.him_ppo_cfg import Elf3HIMRunnerCfg

    cfg = Elf3HIMRunnerCfg()
    assert cfg.class_name == "HIMOnPolicyRunner"
    assert cfg.algorithm.class_name == "HIMPPO"
    assert cfg.policy.class_name == "HIMActorCritic"
    assert cfg.obs_groups == {"policy": ["policy"], "critic": ["critic"]}
    assert cfg.policy.num_one_step_obs == C.num_one_step_actor_obs()
    assert cfg.policy.actor_history_length == C.NUM_ACTOR_HISTORY
    assert cfg.policy.num_one_step_critic_obs == C.num_one_step_critic_obs()
    assert cfg.algorithm.learning_rate == 1.0e-3
    assert cfg.algorithm.estimator_learning_rate == 1.0e-3
    assert cfg.algorithm.schedule == "adaptive"
    assert cfg.algorithm.entropy_coef == 0.01
    assert len(cfg.algorithm.mirror["dof_mirror_indices"]) == C.NUM_ROBOT_DOFS
    assert len(cfg.algorithm.mirror["action_mirror_indices"]) == C.NUM_POLICY_ACTIONS


def test_m4_classes_match_rsl_312_public_constructor_and_step_contracts():
    require_production()
    from openhomie_isaaclab.him_rl import HIMPPO, HIMActorCritic, HIMOnPolicyRunner, HIMRolloutStorage
    from rsl_rl.algorithms import PPO
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.storage import RolloutStorage

    assert issubclass(HIMPPO, PPO)
    assert issubclass(HIMOnPolicyRunner, OnPolicyRunner)
    assert issubclass(HIMRolloutStorage, RolloutStorage)
    assert list(inspect.signature(HIMActorCritic.__init__).parameters)[:4] == [
        "self", "obs", "obs_groups", "num_actions"
    ]
    assert list(inspect.signature(HIMPPO.process_env_step).parameters)[:5] == [
        "self", "obs", "rewards", "dones", "extras"
    ]


def test_pinned_public_dependency_versions_are_exact():
    assert importlib.metadata.version("rsl-rl-lib") == "3.1.2"
    assert onnx.__version__ == "1.21.0"
    assert onnxruntime.__version__ == "1.28.0"
