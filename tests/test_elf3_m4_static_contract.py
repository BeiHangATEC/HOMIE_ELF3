from __future__ import annotations

import ast
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import onnx
import onnxruntime

from openhomie_isaaclab import elf3_constants as C

from _elf3_m4_helpers import make_obs


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab"
HIM = PACKAGE / "him_rl"
AGENTS = PACKAGE / "tasks/locomotion/elf3/agents"
CFG_ENTRY_POINT = (
    "openhomie_isaaclab.tasks.locomotion.elf3.agents."
    "him_ppo_cfg:Elf3HIMRunnerCfg"
)
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


def test_generic_him_import_graph_has_no_robot_or_task_dependency():
    require_production()
    for path in PRODUCTION[:8]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.append(node.module or "")
        assert not any(
            "tasks" in module or "elf3" in module for module in imported_modules
        ), f"generic import leak in {path.name}: {imported_modules}"


def test_agent_sources_exclude_machine_paths_and_reference_dimensions():
    require_production()
    for path in PRODUCTION[8:]:
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text and "wang-sm" not in text, path
        tree = ast.parse(text, filename=str(path))
        integers = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        }
        assert not integers.intersection({80, 83, 480}), path


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
    assert cfg.seed == 42
    assert cfg.device == "cuda:0"
    assert cfg.max_iterations == 100_000
    assert cfg.clip_actions == 100.0
    assert cfg.experiment_name == "elf3_homie_him_isaaclab"
    assert cfg.algorithm.class_name == "HIMPPO"
    assert cfg.policy.class_name == "HIMActorCritic"
    assert cfg.obs_groups == {"policy": ["policy"], "critic": ["critic"]}
    assert cfg.policy.num_one_step_obs == C.num_one_step_actor_obs()
    assert cfg.policy.actor_history_length == C.NUM_ACTOR_HISTORY
    assert cfg.policy.num_one_step_critic_obs == C.num_one_step_critic_obs()
    assert C.NUM_ROBOT_DOFS == 28
    assert C.NUM_POLICY_ACTIONS == 12
    assert C.num_one_step_actor_obs() == 78
    assert C.num_one_step_critic_obs() == 81
    assert C.num_actor_obs() == 468
    assert cfg.policy.actor_hidden_dims == [512, 256, 256]
    assert cfg.policy.critic_hidden_dims == [512, 256, 256]
    assert cfg.policy.estimator_hidden_dims == [256, 256]
    assert cfg.policy.estimator_target_hidden_dims == [256, 256]
    assert cfg.policy.estimator_latent_dim == 32
    assert cfg.policy.estimator_num_prototypes == 64
    assert cfg.algorithm.learning_rate == 1.0e-3
    assert cfg.algorithm.estimator_learning_rate is None
    assert cfg.algorithm.schedule == "adaptive"
    assert cfg.algorithm.entropy_coef == 0.01
    assert cfg.algorithm.rnd_cfg is None
    assert cfg.algorithm.symmetry_cfg is None
    assert cfg.algorithm.mirror == {
        "dof_mirror_indices": list(C.DOF_MIRROR_INDICES),
        "dof_mirror_signs": list(C.DOF_MIRROR_SIGNS),
        "action_mirror_indices": list(C.ACTION_MIRROR_INDICES),
        "action_mirror_signs": list(C.ACTION_MIRROR_SIGNS),
        "obs_mirror_signs": list(C.OBS_HEAD_MIRROR_SIGNS),
        "critic_tail_mirror_signs": list(C.CRITIC_TAIL_MIRROR_SIGNS),
    }


def test_agent_cfg_plain_python_import_does_not_load_pxr():
    require_production()
    source_root = str(PACKAGE.parent)
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        source_root
        if not current_pythonpath
        else os.pathsep.join((source_root, current_pythonpath))
    )
    code = (
        "import sys; "
        "from openhomie_isaaclab.tasks.locomotion.elf3.agents.him_ppo_cfg "
        "import Elf3HIMRunnerCfg; "
        "Elf3HIMRunnerCfg(); "
        "assert 'pxr' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _assert_plain_config(value):
    if isinstance(value, dict):
        assert all(isinstance(key, str) for key in value)
        for nested in value.values():
            _assert_plain_config(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_plain_config(nested)
    else:
        assert value is None or isinstance(value, (str, int, float, bool))


def test_agent_cfg_entry_point_resolves_and_serializes_to_plain_runner_dict():
    require_production()
    from openhomie_isaaclab.tasks.locomotion.elf3.agents import (
        HIM_PPO_CFG_ENTRY_POINT,
    )

    assert HIM_PPO_CFG_ENTRY_POINT == CFG_ENTRY_POINT
    module_name, attribute_name = CFG_ENTRY_POINT.split(":", maxsplit=1)
    cfg_class = getattr(importlib.import_module(module_name), attribute_name)
    payload = cfg_class().to_dict()
    assert isinstance(payload, dict)
    json.dumps(payload)
    _assert_plain_config(payload)


def test_agent_cfg_to_dict_constructs_him_runner_without_isaac_sim():
    require_production()
    from openhomie_isaaclab.him_rl import HIMOnPolicyRunner

    module_name, attribute_name = CFG_ENTRY_POINT.split(":", maxsplit=1)
    cfg_class = getattr(importlib.import_module(module_name), attribute_name)

    class FakeVecEnv:
        num_envs = 2
        num_actions = C.NUM_POLICY_ACTIONS
        device = "cpu"
        cfg = {}

        def get_observations(self):
            return make_obs(self.num_envs, random=False)

    runner = HIMOnPolicyRunner(FakeVecEnv(), cfg_class().to_dict())
    assert runner.alg.policy.num_actions == C.NUM_POLICY_ACTIONS


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
