import ast
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = REPO_ROOT / "HomieRL/legged_gym/resources/robots/elf3_description/urdf/elf3.urdf"
CONFIG_PATH = REPO_ROOT / "HomieRL/legged_gym/legged_gym/envs/g1/elf3_config.py"
sys.path.insert(0, str(REPO_ROOT / "HomieRL/rsl_rl"))

from rsl_rl.algorithms.him_ppo import HIMPPO


def load_elf3_config_assignments(path):
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    elf3 = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Elf3RoughCfg")
    nested_classes = {node.name: node for node in elf3.body if isinstance(node, ast.ClassDef)}
    result = {}

    for class_name in ("init_state", "control", "asset", "env"):
        values = {}
        for node in nested_classes[class_name].body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            expression = ast.Expression(body=node.value)
            try:
                values[node.targets[0].id] = eval(
                    compile(ast.fix_missing_locations(expression), str(path), "eval"),
                    {"__builtins__": {}},
                    values,
                )
            except (NameError, TypeError):
                continue
        result[class_name] = values

    return result


class Elf3FixedWaistTest(unittest.TestCase):
    def test_fixed_waist_matches_training_layout(self):
        robot = ET.parse(URDF_PATH).getroot()
        joints = robot.findall("joint")
        joint_types = {joint.get("name"): joint.get("type") for joint in joints}

        for name in ("waist_y_joint", "waist_x_joint", "waist_z_joint"):
            self.assertEqual(joint_types[name], "fixed")

        movable_names = {joint.get("name") for joint in joints if joint.get("type") != "fixed"}
        self.assertEqual(sum(joint.get("type") == "revolute" for joint in joints), 26)
        self.assertEqual(sum(joint.get("type") == "fixed" for joint in joints), 9)

        config = load_elf3_config_assignments(CONFIG_PATH)
        policy_dof_names = config["asset"]["policy_dof_names"]
        upper_body_dof_names = config["asset"]["upper_body_dof_names"]
        self.assertEqual(config["env"]["num_actions"], 12)
        self.assertEqual(config["env"]["num_dofs"], 26)
        self.assertEqual(config["env"]["num_one_step_observations"], 74)
        self.assertEqual(config["env"]["num_one_step_privileged_obs"], 77)
        self.assertEqual(config["env"]["num_observations"], 444)
        self.assertEqual(len(policy_dof_names), 12)
        self.assertEqual(len(upper_body_dof_names), 14)
        self.assertNotIn("waist_y_joint", config["init_state"]["default_joint_angles"])
        self.assertNotIn("waist_z_joint", config["init_state"]["default_joint_angles"])
        self.assertNotIn("waist", config["control"]["stiffness"])
        self.assertNotIn("waist", config["control"]["damping"])
        self.assertEqual(set(policy_dof_names + upper_body_dof_names), movable_names)

    def test_elf3_symmetry_is_involutive_for_26_dofs(self):
        ppo = HIMPPO.__new__(HIMPPO)
        ppo.actor_critic = SimpleNamespace(
            num_one_step_obs=74,
            actor_history_length=2,
            num_one_step_critic_obs=77,
            critic_history_length=1,
        )
        actor_obs = torch.arange(2 * 2 * 74, dtype=torch.float32).reshape(2, 148)
        critic_obs = torch.arange(2 * 77, dtype=torch.float32).reshape(2, 77)
        actions = torch.arange(24, dtype=torch.float32).reshape(2, 12)

        flipped_actor = ppo.flip_g1_actor_obs(actor_obs)
        dof_indices = torch.tensor(
            [
                6, 7, 8, 9, 10, 11,
                0, 1, 2, 3, 4, 5,
                19, 20, 21, 22, 23, 24, 25,
                12, 13, 14, 15, 16, 17, 18,
            ]
        )
        dof_signs = torch.tensor(
            [1.0, -1.0, -1.0, 1.0, 1.0, -1.0] * 2
            + [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0] * 2
        )
        first_frame = actor_obs.reshape(2, 2, 74)[0, 0]
        flipped_first_frame = flipped_actor.reshape(2, 2, 74)[0, 0]
        self.assertTrue(
            torch.equal(flipped_first_frame[10:36], first_frame[10:36][dof_indices] * dof_signs)
        )

        self.assertTrue(torch.equal(ppo.flip_g1_actor_obs(flipped_actor), actor_obs))
        self.assertTrue(
            torch.equal(ppo.flip_g1_critic_obs(ppo.flip_g1_critic_obs(critic_obs)), critic_obs)
        )
        self.assertTrue(torch.equal(ppo.flip_g1_actions(ppo.flip_g1_actions(actions)), actions))

    def test_g1_symmetry_route_remains_unchanged(self):
        ppo = HIMPPO.__new__(HIMPPO)
        ppo.actor_critic = SimpleNamespace(
            num_one_step_obs=76,
            actor_history_length=2,
            num_one_step_critic_obs=79,
            critic_history_length=1,
        )
        actor_obs = torch.arange(2 * 2 * 76, dtype=torch.float32).reshape(2, 152)
        critic_obs = torch.arange(2 * 79, dtype=torch.float32).reshape(2, 79)

        self.assertTrue(
            torch.equal(ppo.flip_g1_actor_obs(ppo.flip_g1_actor_obs(actor_obs)), actor_obs)
        )
        self.assertTrue(
            torch.equal(ppo.flip_g1_critic_obs(ppo.flip_g1_critic_obs(critic_obs)), critic_obs)
        )


if __name__ == "__main__":
    unittest.main()
