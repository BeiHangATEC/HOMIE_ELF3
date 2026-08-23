import ast
import importlib.util
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGGED_GYM_ROOT = REPO_ROOT / "HomieRL/legged_gym"
RSL_RL_ROOT = REPO_ROOT / "HomieRL/rsl_rl"
CONFIG_PATH = LEGGED_GYM_ROOT / "legged_gym/envs/g1/elf3_config.py"
STAGED_TRAINING_PATH = LEGGED_GYM_ROOT / "legged_gym/envs/base/staged_training.py"
sys.path.insert(0, str(LEGGED_GYM_ROOT))
sys.path.insert(0, str(RSL_RL_ROOT))


def load_staged_training_module():
    spec = importlib.util.spec_from_file_location("elf3_staged_training", STAGED_TRAINING_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def nested_assignments(class_node, nested_name):
    nested = next(
        node for node in class_node.body
        if isinstance(node, ast.ClassDef) and node.name == nested_name
    )
    values = {}
    for node in nested.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if not isinstance(node.targets[0], ast.Name):
            continue
        expression = ast.Expression(body=node.value)
        try:
            values[node.targets[0].id] = eval(
                compile(ast.fix_missing_locations(expression), str(CONFIG_PATH), "eval"),
                {"__builtins__": {}},
                values,
            )
        except (NameError, TypeError):
            continue
    return values


class DummyState:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.state = state


class DummyEnv:
    def __init__(self):
        self.curriculum_state = {
            "action_curriculum_ratio": 0.0,
            "stage1_qualified_windows": 0,
            "stage1_ready": 0,
        }

    def get_curriculum_state(self):
        return dict(self.curriculum_state)

    def set_curriculum_state(self, state):
        self.curriculum_state.update(state)

    def reset_curriculum_state(self):
        self.curriculum_state = {
            "action_curriculum_ratio": 0.0,
            "stage1_qualified_windows": 0,
            "stage1_ready": 0,
        }


class Elf3StagedTrainingTest(unittest.TestCase):
    def test_stage_configs_keep_identical_policy_layout(self):
        module = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
        classes = {
            node.name: node for node in module.body if isinstance(node, ast.ClassDef)
        }
        common_env = nested_assignments(classes["Elf3RoughCfg"], "env")
        height_commands = nested_assignments(classes["Elf3HeightCfg"], "commands")
        height_rewards = nested_assignments(classes["Elf3HeightCfg"], "rewards")
        height_runner = nested_assignments(classes["Elf3HeightCfgPPO"], "runner")
        walk_commands = nested_assignments(classes["Elf3WalkCfg"], "commands")
        walk_runner = nested_assignments(classes["Elf3WalkCfgPPO"], "runner")
        toeout_commands = nested_assignments(
            classes["Elf3WalkToeOutCfg"], "commands"
        )
        toeout_rewards = nested_assignments(
            classes["Elf3WalkToeOutCfg"], "rewards"
        )
        toeout_scales = nested_assignments(
            next(
                node
                for node in classes["Elf3WalkToeOutCfg"].body
                if isinstance(node, ast.ClassDef) and node.name == "rewards"
            ),
            "scales",
        )
        toeout_algorithm = nested_assignments(
            classes["Elf3WalkToeOutCfgPPO"], "algorithm"
        )
        toeout_runner = nested_assignments(
            classes["Elf3WalkToeOutCfgPPO"], "runner"
        )

        self.assertEqual(common_env["num_dofs"], 26)
        self.assertEqual(common_env["num_actions"], 12)
        self.assertEqual(common_env["num_observations"], 444)
        self.assertEqual(height_commands["training_mode"], "height")
        self.assertEqual(height_commands["resampling_time"], 6.0)
        self.assertEqual(height_rewards["base_height_target"], 1.0)
        self.assertEqual(height_runner["max_iterations"], 20000)
        self.assertEqual(height_runner["save_interval"], 100)
        self.assertEqual(height_runner["logger"], "tensorboard")
        self.assertEqual(walk_commands["training_mode"], "walk")
        self.assertEqual(walk_commands["resampling_time"], 6.0)
        self.assertEqual(walk_commands["height_target"], 1.0)
        self.assertEqual(walk_commands["height_min"], 0.3)
        self.assertEqual(walk_commands["height_max"], 1.0)
        self.assertEqual(walk_commands["height_endpoint_probability"], 0.25)
        self.assertEqual(walk_commands["height_slew_rate"], 0.20)
        self.assertEqual(walk_commands["walk_command_probability"], 0.5)
        self.assertEqual(walk_commands["squat_command_probability"], 0.5)
        self.assertEqual(walk_runner["max_iterations"], 100000)
        self.assertEqual(walk_runner["save_interval"], 200)
        self.assertEqual(walk_runner["logger"], "tensorboard")
        self.assertEqual(toeout_commands["toe_out_start_height"], 0.735)
        self.assertEqual(toeout_commands["toe_out_full_angle_height"], 0.50)
        self.assertEqual(toeout_commands["toe_out_max_angle_deg"], 15.0)
        self.assertEqual(toeout_rewards["toe_out_tracking_sigma"], 0.05)
        self.assertTrue(toeout_rewards["enforce_squat_feet_distance_bounds"])
        self.assertEqual(toeout_scales["squat_toe_out"], 2.0)
        self.assertEqual(toeout_algorithm["learning_rate"], 2.0e-4)
        self.assertEqual(toeout_algorithm["entropy_coef"], 0.005)
        self.assertEqual(toeout_runner["max_iterations"], 5000)
        self.assertEqual(toeout_runner["save_interval"], 200)
        self.assertEqual(toeout_runner["experiment_name"], "elf3_walk_toeout")

    def test_squat_toe_out_targets_and_distance_bounds(self):
        staged_training = load_staged_training_module()

        heights = torch.tensor([0.735, 0.65, 0.60, 0.50, 0.30])
        ratios = staged_training.squat_toe_out_ratio(heights, 0.735, 0.50)
        targets = staged_training.squat_toe_out_targets(
            heights, 0.735, 0.50, 15.0
        )
        target_degrees = torch.rad2deg(targets)
        expected_left = torch.tensor([0.0, 5.425532, 8.617021, 15.0, 15.0])
        self.assertTrue(torch.allclose(target_degrees[:, 0], expected_left, atol=1.0e-5))
        self.assertTrue(torch.allclose(target_degrees[:, 0], -target_degrees[:, 1]))
        self.assertTrue(
            torch.allclose(
                ratios, expected_left / 15.0, atol=1.0e-6
            )
        )

        distances = torch.tensor([0.19, 0.20, 0.275, 0.35, 0.36])
        rewards = staged_training.bounded_lateral_distance_reward(
            distances, 0.20, 0.35
        )
        self.assertTrue(torch.allclose(rewards[1:4], torch.zeros(3)))
        self.assertLess(float(rewards[0]), 0.0)
        self.assertLess(float(rewards[4]), 0.0)

    def test_toe_out_targets_are_invariant_under_mirror_augmentation(self):
        from rsl_rl.algorithms.him_ppo import HIMPPO

        action = torch.zeros((1, 12))
        action[0, 2] = 0.4
        action[0, 8] = -0.4
        mirrored = HIMPPO.flip_g1_actions(HIMPPO.__new__(HIMPPO), action)
        self.assertTrue(torch.equal(mirrored, action))

    def test_height_commands_are_bounded_and_slew_limited(self):
        staged_training = load_staged_training_module()

        torch.manual_seed(5)
        targets = staged_training.sample_height_targets(10000, 0.3, 1.0, 0.25, "cpu")
        self.assertGreaterEqual(float(targets.min()), 0.3)
        self.assertLessEqual(float(targets.max()), 1.0)
        self.assertAlmostEqual(float((targets == 0.3).float().mean()), 0.25, delta=0.03)
        self.assertAlmostEqual(float((targets == 1.0).float().mean()), 0.25, delta=0.03)

        current = torch.tensor([1.0, 0.3, 0.6])
        target = torch.tensor([0.3, 1.0, 0.61])
        updated = staged_training.slew_height_commands(current, target, 0.20, 0.02)
        self.assertTrue(torch.all(torch.abs(updated - current) <= 0.0040001))
        self.assertTrue(torch.all(updated >= 0.3))
        self.assertTrue(torch.all(updated <= 1.0))

    def test_walk_commands_are_half_stationary_and_bounded(self):
        staged_training = load_staged_training_module()

        torch.manual_seed(7)
        commands = staged_training.sample_walk_commands(
            10000,
            (-0.5, 0.5),
            (-0.3, 0.3),
            (-0.5, 0.5),
            0.5,
            "cpu",
        )
        moving = torch.any(commands != 0.0, dim=1)
        self.assertAlmostEqual(float(moving.float().mean()), 0.5, delta=0.03)
        self.assertTrue(torch.all(commands[:, 0].abs() <= 0.5))
        self.assertTrue(torch.all(commands[:, 1].abs() <= 0.3))
        self.assertTrue(torch.all(commands[:, 2].abs() <= 0.5))

    def test_stage2_commands_mix_squat_walk_and_standing(self):
        staged_training = load_staged_training_module()

        torch.manual_seed(11)
        squat_mask = staged_training.sample_squat_modes(20000, 0.5, "cpu")
        commands, height_targets = staged_training.sample_stage2_commands(
            squat_mask,
            (-0.5, 0.5),
            (-0.3, 0.3),
            (-0.5, 0.5),
            0.5,
            0.3,
            1.0,
            0.25,
            1.0,
        )

        moving = torch.any(commands != 0.0, dim=1)
        standing = ~squat_mask & ~moving
        self.assertAlmostEqual(float(squat_mask.float().mean()), 0.5, delta=0.03)
        self.assertAlmostEqual(float(moving.float().mean()), 0.25, delta=0.03)
        self.assertAlmostEqual(float(standing.float().mean()), 0.25, delta=0.03)
        self.assertTrue(torch.all(commands[squat_mask] == 0.0))
        self.assertTrue(torch.all(commands[:, 0].abs() <= 0.5))
        self.assertTrue(torch.all(commands[:, 1].abs() <= 0.3))
        self.assertTrue(torch.all(commands[:, 2].abs() <= 0.5))
        self.assertGreaterEqual(float(height_targets[squat_mask].min()), 0.3)
        self.assertLessEqual(float(height_targets[squat_mask].max()), 1.0)
        self.assertAlmostEqual(
            float((height_targets[squat_mask] == 0.3).float().mean()),
            0.25,
            delta=0.03,
        )
        self.assertAlmostEqual(
            float((height_targets[squat_mask] == 1.0).float().mean()),
            0.25,
            delta=0.03,
        )
        self.assertTrue(torch.all(height_targets[~squat_mask] == 1.0))

    def test_height_curriculum_requires_both_metrics_and_marks_ready(self):
        staged_training = load_staged_training_module()

        state = staged_training.evaluate_height_curriculum(0.0, 89, 100, 0.05, 0, 0.05, 0.9, 0.08, 5)
        self.assertEqual(state.ratio, 0.0)
        state = staged_training.evaluate_height_curriculum(0.0, 95, 100, 0.09, 0, 0.05, 0.9, 0.08, 5)
        self.assertEqual(state.ratio, 0.0)
        state = staged_training.evaluate_height_curriculum(0.0, 95, 100, 0.05, 0, 0.05, 0.9, 0.08, 5)
        self.assertEqual(state.ratio, 0.05)

        ratio = 0.95
        streak = 0
        ready = 0
        for _ in range(5):
            state = staged_training.evaluate_height_curriculum(
                ratio, 95, 100, 0.05, streak, 0.05, 0.9, 0.08, 5
            )
            ratio, streak, ready = state.ratio, state.qualified_windows, state.ready
        self.assertEqual(ratio, 1.0)
        self.assertEqual(streak, 5)
        self.assertEqual(ready, 1)

    def test_upper_body_height_amplification_is_linear(self):
        staged_training = load_staged_training_module()

        heights = torch.tensor([1.0, 0.65, 0.3, 0.1, 1.2])
        gains = staged_training.upper_height_amplification(heights, 0.3, 1.0, 1.5)
        self.assertTrue(torch.allclose(gains, torch.tensor([1.0, 1.25, 1.5, 1.5, 1.0])))

    def test_checkpoint_resume_pretrained_and_legacy_compatibility(self):
        from rsl_rl.runners.him_on_policy_runner import HIMOnPolicyRunner

        runner = HIMOnPolicyRunner.__new__(HIMOnPolicyRunner)
        runner.device = "cpu"
        runner.current_learning_iteration = 12
        runner.tot_timesteps = 10
        runner.tot_time = 2.0
        runner.env = DummyEnv()
        runner.env.curriculum_state["action_curriculum_ratio"] = 0.45
        runner.alg = SimpleNamespace(
            actor_critic=DummyState({"network": 1}),
            optimizer=DummyState({"policy_optimizer": 2}),
        )
        runner.alg.actor_critic.estimator = SimpleNamespace(
            optimizer=DummyState({"estimator_optimizer": 3})
        )
        runner.initial_optimizer_state_dict = {"policy_optimizer": 0}
        runner.initial_estimator_optimizer_state_dict = {"estimator_optimizer": 0}

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model_12.pt"
            runner.save(checkpoint)

            runner.current_learning_iteration = 0
            runner.env.reset_curriculum_state()
            runner.load(checkpoint)
            self.assertEqual(runner.current_learning_iteration, 13)
            self.assertEqual(runner.env.curriculum_state["action_curriculum_ratio"], 0.45)
            self.assertEqual(runner.alg.optimizer.state["policy_optimizer"], 2)

            runner.current_learning_iteration = 99
            runner.env.curriculum_state["action_curriculum_ratio"] = 0.8
            runner.load_pretrained(checkpoint)
            self.assertEqual(runner.current_learning_iteration, 0)
            self.assertEqual(runner.env.curriculum_state["action_curriculum_ratio"], 0.0)
            self.assertEqual(runner.alg.optimizer.state["policy_optimizer"], 0)
            self.assertEqual(
                runner.alg.actor_critic.estimator.optimizer.state["estimator_optimizer"], 0
            )
            self.assertEqual(runner.tot_timesteps, 0)
            self.assertEqual(runner.tot_time, 0)

            legacy = Path(tmp) / "legacy.pt"
            torch.save(
                {
                    "model_state_dict": {"network": 4},
                    "optimizer_state_dict": {"policy_optimizer": 5},
                    "estimator_optimizer_state_dict": {"estimator_optimizer": 6},
                    "iter": 7,
                    "infos": None,
                },
                legacy,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                runner.load(legacy)
            self.assertEqual(runner.current_learning_iteration, 7)
            self.assertTrue(any("课程状态" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
