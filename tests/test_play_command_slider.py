import ast
import importlib.util
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDER_PATH = (
    REPO_ROOT
    / "HomieRL/legged_gym/legged_gym/utils/command_slider.py"
)
ROBOT_PATH = (
    REPO_ROOT
    / "HomieRL/legged_gym/legged_gym/envs/base/legged_robot.py"
)


def load_slider_module():
    spec = importlib.util.spec_from_file_location("elf3_command_slider", SLIDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SliderCommandTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.slider = load_slider_module()

    def test_initial_values_and_invalid_initial_values(self):
        command = self.slider.SliderCommand(
            x_vel=0.2, y_vel=-0.1, yaw_vel=0.3, height=0.8
        )
        self.assertEqual(
            command.snapshot(),
            {"x_vel": 0.2, "y_vel": -0.1, "yaw_vel": 0.3, "height": 0.8},
        )
        with self.assertRaises(ValueError):
            self.slider.SliderCommand(height=1.02)
        with self.assertRaises(ValueError):
            self.slider.SliderCommand(x_vel=float("nan"))

    def test_updates_are_clamped_and_mapped_to_environment(self):
        command = self.slider.SliderCommand()
        command.set_values(x_vel=5.0, y_vel=-5.0, yaw_vel=2.0, height=0.1)
        snapshot = command.snapshot()
        self.assertEqual(
            snapshot,
            {"x_vel": 1.2, "y_vel": -0.5, "yaw_vel": 0.8, "height": 0.3},
        )

        env = SimpleNamespace(
            commands=np.zeros((3, 5), dtype=np.float32),
            height_command_targets=np.zeros(3, dtype=np.float32),
        )
        self.slider.apply_command_snapshot(env, snapshot)
        np.testing.assert_allclose(env.commands[:, 0], 1.2)
        np.testing.assert_allclose(env.commands[:, 1], -0.5)
        np.testing.assert_allclose(env.commands[:, 2], 0.8)
        np.testing.assert_allclose(env.commands[:, 3], 0.0)
        np.testing.assert_allclose(env.commands[:, 4], 0.3)
        np.testing.assert_allclose(env.height_command_targets, 0.3)

    def test_custom_limits_support_task_specific_and_fixed_commands(self):
        limits = {
            "x_vel": (-0.5, 0.5),
            "y_vel": (-0.3, 0.3),
            "yaw_vel": (-0.5, 0.5),
            "height": (1.0, 1.0),
        }
        command = self.slider.SliderCommand(height=1.0, limits=limits)
        command.set_values(x_vel=1.0, y_vel=-1.0, yaw_vel=0.2, height=0.5)
        self.assertEqual(command.limits, limits)
        self.assertEqual(
            command.snapshot(),
            {"x_vel": 0.5, "y_vel": -0.3, "yaw_vel": 0.2, "height": 1.0},
        )
        with self.assertRaises(ValueError):
            self.slider.SliderCommand(height=0.8, limits=limits)
        with self.assertRaises(ValueError):
            self.slider.SliderCommand(limits={"x_vel": (-1.0, 1.0)})

    def test_mode_switch_clamps_commands_to_active_training_distribution(self):
        limits = {
            "x_vel": (-0.5, 0.5),
            "y_vel": (-0.3, 0.3),
            "yaw_vel": (-0.5, 0.5),
            "height": (0.3, 1.0),
        }
        mode_limits = {
            "walk": {**limits, "height": (1.0, 1.0)},
            "squat": {
                "x_vel": (0.0, 0.0),
                "y_vel": (0.0, 0.0),
                "yaw_vel": (0.0, 0.0),
                "height": (0.3, 1.0),
            },
        }
        command = self.slider.SliderCommand(
            x_vel=0.2,
            height=1.0,
            limits=limits,
            mode_limits=mode_limits,
            mode="walk",
        )
        command.set_mode("squat")
        command.set_values(height=0.55)
        values, mode = command.state_snapshot()
        self.assertEqual(mode, "squat")
        self.assertEqual(values["x_vel"], 0.0)
        self.assertEqual(values["height"], 0.55)

        command.set_mode("walk")
        values, mode = command.state_snapshot()
        self.assertEqual(mode, "walk")
        self.assertEqual(values["height"], 1.0)
        self.assertEqual(command.active_limits()["height"], (1.0, 1.0))
        with self.assertRaises(ValueError):
            command.set_mode("unsupported")

    def test_reset_speeds_and_restore_defaults(self):
        command = self.slider.SliderCommand(
            x_vel=0.2, y_vel=-0.1, yaw_vel=0.3, height=0.8
        )
        command.set_values(x_vel=0.9, y_vel=0.4, yaw_vel=-0.6, height=0.5)
        command.reset_speeds()
        self.assertEqual(
            command.snapshot(),
            {"x_vel": 0.0, "y_vel": 0.0, "yaw_vel": 0.0, "height": 0.5},
        )
        command.restore_defaults()
        self.assertEqual(
            command.snapshot(),
            {"x_vel": 0.2, "y_vel": -0.1, "yaw_vel": 0.3, "height": 0.8},
        )

    def test_concurrent_updates_and_snapshots_stay_bounded(self):
        command = self.slider.SliderCommand()
        errors = []

        def writer(offset):
            try:
                for index in range(1000):
                    value = (index + offset) / 100.0 - 5.0
                    command.set_values(
                        x_vel=value,
                        y_vel=-value,
                        yaw_vel=value,
                        height=value,
                    )
            except BaseException as error:
                errors.append(error)

        def reader():
            try:
                for _ in range(1000):
                    snapshot = command.snapshot()
                    for name, value in snapshot.items():
                        lower, upper = self.slider.COMMAND_LIMITS[name]
                        self.assertLessEqual(lower, value)
                        self.assertLessEqual(value, upper)
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=writer, args=(offset,)) for offset in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])


class ManualCommandModeStaticTest(unittest.TestCase):
    def test_manual_mode_skips_periodic_and_reset_resampling(self):
        source = ROBOT_PATH.read_text(encoding="utf-8")
        module = ast.parse(source)
        robot = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "LeggedRobot"
        )
        methods = {
            node.name: ast.get_source_segment(source, node)
            for node in robot.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn('getattr(self.cfg.commands, "use_random", True)', methods["reset_idx"])
        self.assertIn('not getattr(self.cfg.commands, "use_random", True)', methods["_resample_commands"])


if __name__ == "__main__":
    unittest.main()
