from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bxi_example_py_elf3.framework.joints import JointLayout, JointStateBuffer
from bxi_example_py_elf3.framework.platform.api import RobotObservation
from bxi_example_py_elf3.framework.runtime.controller import RobotControlFramework


def _observation(raw_height_rate: object = 0.0) -> RobotObservation:
    joints = JointStateBuffer(JointLayout(("joint",))).view
    return RobotObservation(
        joints=joints,
        quat_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64),
        quat_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
        omega=np.zeros(3, dtype=np.float64),
        raw_cmd_vel=np.asarray((0.1, -0.2, 0.3), dtype=np.float32),
        raw_height_rate=raw_height_rate,
    )


def _framework_for(observation: RobotObservation) -> RobotControlFramework:
    framework = object.__new__(RobotControlFramework)
    framework._robot_layout = observation.joints.layout
    framework._robot_joints = None
    framework._inference_frame = SimpleNamespace(
        joints=None,
        timestamp_ns=0,
        linear_acceleration=None,
    )
    framework.current_quat_xyzw = np.zeros(4, dtype=np.float64)
    framework.current_quat_wxyz = np.zeros(4, dtype=np.float64)
    framework.current_omega = np.zeros(3, dtype=np.float64)
    framework.current_linear_acceleration = np.zeros(3, dtype=np.float64)
    framework.current_raw_cmd_vel = np.zeros(3, dtype=np.float32)
    framework.current_raw_height_rate = 0.0
    framework._initial_state_entered = True
    return framework


def test_robot_observation_defaults_height_rate_without_changing_raw_velocity() -> None:
    observation = _observation()

    assert observation.raw_height_rate == 0.0
    assert observation.raw_cmd_vel.shape == (3,)


def test_robot_observation_preserves_linear_acceleration_positional_argument() -> None:
    observation = _observation()
    linear_acceleration = np.asarray((0.0, 0.0, 9.81), dtype=np.float64)

    positional_observation = RobotObservation(
        observation.joints,
        observation.quat_xyzw,
        observation.quat_wxyz,
        observation.omega,
        observation.raw_cmd_vel,
        linear_acceleration,
    )

    assert positional_observation.linear_acceleration is linear_acceleration
    assert positional_observation.raw_height_rate == 0.0


def test_set_observation_copies_scalar_height_rate() -> None:
    observation = _observation(np.asarray(-0.25, dtype=np.float32))
    framework = _framework_for(observation)

    framework._set_observation(observation)

    assert framework.current_raw_height_rate == pytest.approx(-0.25)
    np.testing.assert_allclose(framework.current_raw_cmd_vel, (0.1, -0.2, 0.3))


@pytest.mark.parametrize(
    "raw_height_rate",
    (np.nan, np.inf, -np.inf, np.asarray((0.2,)), "0.2", 1.0 + 0.0j),
)
def test_set_observation_rejects_non_finite_or_non_scalar_height_rate(
    raw_height_rate: object,
) -> None:
    observation = _observation(raw_height_rate)
    framework = _framework_for(observation)

    with pytest.raises(ValueError, match="raw_height_rate must be a finite scalar"):
        framework._set_observation(observation)
