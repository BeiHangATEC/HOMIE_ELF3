import numpy as np
import pytest

from bxi_example_py_elf3.homie_pico.mixer import ARM_JOINTS
from bxi_example_py_elf3.homie_pico.runtime_support import ArmGravityCompensator


class _GravityModel:
    def __init__(self, output=None):
        self.output = np.arange(14, dtype=np.float64) if output is None else output
        self.positions = None
        self.gravity = None

    def compute(self, positions, gravity, output):
        self.positions = positions.copy()
        self.gravity = gravity.copy()
        output[:] = self.output


def _compensator(model=None, **overrides):
    kwargs = {
        "gravity_scale": 1.0,
        "torque_limit_scale": 0.8,
        "sample_timeout_s": 0.5,
    }
    kwargs.update(overrides)
    return ArmGravityCompensator(
        model or _GravityModel(),
        np.full(7, 10.0),
        **kwargs,
    )


def test_requires_fresh_joint_and_imu_feedback():
    compensator = _compensator()
    assert compensator.compute(now=1.0) is None

    compensator.observe_joint_state(
        ARM_JOINTS, np.zeros(14), received_at=1.0
    )
    assert compensator.compute(now=1.0) is None

    compensator.observe_orientation_xyzw((0.0, 0.0, 0.0, 1.0), received_at=1.0)
    assert compensator.compute(now=1.6) is None


def test_identity_orientation_projects_standard_gravity_and_clips_effort():
    model = _GravityModel()
    compensator = _compensator(model)
    positions = np.linspace(-0.5, 0.5, 14)
    compensator.observe_joint_state(ARM_JOINTS, positions, received_at=1.0)
    compensator.observe_orientation_xyzw((0.0, 0.0, 0.0, 2.0), received_at=1.0)

    output = compensator.compute(now=1.1)

    np.testing.assert_allclose(model.positions, positions)
    np.testing.assert_allclose(model.gravity, (0.0, 0.0, -9.80665))
    np.testing.assert_allclose(output, np.clip(np.arange(14), -8.0, 8.0))


def test_rejects_incomplete_named_joint_feedback():
    compensator = _compensator()
    with pytest.raises(ValueError, match="missing"):
        compensator.observe_joint_state(
            ARM_JOINTS[:-1], np.zeros(13), received_at=1.0
        )


def test_rejects_zero_norm_imu_quaternion():
    compensator = _compensator()
    with pytest.raises(ValueError, match="zero norm"):
        compensator.observe_orientation_xyzw((0.0, 0.0, 0.0, 0.0), received_at=1.0)
