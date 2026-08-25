import numpy as np
import pytest

from bxi_example_py_elf3.homie_pico.mixer import (
    ARM_JOINTS,
    HEAD_JOINTS,
    HomiePicoArmMixer,
)


TARGET_STATE = "com.bxi.homie/homie"


def _mixer(**overrides):
    kwargs = {
        "target_state": TARGET_STATE,
        "nominal_arm_position": np.zeros(14),
        "arm_kp": np.arange(1, 15),
        "arm_kd": np.ones(14),
        "state_timeout_s": 0.5,
        "reference_timeout_s": 0.5,
        "grip_timeout_s": 0.5,
        "arm_gain_ramp_s": 0.4,
    }
    kwargs.update(overrides)
    return HomiePicoArmMixer(**kwargs)


def _reference(mixer, *, now=1.0, arm_value=1.0, head=(0.4, -0.6)):
    mixer.observe_reference(
        (*ARM_JOINTS, *HEAD_JOINTS),
        (*([arm_value] * 14), *head),
        received_at=now,
    )


def test_requires_homie_state_before_publishing():
    mixer = _mixer()
    _reference(mixer)
    mixer.observe_grip("left", 1.0, received_at=1.0)
    assert mixer.step(now=1.0, dt=0.1) is None


def test_left_grip_ramps_only_left_arm_and_head():
    mixer = _mixer()
    mixer.observe_state(TARGET_STATE, received_at=1.0)
    _reference(mixer)
    mixer.observe_grip("left", 1.0, received_at=1.0)
    mixer.observe_grip("right", 0.0, received_at=1.0)

    command = mixer.step(now=1.0, dt=0.1)

    assert command is not None
    np.testing.assert_allclose(command.tracking_blend, (0.25, 0.0))
    np.testing.assert_allclose(command.arm_position[:7], 0.25)
    np.testing.assert_allclose(command.arm_position[7:], 0.0)
    np.testing.assert_allclose(command.head_position, (0.15, -0.2))


def test_reference_timeout_blends_to_policy_then_releases():
    mixer = _mixer()
    mixer.observe_state(TARGET_STATE, received_at=1.0)
    _reference(mixer)
    mixer.observe_grip("left", 1.0, received_at=1.0)
    assert mixer.step(now=1.0, dt=0.4) is not None

    mixer.observe_state(TARGET_STATE, received_at=1.6)
    assert mixer.step(now=1.6, dt=0.2) is not None
    assert mixer.step(now=1.8, dt=0.2) is None


def test_leaving_homie_releases_immediately():
    mixer = _mixer()
    mixer.observe_state(TARGET_STATE, received_at=1.0)
    _reference(mixer)
    mixer.observe_grip("right", 1.0, received_at=1.0)
    assert mixer.step(now=1.0, dt=0.1) is not None

    mixer.observe_state("com.bxi.basic_actions/pd_brake", received_at=1.1)
    assert mixer.step(now=1.1, dt=0.1) is None
    np.testing.assert_allclose(mixer.tracking_blend, 0.0)


def test_missing_named_arm_joint_is_rejected():
    mixer = _mixer()
    with pytest.raises(ValueError, match="missing arm joint"):
        mixer.observe_reference(
            ARM_JOINTS[:-1],
            [0.0] * 13,
            received_at=1.0,
        )
