from __future__ import annotations

import json

from bxi_example_py_elf3.homie_pico.hardware_preflight import (
    HARDWARE_JOINTS,
    HardwareReadiness,
    decode_motor_receive_mask,
)
from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS


def test_motor_receive_mask_603fffff_only_misses_right_arm_bus4():
    report = decode_motor_receive_mask(0x603FFFFF)

    assert report["ready"] is False
    assert report["received_count"] == 22
    assert report["missing"] == list(HARDWARE_JOINTS[22:29])
    assert report["disabled"] == list(HARDWARE_JOINTS[29:31])
    assert report["by_bus"]["can4"]["received"] == []
    assert len(report["by_bus"]["can4"]["missing"]) == 7


def test_motor_receive_mask_complete_for_enabled_motors():
    report = decode_motor_receive_mask(0x7FFFFFFF)

    assert report["ready"] is True
    assert report["missing_count"] == 0
    assert report["received_count"] == 29


def test_live_readiness_accepts_fresh_full_rate_control_chain():
    readiness = HardwareReadiness()
    names = ELF3_POLICY_JOINTS.names
    zeros = [0.0] * len(names)
    start = 10.0

    for index in range(101):
        now = start + index * 0.02
        readiness.observe_actuator_states(names, zeros, now=now)
        readiness.observe_imu((0.0, 0.0, 0.0, 1.0), now=now)
        readiness.observe_actuator_commands(
            names, zeros, zeros, zeros, zeros, zeros, now=now
        )
        if index % 5 == 0:
            readiness.observe_state_machine(
                json.dumps({"current": {"name": "com.bxi.basic_actions/normal"}}),
                now=now,
            )

    report = readiness.report(now=12.0)
    assert report["ready"] is True
    assert report["current_state"] == "com.bxi.basic_actions/normal"


def test_live_readiness_rejects_missing_lower_body_joint_and_bad_imu():
    readiness = HardwareReadiness()
    names = ELF3_POLICY_JOINTS.names[1:]
    zeros = [0.0] * len(names)

    for index in range(101):
        now = 10.0 + index * 0.02
        readiness.observe_actuator_states(names, zeros, now=now)
        readiness.observe_imu((0.0, 0.0, 0.0, 0.0), now=now)

    report = readiness.report(now=12.0)
    assert report["ready"] is False
    assert "waist_y_joint" in report["topics"]["actuator_states"]["detail"]
    assert "norm=0.000000" in report["topics"]["imu"]["detail"]
