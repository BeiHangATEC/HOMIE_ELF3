"""Contracts for the ELF3 V3 height-and-motion demonstration timeline."""

from __future__ import annotations

import importlib

import pytest


def _api():
    return importlib.import_module(
        "openhomie_isaaclab.workflows.elf3_c3_motion_demo"
    )


def test_motion_timeline_is_the_user_requested_33_second_sequence():
    module = _api()

    assert module.FPS == 50
    assert module.STEPS == 1650
    assert module.DURATION_SECONDS == 33.0
    assert module.LOW_HEIGHT == 0.30
    assert module.OUT_OF_DISTRIBUTION_HEIGHT is True

    expected = {
        0: ("stand_initial", (0.0, 0.0, 0.0, 1.01), 1),
        99: ("stand_initial", (0.0, 0.0, 0.0, 1.01), 1),
        100: ("crouch_down", (0.0, 0.0, 0.0, 1.01), 2),
        349: ("crouch_down", (0.0, 0.0, 0.0, 0.30), 2),
        350: ("crouch_hold", (0.0, 0.0, 0.0, 0.30), 2),
        449: ("crouch_hold", (0.0, 0.0, 0.0, 0.30), 2),
        450: ("stand_up", (0.0, 0.0, 0.0, 0.30), 2),
        699: ("stand_up", (0.0, 0.0, 0.0, 1.01), 2),
        700: ("walk_forward", (0.5, 0.0, 0.0, 1.01), 0),
        949: ("walk_forward", (0.5, 0.0, 0.0, 1.01), 0),
        950: ("stand_middle", (0.0, 0.0, 0.0, 1.01), 1),
        1049: ("stand_middle", (0.0, 0.0, 0.0, 1.01), 1),
        1050: ("walk_backward", (-0.3, 0.0, 0.0, 1.01), 0),
        1299: ("walk_backward", (-0.3, 0.0, 0.0, 1.01), 0),
        1300: ("stand_before_turn", (0.0, 0.0, 0.0, 1.01), 1),
        1399: ("stand_before_turn", (0.0, 0.0, 0.0, 1.01), 1),
        1400: ("turn_positive", (0.0, 0.0, 0.5, 1.01), 0),
        1649: ("turn_positive", (0.0, 0.0, 0.5, 1.01), 0),
    }
    for step, (name, command, mode) in expected.items():
        entry = module.command_at_step(step)
        assert entry.name == name
        assert entry.mode == mode
        assert entry.command == pytest.approx(command)


@pytest.mark.parametrize("step", (-1, 1650))
def test_motion_timeline_rejects_steps_outside_the_recording(step):
    with pytest.raises(ValueError, match="step"):
        _api().command_at_step(step)
