"""The ELF3 height/velocity stage ladder.

Ordered so training starts from the pose the robot naturally holds and works
*down*, then widens the velocity envelope. The previous attempt ran this
backwards -- its first stage commanded 0.34 m, which needs roughly -2.88 rad of
hip pitch, the most contorted pose in the whole workspace. Training began at
its hardest point and got easier from there.

Heights are torso-origin-to-sole distances in metres, and every one of them is
inside what the legs actually reach with the hip within about -1.2 rad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping


#: Deepest crouch the ladder ever asks for. Deeper is kinematically possible
#: but demands extreme hip pitch, so it is deliberately out of scope.
CROUCH_MIN_HEIGHT = 0.78
CROUCH_LOW_NOMINAL_MAX_HEIGHT = 0.90


@dataclass(frozen=True)
class StageSpec:
    name: str
    walk_height: float
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_yaw: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "walk_height": self.walk_height,
            "lin_vel_x": list(self.lin_vel_x),
            "lin_vel_y": list(self.lin_vel_y),
            "ang_vel_yaw": list(self.ang_vel_yaw),
        }


#: S* stages descend in height at a narrow velocity envelope; V* stages hold
#: the standing height and widen the envelope.
STAGE_ORDER = ("S0", "S1", "S2", "S3", "S4", "S5", "V1", "V2", "V3")

_NARROW_X = (-0.20, 0.30)
_NARROW_Y = (-0.10, 0.10)
_NARROW_YAW = (-0.20, 0.20)

_STAGE_SPECS = {
    # Learn to stand and step at the natural pose first.
    "S0": StageSpec("S0", 1.01, _NARROW_X, _NARROW_Y, _NARROW_YAW),
    "S1": StageSpec("S1", 0.97, _NARROW_X, _NARROW_Y, _NARROW_YAW),
    "S2": StageSpec("S2", 0.93, _NARROW_X, _NARROW_Y, _NARROW_YAW),
    "S3": StageSpec("S3", 0.89, _NARROW_X, _NARROW_Y, _NARROW_YAW),
    "S4": StageSpec("S4", 0.85, _NARROW_X, _NARROW_Y, _NARROW_YAW),
    "S5": StageSpec("S5", 0.80, _NARROW_X, _NARROW_Y, _NARROW_YAW),
    # Then go fast, back at standing height.
    "V1": StageSpec("V1", 1.01, (-0.40, 0.60), (-0.20, 0.20), (-0.40, 0.40)),
    "V2": StageSpec("V2", 1.01, (-0.60, 0.90), (-0.35, 0.35), (-0.60, 0.60)),
    "V3": StageSpec("V3", 1.01, (-0.80, 1.20), (-0.50, 0.50), (-0.80, 0.80)),
}

STAGE_SPECS: Mapping[str, StageSpec] = MappingProxyType(_STAGE_SPECS)


def get_stage(stage: str | StageSpec) -> StageSpec:
    if isinstance(stage, StageSpec):
        expected = STAGE_SPECS.get(stage.name)
        if expected is None or stage != expected:
            raise ValueError("ELF3 command stage must match a canonical stage")
        return expected
    if not isinstance(stage, str):
        raise TypeError("ELF3 command stage must be a name or StageSpec")
    try:
        return STAGE_SPECS[stage]
    except KeyError as exc:
        raise ValueError(f"Unknown ELF3 command stage: {stage}") from exc


def next_stage(stage: str | StageSpec) -> str | None:
    index = STAGE_ORDER.index(get_stage(stage).name)
    return STAGE_ORDER[index + 1] if index + 1 < len(STAGE_ORDER) else None


def canonical_stage_definitions() -> list[dict[str, Any]]:
    return [STAGE_SPECS[name].to_dict() for name in STAGE_ORDER]


def effective_crouch_low_bounds(
    stage: str | StageSpec,
    *,
    minimum: float = CROUCH_MIN_HEIGHT,
    nominal_maximum: float = CROUCH_LOW_NOMINAL_MAX_HEIGHT,
) -> tuple[float, float]:
    """Resolve the shallow-crouch band without exceeding the stage height."""
    for label, value in (("minimum", minimum), ("nominal maximum", nominal_maximum)):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"crouch {label} must be a real number")
        if not math.isfinite(float(value)):
            raise ValueError(f"crouch {label} must be finite")
    minimum = float(minimum)
    nominal_maximum = float(nominal_maximum)
    if minimum > nominal_maximum:
        raise ValueError("crouch minimum must not exceed its nominal maximum")

    spec = get_stage(stage)
    if minimum > spec.walk_height:
        raise ValueError("crouch minimum must not exceed the stage walk height")
    return minimum, min(nominal_maximum, spec.walk_height)
