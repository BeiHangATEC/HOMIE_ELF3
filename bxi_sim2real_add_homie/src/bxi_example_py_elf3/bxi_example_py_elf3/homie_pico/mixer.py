"""Pure command mixing for the HOMIE + PICO upper-body bridge."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


ARM_JOINTS = (
    "l_shoulder_y_joint",
    "l_shoulder_x_joint",
    "l_shoulder_z_joint",
    "l_elbow_y_joint",
    "l_wrist_x_joint",
    "l_wrist_y_joint",
    "l_wrist_z_joint",
    "r_shoulder_y_joint",
    "r_shoulder_x_joint",
    "r_shoulder_z_joint",
    "r_elbow_y_joint",
    "r_wrist_x_joint",
    "r_wrist_y_joint",
    "r_wrist_z_joint",
)
HEAD_JOINTS = ("head_y_joint", "head_z_joint")


FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class HomiePicoCommand:
    """One validated upper-body override sample."""

    arm_position: FloatArray
    arm_kp: FloatArray
    arm_kd: FloatArray
    head_position: FloatArray
    tracking_blend: FloatArray


class HomiePicoArmMixer:
    """Gate PICO arm targets on HOMIE state, freshness and grip input."""

    def __init__(
        self,
        *,
        target_state: str,
        nominal_arm_position: Sequence[float],
        arm_kp: Sequence[float],
        arm_kd: Sequence[float],
        state_timeout_s: float = 0.5,
        reference_timeout_s: float = 0.5,
        grip_timeout_s: float = 0.5,
        grip_threshold: float = 0.5,
        arm_gain_ramp_s: float = 0.4,
        head_control_enabled: bool = True,
        head_pitch_limit_rad: float = 0.5,
        head_yaw_limit_rad: float = 1.0,
        head_pitch_speed_rad_s: float = 1.5,
        head_yaw_speed_rad_s: float = 2.0,
        head_deadband_rad: float = 0.015,
    ) -> None:
        if not isinstance(target_state, str) or not target_state:
            raise ValueError("target_state must be a non-empty string")
        positive = {
            "state_timeout_s": state_timeout_s,
            "reference_timeout_s": reference_timeout_s,
            "grip_timeout_s": grip_timeout_s,
            "arm_gain_ramp_s": arm_gain_ramp_s,
            "head_pitch_limit_rad": head_pitch_limit_rad,
            "head_yaw_limit_rad": head_yaw_limit_rad,
            "head_pitch_speed_rad_s": head_pitch_speed_rad_s,
            "head_yaw_speed_rad_s": head_yaw_speed_rad_s,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(grip_threshold)) or not 0.0 <= grip_threshold <= 1.0:
            raise ValueError("grip_threshold must be within [0, 1]")
        if not math.isfinite(float(head_deadband_rad)) or head_deadband_rad < 0.0:
            raise ValueError("head_deadband_rad must be finite and non-negative")

        self.target_state = target_state
        self.state_timeout_s = float(state_timeout_s)
        self.reference_timeout_s = float(reference_timeout_s)
        self.grip_timeout_s = float(grip_timeout_s)
        self.grip_threshold = float(grip_threshold)
        self.arm_gain_ramp_s = float(arm_gain_ramp_s)
        self.head_control_enabled = bool(head_control_enabled)
        self.head_limits = np.asarray(
            (head_pitch_limit_rad, head_yaw_limit_rad), dtype=np.float32
        )
        self.head_speeds = np.asarray(
            (head_pitch_speed_rad_s, head_yaw_speed_rad_s), dtype=np.float32
        )
        self.head_deadband_rad = float(head_deadband_rad)

        self._nominal_arm = self._vector(nominal_arm_position, 14, "nominal arm")
        self._arm_kp = self._vector(arm_kp, 14, "arm kp")
        self._arm_kd = self._vector(arm_kd, 14, "arm kd")
        if np.any(self._arm_kp < 0.0) or np.any(self._arm_kd < 0.0):
            raise ValueError("arm gains must be non-negative")

        self._state_name = ""
        self._state_received_at: float | None = None
        self._reference_arm = self._nominal_arm.copy()
        self._reference_head = np.zeros(2, dtype=np.float32)
        self._reference_has_head = False
        self._reference_received_at: float | None = None
        self._grip = np.zeros(2, dtype=np.float32)
        self._grip_received_at: list[float | None] = [None, None]
        self._tracking_blend = np.zeros(2, dtype=np.float32)
        self._arm_output = self._nominal_arm.copy()
        self._head_output = np.zeros(2, dtype=np.float32)

    @staticmethod
    def _vector(values: Sequence[float], size: int, label: str) -> FloatArray:
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (size,) or not np.isfinite(result).all():
            raise ValueError(f"{label} must contain {size} finite values")
        return result.copy()

    @staticmethod
    def _timestamp(value: float) -> float:
        timestamp = float(value)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        return timestamp

    @property
    def tracking_blend(self) -> FloatArray:
        return self._tracking_blend.copy()

    def observe_state(self, state_name: str, *, received_at: float) -> None:
        if not isinstance(state_name, str) or not state_name:
            raise ValueError("state_name must be a non-empty string")
        self._state_name = state_name
        self._state_received_at = self._timestamp(received_at)

    def observe_reference(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        *,
        received_at: float,
    ) -> None:
        joint_names = tuple(names)
        if len(joint_names) != len(positions) or len(set(joint_names)) != len(
            joint_names
        ):
            raise ValueError("PICO reference names and positions must be unique and aligned")
        by_name = dict(zip(joint_names, positions))
        try:
            arm = self._vector(
                [by_name[name] for name in ARM_JOINTS], 14, "PICO arm reference"
            )
        except KeyError as exc:
            raise ValueError(f"PICO reference is missing arm joint {exc.args[0]}") from exc

        head_values = [by_name.get(name) for name in HEAD_JOINTS]
        has_head = all(value is not None for value in head_values)
        if has_head:
            head = self._vector(head_values, 2, "PICO head reference")
            np.copyto(self._reference_head, head)
        np.copyto(self._reference_arm, arm)
        self._reference_has_head = has_head
        self._reference_received_at = self._timestamp(received_at)

    def observe_grip(self, side: str, value: float, *, received_at: float) -> None:
        if side not in {"left", "right"}:
            raise ValueError("grip side must be 'left' or 'right'")
        grip = float(value)
        if not math.isfinite(grip):
            raise ValueError("grip value must be finite")
        index = 0 if side == "left" else 1
        self._grip[index] = float(np.clip(grip, 0.0, 1.0))
        self._grip_received_at[index] = self._timestamp(received_at)

    @staticmethod
    def _fresh(received_at: float | None, now: float, timeout_s: float) -> bool:
        return received_at is not None and 0.0 <= now - received_at <= timeout_s

    def _state_active(self, now: float) -> bool:
        return self._state_name == self.target_state and self._fresh(
            self._state_received_at, now, self.state_timeout_s
        )

    def _advance_blends(self, active: tuple[bool, bool], dt: float) -> None:
        maximum_step = dt / self.arm_gain_ramp_s
        for index, enabled in enumerate(active):
            target = 1.0 if enabled else 0.0
            self._tracking_blend[index] += float(
                np.clip(
                    target - self._tracking_blend[index],
                    -maximum_step,
                    maximum_step,
                )
            )

    def _advance_head(self, desired: FloatArray, dt: float) -> None:
        target = np.clip(desired, -self.head_limits, self.head_limits).copy()
        target[np.abs(target) < self.head_deadband_rad] = 0.0
        maximum_step = self.head_speeds * dt
        self._head_output += np.clip(
            target - self._head_output,
            -maximum_step,
            maximum_step,
        )

    def step(self, *, now: float, dt: float) -> HomiePicoCommand | None:
        timestamp = self._timestamp(now)
        elapsed = float(dt)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("dt must be finite and non-negative")

        state_active = self._state_active(timestamp)
        if not state_active:
            self._tracking_blend.fill(0.0)
            self._head_output.fill(0.0)
            return None

        reference_fresh = self._fresh(
            self._reference_received_at, timestamp, self.reference_timeout_s
        )
        arm_active = []
        for index in range(2):
            grip_fresh = self._fresh(
                self._grip_received_at[index], timestamp, self.grip_timeout_s
            )
            arm_active.append(
                reference_fresh
                and grip_fresh
                and self._grip[index] > self.grip_threshold
            )
        self._advance_blends((arm_active[0], arm_active[1]), elapsed)

        for side in range(2):
            start = side * 7
            end = start + 7
            blend = self._tracking_blend[side]
            self._arm_output[start:end] = self._nominal_arm[start:end] + blend * (
                self._reference_arm[start:end] - self._nominal_arm[start:end]
            )

        head_live = (
            self.head_control_enabled
            and reference_fresh
            and self._reference_has_head
        )
        desired_head = (
            self._reference_head if head_live else np.zeros(2, dtype=np.float32)
        )
        self._advance_head(desired_head, elapsed)

        if (
            not any(arm_active)
            and float(np.max(self._tracking_blend)) <= 1.0e-6
            and float(np.max(np.abs(self._head_output))) <= 1.0e-6
        ):
            return None
        return HomiePicoCommand(
            arm_position=self._arm_output.copy(),
            arm_kp=self._arm_kp.copy(),
            arm_kd=self._arm_kd.copy(),
            head_position=self._head_output.copy(),
            tracking_blend=self._tracking_blend.copy(),
        )


__all__ = [
    "ARM_JOINTS",
    "HEAD_JOINTS",
    "HomiePicoArmMixer",
    "HomiePicoCommand",
]
