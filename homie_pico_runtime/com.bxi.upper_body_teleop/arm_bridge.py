"""SONIC-compatible PICO pose receiver with Pinocchio dual-arm IK."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pinocchio as pin
import rclpy
import zmq
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from bxi_example_py_elf3.framework.mod_api import NodeBuildContext


_REQUIRED_PINOCCHIO_API = (
    "buildModelFromUrdf",
    "forwardKinematics",
    "computeJointJacobians",
    "integrate",
)
_missing_pinocchio_api = tuple(
    name for name in _REQUIRED_PINOCCHIO_API if not callable(getattr(pin, name, None))
)
if _missing_pinocchio_api:
    raise ImportError(
        "robotics Pinocchio is required; imported incompatible module "
        f"{getattr(pin, '__file__', '<unknown>')} missing API: "
        + ", ".join(_missing_pinocchio_api)
    )


HEADER_SIZE = 1280
POSE_STREAM_MODE = 1
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
BUTTON_FIELDS = (
    "left_trigger",
    "right_trigger",
    "left_grip",
    "right_grip",
)
DTYPE_MAP = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
    "bool": np.dtype("?"),
}
DEFAULTS: dict[str, object] = {
    "pico_host": "127.0.0.1",
    "pico_port": 5556,
    "pico_topic": "pose",
    "robot_state_topic": "pico_control_joint_states",
    "reference_topic": "pico_control_joint_commands",
    "rate_hz": 50.0,
    "stale_timeout_s": 0.5,
    "required_consecutive_frames": 3,
    "ik_iterations": 24,
    "ik_damping": 0.001,
    "ik_step_size": 0.7,
    "ik_tolerance": 1.0e-4,
    "maximum_position_error_m": 0.003,
    "maximum_orientation_error_rad": 0.03,
    "maximum_joint_step_rad": 0.12,
    "joint_limit_margin_rad": 0.0872665,
    "joint_centering_gain": 0.005,
    "swivel_continuity_gain": 0.02,
    "swivel_min_radius_m": 0.02,
}


def _decode_packed_message(message: bytes, topic: str) -> dict[str, np.ndarray] | None:
    prefix = topic.encode("utf-8")
    if not message.startswith(prefix):
        return None
    payload = message[len(prefix) :]
    if len(payload) < HEADER_SIZE:
        raise ValueError("PICO packet is shorter than its fixed header")
    raw_header = payload[:HEADER_SIZE].split(b"\x00", 1)[0]
    if not raw_header:
        raise ValueError("PICO packet has an empty header")
    header: dict[str, Any] = json.loads(raw_header.decode("utf-8"))
    data = memoryview(payload[HEADER_SIZE:])
    fields: dict[str, np.ndarray] = {}
    offset = 0
    for description in header.get("fields", ()):
        name = str(description["name"])
        dtype = DTYPE_MAP.get(str(description["dtype"]))
        if dtype is None:
            raise ValueError(
                f"unsupported PICO dtype for {name}: {description['dtype']}"
            )
        shape = tuple(int(value) for value in description.get("shape", ()))
        count = int(np.prod(shape)) if shape else 1
        byte_count = dtype.itemsize * count
        if offset + byte_count > len(data):
            raise ValueError(f"PICO field {name} exceeds payload bounds")
        values = np.frombuffer(
            data[offset : offset + byte_count],
            dtype=dtype,
            count=count,
        )
        fields[name] = values.reshape(shape).copy()
        offset += byte_count
    return fields


def _scalar(fields: dict[str, np.ndarray], name: str) -> float | None:
    try:
        values = np.asarray(fields[name], dtype=np.float64).reshape(-1)
    except (KeyError, TypeError, ValueError):
        return None
    if values.size != 1 or not math.isfinite(float(values[0])):
        return None
    return float(values[0])


def _rotation_from_wxyz(value: object) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    if not np.isfinite(quaternion).all():
        raise ValueError("PICO wrist quaternion contains non-finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-8:
        raise ValueError("PICO wrist quaternion has zero norm")
    w, x, y, z = quaternion / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


class SourceGate:
    """Accept only calibrated, progressing SONIC POSE messages."""

    def __init__(self, required_consecutive: int) -> None:
        self.required_consecutive = max(1, int(required_consecutive))
        self.streak = 0
        self.last_frame: int | None = None

    def observe(self, fields: dict[str, np.ndarray]) -> bool:
        try:
            mode = int(np.asarray(fields["stream_mode"]).reshape(-1)[-1])
            calibrated = bool(np.asarray(fields["calibration_ready"]).reshape(-1)[-1])
            newest = int(np.asarray(fields["frame_index"]).reshape(-1)[-1])
            position = np.asarray(fields["vr_position"], dtype=np.float64).reshape(3, 3)
            orientation = np.asarray(
                fields["vr_orientation"], dtype=np.float64
            ).reshape(3, 4)
        except (KeyError, TypeError, ValueError, IndexError):
            self.streak = 0
            return False
        if (
            mode != POSE_STREAM_MODE
            or not calibrated
            or not np.isfinite(position).all()
            or not np.isfinite(orientation).all()
        ):
            self.streak = 0
            return False
        if self.last_frame is not None:
            if newest == self.last_frame:
                return False
            if newest < self.last_frame:
                self.streak = 0
        self.last_frame = newest
        self.streak += 1
        return self.streak >= self.required_consecutive


@dataclass(frozen=True, slots=True)
class ArmIkResult:
    """Validated dual-arm IK outcome.

    A failed solve deliberately carries no command.  The bridge then stops
    refreshing the reference topic, allowing the state-side freshness gate to
    blend both arms back to their policy PD pose.
    """

    success: bool
    positions: np.ndarray | None
    reason: str
    position_error_m: float
    orientation_error_rad: float
    maximum_joint_delta_rad: float
    minimum_limit_margin_rad: float
    iterations: int
    maximum_swivel_delta_rad: float = math.inf


class PinocchioArmIk:
    """Validated DLS IK for both arms in the waist-z anchor frame."""

    def __init__(
        self,
        urdf_path: str,
        *,
        iterations: int,
        damping: float,
        step_size: float,
        tolerance: float,
        maximum_joint_step: float,
        maximum_position_error: float = 0.003,
        maximum_orientation_error: float = 0.03,
        joint_limit_margin: float = 0.0872665,
        joint_centering_gain: float = 0.005,
        swivel_continuity_gain: float = 0.02,
        swivel_min_radius: float = 0.02,
    ) -> None:
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.iterations = max(1, int(iterations))
        self.damping = float(damping)
        self.step_size = float(step_size)
        self.tolerance = float(tolerance)
        self.maximum_joint_step = float(maximum_joint_step)
        self.maximum_position_error = float(maximum_position_error)
        self.maximum_orientation_error = float(maximum_orientation_error)
        self.joint_limit_margin = float(joint_limit_margin)
        self.joint_centering_gain = float(joint_centering_gain)
        self.swivel_continuity_gain = float(swivel_continuity_gain)
        self.swivel_min_radius = float(swivel_min_radius)
        if (
            min(
                self.damping,
                self.step_size,
                self.tolerance,
                self.maximum_joint_step,
                self.maximum_position_error,
                self.maximum_orientation_error,
                self.swivel_min_radius,
            )
            <= 0.0
        ):
            raise ValueError("Pinocchio IK numeric parameters must be positive")
        if (
            not math.isfinite(self.joint_centering_gain)
            or self.joint_centering_gain < 0.0
        ):
            raise ValueError("joint_centering_gain must be finite and non-negative")
        if not math.isfinite(self.joint_limit_margin) or self.joint_limit_margin < 0.0:
            raise ValueError("joint_limit_margin must be finite and non-negative")
        if (
            not math.isfinite(self.swivel_continuity_gain)
            or self.swivel_continuity_gain < 0.0
        ):
            raise ValueError("swivel_continuity_gain must be finite and non-negative")
        self.q = pin.neutral(self.model)
        self._measured = self.q.copy()
        self._seed = self.q.copy()
        self._has_measurement = False
        self._velocity = np.zeros(self.model.nv, dtype=np.float64)
        self._damping_identity = np.eye(6, dtype=np.float64) * self.damping
        self._arm_identity = np.eye(7, dtype=np.float64)
        self._name_to_q = self._compile_configuration_indices()
        self._arm_q_indices = np.asarray(
            [self._name_to_q[name] for name in ARM_JOINTS], dtype=np.intp
        )
        self._arm_v_indices = {
            "left": self._velocity_indices(ARM_JOINTS[:7]),
            "right": self._velocity_indices(ARM_JOINTS[7:]),
        }
        self._arm_side_q_indices = {
            "left": self._arm_q_indices[:7],
            "right": self._arm_q_indices[7:],
        }
        hard_lower = self.model.lowerPositionLimit[self._arm_q_indices]
        hard_upper = self.model.upperPositionLimit[self._arm_q_indices]
        if not np.isfinite(hard_lower).all() or not np.isfinite(hard_upper).all():
            raise ValueError("ELF3 arm joints must have finite URDF limits")
        if np.any(hard_upper - hard_lower <= 2.0 * self.joint_limit_margin):
            raise ValueError("joint_limit_margin leaves an empty arm joint range")
        self._soft_lower = hard_lower + self.joint_limit_margin
        self._soft_upper = hard_upper - self.joint_limit_margin
        self._soft_center = 0.5 * (self._soft_lower + self._soft_upper)
        self._soft_half_range = 0.5 * (self._soft_upper - self._soft_lower)
        self._anchor_frame = self._required_frame("waist_z_link")
        self._shoulder_frames = {
            "left": self._required_frame("l_shoulder_y_link"),
            "right": self._required_frame("r_shoulder_y_link"),
        }
        self._elbow_frames = {
            "left": self._required_frame("l_elbow_y_link"),
            "right": self._required_frame("r_elbow_y_link"),
        }
        self._wrist_frames = {
            "left": self._required_frame("l_wrist_z_link"),
            "right": self._required_frame("r_wrist_z_link"),
        }

    def _compile_configuration_indices(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for joint_id in range(1, self.model.njoints):
            if self.model.nqs[joint_id] != 1 or self.model.nvs[joint_id] != 1:
                continue
            result[str(self.model.names[joint_id])] = int(self.model.idx_qs[joint_id])
        missing = set(ARM_JOINTS) - set(result)
        if missing:
            raise ValueError(f"ELF3 URDF is missing arm joints: {sorted(missing)}")
        return result

    def _velocity_indices(self, names: tuple[str, ...]) -> np.ndarray:
        indices = []
        for name in names:
            joint_id = int(self.model.getJointId(name))
            if joint_id <= 0 or self.model.nvs[joint_id] != 1:
                raise ValueError(f"ELF3 URDF joint is not one-DoF: {name}")
            indices.append(int(self.model.idx_vs[joint_id]))
        return np.asarray(indices, dtype=np.intp)

    def _required_frame(self, name: str) -> int:
        frame_id = int(self.model.getFrameId(name))
        if frame_id >= self.model.nframes:
            raise ValueError(f"ELF3 URDF is missing frame: {name}")
        return frame_id

    @property
    def has_measurement(self) -> bool:
        return self._has_measurement

    @property
    def measured_arm_positions(self) -> np.ndarray:
        return self._measured[self._arm_q_indices].astype(np.float32, copy=True)

    def update_measured(self, names: tuple[str, ...], positions: np.ndarray) -> bool:
        seen_arm_joints: set[str] = set()
        for name, position in zip(names, positions):
            q_index = self._name_to_q.get(name)
            value = float(position)
            if q_index is not None and math.isfinite(value):
                self._measured[q_index] = value
                if name in ARM_JOINTS:
                    seen_arm_joints.add(name)
        np.clip(
            self._measured,
            self.model.lowerPositionLimit,
            self.model.upperPositionLimit,
            out=self._measured,
        )
        complete = len(seen_arm_joints) == len(ARM_JOINTS)
        self._has_measurement = self._has_measurement or complete
        return complete

    @staticmethod
    def _wrapped_angle(value: float) -> float:
        return math.atan2(math.sin(value), math.cos(value))

    def _select_swivel_reference(self, side: str) -> np.ndarray | None:
        shoulder = self.data.oMf[self._shoulder_frames[side]].translation
        wrist = self.data.oMf[self._wrist_frames[side]].translation
        shoulder_to_wrist = wrist - shoulder
        axis_norm = float(np.linalg.norm(shoulder_to_wrist))
        if axis_norm <= 1.0e-8:
            return None
        axis = shoulder_to_wrist / axis_norm
        downward = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
        outward = np.asarray(
            (0.0, 1.0 if side == "left" else -1.0, 0.0),
            dtype=np.float64,
        )
        forward = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        candidates = (downward, outward, forward)
        projections = tuple(
            candidate - axis * float(np.dot(candidate, axis))
            for candidate in candidates
        )
        norms = tuple(float(np.linalg.norm(value)) for value in projections)
        # Prefer gravity-down when it is well-conditioned. The fallback is
        # selected once per solve, so the angular reference cannot switch in
        # the middle of an iterative solve.
        selected = 0 if norms[0] >= 0.25 else int(np.argmax(norms))
        if norms[selected] <= 1.0e-8:
            return None
        return candidates[selected]

    def _swivel_state(
        self,
        side: str,
        reference_axis: np.ndarray | None,
        *,
        with_gradient: bool,
    ) -> tuple[float, np.ndarray | None] | None:
        if reference_axis is None:
            return None
        shoulder = self.data.oMf[self._shoulder_frames[side]].translation
        elbow = self.data.oMf[self._elbow_frames[side]].translation
        wrist = self.data.oMf[self._wrist_frames[side]].translation
        shoulder_to_wrist = wrist - shoulder
        axis_norm = float(np.linalg.norm(shoulder_to_wrist))
        if axis_norm <= 1.0e-8:
            return None
        axis = shoulder_to_wrist / axis_norm
        shoulder_to_elbow = elbow - shoulder
        radial = shoulder_to_elbow - axis * float(np.dot(shoulder_to_elbow, axis))
        radial_norm = float(np.linalg.norm(radial))
        reference = reference_axis - axis * float(np.dot(reference_axis, axis))
        reference_norm = float(np.linalg.norm(reference))
        if radial_norm < self.swivel_min_radius or reference_norm <= 1.0e-8:
            return None
        radial_unit = radial / radial_norm
        reference_unit = reference / reference_norm
        angle = math.atan2(
            float(np.dot(axis, np.cross(reference_unit, radial_unit))),
            float(np.dot(reference_unit, radial_unit)),
        )
        if not with_gradient:
            return angle, None
        elbow_jacobian = pin.computeFrameJacobian(
            self.model,
            self.data,
            self.q,
            self._elbow_frames[side],
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        position_gradient = np.cross(axis, radial) / float(np.dot(radial, radial))
        gradient = position_gradient @ elbow_jacobian[:3, self._arm_v_indices[side]]
        if not np.isfinite(gradient).all():
            return None
        return angle, gradient

    def _solve_side(
        self,
        side: str,
        target: pin.SE3,
        seed_swivel: float | None,
        swivel_reference: np.ndarray | None,
    ) -> int:
        velocity_indices = self._arm_v_indices[side]
        q_indices = self._arm_side_q_indices[side]
        side_offset = 0 if side == "left" else 7
        soft_lower = self._soft_lower[side_offset : side_offset + 7]
        soft_upper = self._soft_upper[side_offset : side_offset + 7]
        soft_center = self._soft_center[side_offset : side_offset + 7]
        soft_half_range = self._soft_half_range[side_offset : side_offset + 7]
        wrist_frame = self._wrist_frames[side]
        for iteration in range(self.iterations):
            pin.forwardKinematics(self.model, self.data, self.q)
            pin.updateFramePlacements(self.model, self.data)
            world_target = self.data.oMf[self._anchor_frame] * target
            current = self.data.oMf[wrist_frame]
            current_to_target = current.actInv(world_target)
            error = pin.log6(current_to_target).vector
            if float(np.linalg.norm(error)) <= self.tolerance:
                return iteration
            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                self.q,
                wrist_frame,
                pin.ReferenceFrame.LOCAL,
            )
            selected = (
                -pin.Jlog6(current_to_target.inverse()) @ jacobian[:, velocity_indices]
            )
            system = selected @ selected.T + self._damping_identity
            step = -selected.T @ np.linalg.solve(system, error)
            if self.joint_centering_gain > 0.0 or (
                self.swivel_continuity_gain > 0.0 and seed_swivel is not None
            ):
                damped_projector = self._arm_identity - selected.T @ np.linalg.solve(
                    system, selected
                )
                secondary = np.zeros(7, dtype=np.float64)
                if self.joint_centering_gain > 0.0:
                    centering = (soft_center - self.q[q_indices]) / soft_half_range
                    secondary += self.joint_centering_gain * centering
                if self.swivel_continuity_gain > 0.0 and seed_swivel is not None:
                    swivel = self._swivel_state(
                        side,
                        swivel_reference,
                        with_gradient=True,
                    )
                    if swivel is not None:
                        current_swivel, swivel_gradient = swivel
                        assert swivel_gradient is not None
                        swivel_error = self._wrapped_angle(seed_swivel - current_swivel)
                        gradient_norm_squared = float(
                            np.dot(swivel_gradient, swivel_gradient)
                        )
                        if gradient_norm_squared > 1.0e-8:
                            secondary += (
                                self.swivel_continuity_gain
                                * swivel_gradient
                                * swivel_error
                                / gradient_norm_squared
                            )
                step += damped_projector @ secondary
            self._velocity.fill(0.0)
            self._velocity[velocity_indices] = step * self.step_size
            self.q[:] = pin.integrate(self.model, self.q, self._velocity)
            np.clip(
                self.q,
                self.model.lowerPositionLimit,
                self.model.upperPositionLimit,
                out=self.q,
            )
            self.q[q_indices] = np.clip(self.q[q_indices], soft_lower, soft_upper)
        return self.iterations

    def _result(
        self,
        *,
        success: bool,
        reason: str,
        position_error: float = math.inf,
        orientation_error: float = math.inf,
        joint_delta: float = math.inf,
        limit_margin: float = -math.inf,
        iterations: int = 0,
        swivel_delta: float = math.inf,
        publishable: bool = False,
    ) -> ArmIkResult:
        positions = (
            self.q[self._arm_q_indices].astype(np.float32, copy=True)
            if success or publishable
            else None
        )
        if positions is None:
            np.copyto(self.q, self._seed)
        return ArmIkResult(
            success=success,
            positions=positions,
            reason=reason,
            position_error_m=float(position_error),
            orientation_error_rad=float(orientation_error),
            maximum_joint_delta_rad=float(joint_delta),
            minimum_limit_margin_rad=float(limit_margin),
            iterations=int(iterations),
            maximum_swivel_delta_rad=float(swivel_delta),
        )

    def solve(self, positions: np.ndarray, orientations: np.ndarray) -> ArmIkResult:
        positions = np.asarray(positions, dtype=np.float64).reshape(2, 3)
        orientations = np.asarray(orientations, dtype=np.float64).reshape(2, 4)
        if not np.isfinite(positions).all() or not np.isfinite(orientations).all():
            raise ValueError("PICO wrist targets contain non-finite values")
        np.copyto(self._seed, self._measured)
        np.copyto(self.q, self._seed)
        if not self._has_measurement:
            return self._result(success=False, reason="missing_measurement")

        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        swivel_references = {
            side: self._select_swivel_reference(side) for side in ("left", "right")
        }
        seed_swivels: dict[str, float | None] = {}
        for side in ("left", "right"):
            swivel = self._swivel_state(
                side,
                swivel_references[side],
                with_gradient=False,
            )
            seed_swivels[side] = None if swivel is None else swivel[0]

        targets = tuple(
            pin.SE3(_rotation_from_wxyz(orientations[index]), positions[index])
            for index in range(2)
        )
        try:
            iterations = sum(
                self._solve_side(
                    side,
                    targets[index],
                    seed_swivels[side],
                    swivel_references[side],
                )
                for side, index in (("left", 0), ("right", 1))
            )
            pin.forwardKinematics(self.model, self.data, self.q)
            pin.updateFramePlacements(self.model, self.data)
        except (FloatingPointError, np.linalg.LinAlgError):
            return self._result(success=False, reason="numerical_failure")

        anchor = self.data.oMf[self._anchor_frame]
        maximum_position_error = 0.0
        maximum_orientation_error = 0.0
        for side, index in (("left", 0), ("right", 1)):
            current = self.data.oMf[self._wrist_frames[side]]
            world_target = anchor * targets[index]
            position_error = float(
                np.linalg.norm(current.translation - world_target.translation)
            )
            orientation_error = float(
                np.linalg.norm(pin.log3(current.rotation.T @ world_target.rotation))
            )
            maximum_position_error = max(maximum_position_error, position_error)
            maximum_orientation_error = max(
                maximum_orientation_error, orientation_error
            )

        arm_q = self.q[self._arm_q_indices]
        seed_q = self._seed[self._arm_q_indices]
        maximum_joint_delta = float(np.max(np.abs(arm_q - seed_q)))
        minimum_limit_margin = float(
            np.min(np.minimum(arm_q - self._soft_lower, self._soft_upper - arm_q))
        )
        maximum_swivel_delta = 0.0
        for side in ("left", "right"):
            seed_swivel = seed_swivels[side]
            if seed_swivel is None:
                continue
            swivel = self._swivel_state(
                side,
                swivel_references[side],
                with_gradient=False,
            )
            if swivel is not None:
                maximum_swivel_delta = max(
                    maximum_swivel_delta,
                    abs(self._wrapped_angle(swivel[0] - seed_swivel)),
                )
        if not (
            np.isfinite(arm_q).all()
            and math.isfinite(maximum_position_error)
            and math.isfinite(maximum_orientation_error)
            and math.isfinite(maximum_joint_delta)
            and math.isfinite(minimum_limit_margin)
            and math.isfinite(maximum_swivel_delta)
        ):
            return self._result(success=False, reason="numerical_failure")
        if minimum_limit_margin < -1.0e-9:
            return self._result(
                success=False,
                reason="soft_limit",
                position_error=maximum_position_error,
                orientation_error=maximum_orientation_error,
                joint_delta=maximum_joint_delta,
                limit_margin=minimum_limit_margin,
                iterations=iterations,
                swivel_delta=maximum_swivel_delta,
            )
        if maximum_position_error > self.maximum_position_error:
            return self._result(
                success=False,
                reason="position_residual",
                position_error=maximum_position_error,
                orientation_error=maximum_orientation_error,
                joint_delta=maximum_joint_delta,
                limit_margin=minimum_limit_margin,
                iterations=iterations,
                swivel_delta=maximum_swivel_delta,
                publishable=True,
            )
        if maximum_orientation_error > self.maximum_orientation_error:
            return self._result(
                success=False,
                reason="orientation_residual",
                position_error=maximum_position_error,
                orientation_error=maximum_orientation_error,
                joint_delta=maximum_joint_delta,
                limit_margin=minimum_limit_margin,
                iterations=iterations,
                swivel_delta=maximum_swivel_delta,
                publishable=True,
            )
        return self._result(
            success=True,
            reason="ok",
            position_error=maximum_position_error,
            orientation_error=maximum_orientation_error,
            joint_delta=maximum_joint_delta,
            limit_margin=minimum_limit_margin,
            iterations=iterations,
            swivel_delta=maximum_swivel_delta,
        )


def _validated_params(raw: dict[str, object]) -> dict[str, object]:
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown arm IK bridge params: {sorted(unknown)}")
    params = {name: raw.get(name, default) for name, default in DEFAULTS.items()}
    for name in (
        "pico_host",
        "pico_topic",
        "robot_state_topic",
        "reference_topic",
    ):
        if not isinstance(params[name], str) or not params[name]:
            raise ValueError(f"{name} must be a non-empty string")
    for name in ("pico_port", "required_consecutive_frames", "ik_iterations"):
        if isinstance(params[name], bool) or not isinstance(params[name], int):
            raise ValueError(f"{name} must be an integer")
    for name in (
        "rate_hz",
        "stale_timeout_s",
        "ik_damping",
        "ik_step_size",
        "ik_tolerance",
        "maximum_position_error_m",
        "maximum_orientation_error_rad",
        "maximum_joint_step_rad",
        "swivel_min_radius_m",
    ):
        value = params[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be a positive number")
    centering_gain = params["joint_centering_gain"]
    if (
        isinstance(centering_gain, bool)
        or not isinstance(centering_gain, (int, float))
        or not math.isfinite(float(centering_gain))
        or float(centering_gain) < 0.0
    ):
        raise ValueError("joint_centering_gain must be a non-negative number")
    limit_margin = params["joint_limit_margin_rad"]
    if (
        isinstance(limit_margin, bool)
        or not isinstance(limit_margin, (int, float))
        or not math.isfinite(float(limit_margin))
        or float(limit_margin) < 0.0
    ):
        raise ValueError("joint_limit_margin_rad must be a non-negative number")
    swivel_gain = params["swivel_continuity_gain"]
    if (
        isinstance(swivel_gain, bool)
        or not isinstance(swivel_gain, (int, float))
        or not math.isfinite(float(swivel_gain))
        or float(swivel_gain) < 0.0
    ):
        raise ValueError("swivel_continuity_gain must be a non-negative number")
    if not 1 <= int(params["pico_port"]) <= 65535:
        raise ValueError("pico_port must be within [1, 65535]")
    return params


class ArmIkBridgeNode(Node):
    """Receive SONIC POSE packets and publish named Pinocchio arm targets."""

    def __init__(self, context: NodeBuildContext) -> None:
        params = _validated_params(dict(context.params))
        super().__init__(context.node_name, namespace=context.namespace or None)
        self._topic = str(params["pico_topic"])
        self._stale_timeout = float(params["stale_timeout_s"])
        self._gate = SourceGate(int(params["required_consecutive_frames"]))
        self._solver = PinocchioArmIk(
            str(context.asset("assets/elf3-dof31.urdf")),
            iterations=int(params["ik_iterations"]),
            damping=float(params["ik_damping"]),
            step_size=float(params["ik_step_size"]),
            tolerance=float(params["ik_tolerance"]),
            maximum_joint_step=float(params["maximum_joint_step_rad"]),
            maximum_position_error=float(params["maximum_position_error_m"]),
            maximum_orientation_error=float(params["maximum_orientation_error_rad"]),
            joint_limit_margin=float(params["joint_limit_margin_rad"]),
            joint_centering_gain=float(params["joint_centering_gain"]),
            swivel_continuity_gain=float(params["swivel_continuity_gain"]),
            swivel_min_radius=float(params["swivel_min_radius_m"]),
        )
        self._maximum_reference_step = self._solver.maximum_joint_step
        self._last_published_arm_positions: np.ndarray | None = None
        qos = QoSProfile(
            depth=1,
            durability=qos_profile_sensor_data.durability,
            reliability=qos_profile_sensor_data.reliability,
        )
        self._state_subscription = self.create_subscription(
            JointState,
            str(params["robot_state_topic"]),
            self._state_callback,
            qos,
        )
        self._command_publisher = self.create_publisher(
            JointState, str(params["reference_topic"]), qos
        )
        self._button_publishers = {
            name: self.create_publisher(Float32, f"pico/{name}", 10)
            for name in BUTTON_FIELDS
        }
        # rclpy.Node owns ``_context`` internally. Keep the ZMQ context under a
        # distinct name or timer creation will treat a zmq.Context as ROS state.
        self._zmq_context = zmq.Context()
        self._subscriber = self._zmq_context.socket(zmq.SUB)
        self._subscriber.setsockopt(zmq.LINGER, 0)
        self._subscriber.setsockopt(zmq.RCVHWM, 16)
        self._subscriber.setsockopt_string(zmq.SUBSCRIBE, self._topic)
        endpoint = f"tcp://{params['pico_host']}:{params['pico_port']}"
        self._subscriber.connect(endpoint)
        self._timer = self.create_timer(1.0 / float(params["rate_hz"]), self._tick)
        self._last_valid_time: float | None = None
        self._stale_warned = False
        self._malformed_warned = False
        self._ik_error_warned = False
        self._ik_rejecting = False
        self._ik_rejection_counts: dict[str, int] = {}
        self._last_ik_summary_time = time.monotonic()
        self._closed = False
        self.get_logger().info(
            f"Pinocchio arm bridge SUB {endpoint} topic='{self._topic}'"
        )

    def _state_callback(self, message: JointState) -> None:
        if len(message.name) != len(message.position) or len(set(message.name)) != len(
            message.name
        ):
            return
        try:
            positions = np.asarray(message.position, dtype=np.float64)
        except (TypeError, ValueError):
            return
        if not np.isfinite(positions).all():
            return
        self._solver.update_measured(tuple(message.name), positions)

    def _publish_buttons(self, fields: dict[str, np.ndarray]) -> None:
        for name, publisher in self._button_publishers.items():
            value = _scalar(fields, name)
            if value is None:
                continue
            message = Float32()
            message.data = float(np.clip(value, 0.0, 1.0))
            publisher.publish(message)

    def _drain_latest(self) -> dict[str, np.ndarray] | None:
        latest = None
        while True:
            try:
                message = self._subscriber.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                return latest
            try:
                decoded = _decode_packed_message(message, self._topic)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                if not self._malformed_warned:
                    self.get_logger().warning(
                        f"ignored malformed PICO POSE packet: {exc}"
                    )
                    self._malformed_warned = True
                continue
            if decoded is not None:
                latest = decoded

    def _rate_limited_arm_reference(self, desired: np.ndarray) -> np.ndarray:
        desired = np.asarray(desired, dtype=np.float32).reshape(14)
        baseline = self._last_published_arm_positions
        if baseline is None:
            baseline = self._solver.measured_arm_positions
        command = baseline + np.clip(
            desired - baseline,
            -self._maximum_reference_step,
            self._maximum_reference_step,
        )
        self._last_published_arm_positions = command
        return command

    def _publish_joint_reference(
        self,
        arm_commands: np.ndarray,
        fields: dict[str, np.ndarray],
    ) -> None:
        names = list(ARM_JOINTS)
        commands = np.asarray(arm_commands, dtype=np.float32).reshape(14).tolist()
        head = fields.get("head_joint_pos")
        if head is not None:
            head_values = np.asarray(head, dtype=np.float32).reshape(-1, 2)[-1]
            if np.isfinite(head_values).all():
                names.extend(HEAD_JOINTS)
                commands.extend(float(value) for value in head_values)
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = names
        message.position = commands
        self._command_publisher.publish(message)

    def _publish_held_reference(self, fields: dict[str, np.ndarray]) -> bool:
        if self._last_published_arm_positions is None:
            return False
        self._publish_joint_reference(self._last_published_arm_positions, fields)
        return True

    def _publish_reference(self, fields: dict[str, np.ndarray]) -> ArmIkResult:
        positions = np.asarray(fields["vr_position"], dtype=np.float64).reshape(3, 3)
        orientations = np.asarray(fields["vr_orientation"], dtype=np.float64).reshape(
            3, 4
        )
        result = self._solver.solve(positions[:2], orientations[:2])
        if result.positions is None:
            self._publish_held_reference(fields)
            return result
        arm_commands = self._rate_limited_arm_reference(result.positions)
        self._publish_joint_reference(arm_commands, fields)
        return result

    def _mark_reference_live(self) -> None:
        self._last_valid_time = time.monotonic()
        self._stale_warned = False
        self._malformed_warned = False

    def _record_ik_rejection(self, result: ArmIkResult) -> None:
        self._ik_rejecting = True
        self._ik_rejection_counts[result.reason] = (
            self._ik_rejection_counts.get(result.reason, 0) + 1
        )
        if not self._ik_error_warned:
            action = (
                "publishing the finite soft-limited best-effort candidate"
                if result.positions is not None
                else "holding the last safe reference"
            )
            self.get_logger().warning(
                "Pinocchio arm IK target quality degraded: "
                f"reason={result.reason}, position_error="
                f"{result.position_error_m:.4f}m, orientation_error="
                f"{result.orientation_error_rad:.4f}rad, max_joint_delta="
                f"{result.maximum_joint_delta_rad:.4f}rad, max_swivel_delta="
                f"{result.maximum_swivel_delta_rad:.4f}rad; {action}"
            )
            self._ik_error_warned = True
        now = time.monotonic()
        if now - self._last_ik_summary_time >= 5.0:
            summary = ", ".join(
                f"{name}={count}"
                for name, count in sorted(self._ik_rejection_counts.items())
            )
            self.get_logger().warning(
                f"Pinocchio arm IK degraded-quality summary: {summary}"
            )
            self._ik_rejection_counts.clear()
            self._last_ik_summary_time = now

    def _record_ik_success(self) -> None:
        if self._ik_rejecting:
            self.get_logger().info(
                "Pinocchio arm IK target quality recovered within residual thresholds"
            )
        self._ik_rejecting = False
        self._ik_error_warned = False

    def _tick(self) -> None:
        if self._closed:
            return
        fields = self._drain_latest()
        if fields is not None:
            self._publish_buttons(fields)
            if self._gate.observe(fields):
                try:
                    result = self._publish_reference(fields)
                    if result.success:
                        self._record_ik_success()
                        self._mark_reference_live()
                    else:
                        self._record_ik_rejection(result)
                        if self._last_published_arm_positions is not None:
                            self._mark_reference_live()
                except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
                    held = self._publish_held_reference({})
                    if held:
                        self._mark_reference_live()
                    if not self._ik_error_warned:
                        self.get_logger().warning(
                            "Pinocchio arm IK rejected malformed PICO target; "
                            f"holding the last safe reference: {exc}"
                        )
                        self._ik_error_warned = True
        if (
            self._last_valid_time is not None
            and time.monotonic() - self._last_valid_time > self._stale_timeout
            and not self._stale_warned
        ):
            self.get_logger().warning(
                "PICO arm reference is stale; state will return both arms to the PD pose"
            )
            self._stale_warned = True

    def destroy_node(self):
        if not self._closed:
            self._closed = True
            self.destroy_timer(self._timer)
            self._subscriber.close(linger=0)
            self._zmq_context.term()
        return super().destroy_node()


def create_node(context: NodeBuildContext) -> ArmIkBridgeNode:
    return ArmIkBridgeNode(context)


def _argument_parser():
    import argparse

    parser = argparse.ArgumentParser()
    for name, default in DEFAULTS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            type=type(default),
            default=default,
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    mod_root = Path(__file__).resolve().parent
    context = NodeBuildContext(
        mod_id="com.bxi.upper_body_teleop",
        node_id="com.bxi.upper_body_teleop/arm_ik_bridge",
        node_name="upper_body_arm_ik_bridge",
        mod_root=mod_root,
        params=vars(args),
    )
    rclpy.init(args=[])
    node = None
    try:
        node = ArmIkBridgeNode(context)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


__all__ = [
    "ARM_JOINTS",
    "ArmIkResult",
    "ArmIkBridgeNode",
    "PinocchioArmIk",
    "SourceGate",
    "_decode_packed_message",
    "create_node",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
