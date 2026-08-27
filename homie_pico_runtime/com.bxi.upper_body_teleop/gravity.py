"""Allocation-stable ELF3 arm gravity compensation.

The kinematic and inertial constants below are taken from
``resources/elf3_dof29/urdf/elf3.urdf``.  Only the two seven-joint arm chains
are retained because this Mod never commands gravity feed-forward on the
waist, legs, or head.
"""

from __future__ import annotations

import math

import numpy as np


ARM_JOINT_SUFFIXES = (
    "shoulder_y_joint",
    "shoulder_x_joint",
    "shoulder_z_joint",
    "elbow_y_joint",
    "wrist_x_joint",
    "wrist_y_joint",
    "wrist_z_joint",
)

LEFT_ARM_JOINTS = tuple(f"l_{suffix}" for suffix in ARM_JOINT_SUFFIXES)
RIGHT_ARM_JOINTS = tuple(f"r_{suffix}" for suffix in ARM_JOINT_SUFFIXES)
ARM_JOINTS = (*LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS)

# URDF effort limits, kept in the same named order as each arm chain.
ARM_EFFORT_LIMITS = np.asarray(
    (45.0, 45.0, 21.0, 45.0, 21.0, 21.0, 21.0),
    dtype=np.float32,
)
ARM_EFFORT_LIMITS.flags.writeable = False

_JOINT_ORIGINS_LEFT = (
    (0.0, 0.178, 0.087),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, -0.256),
    (0.256, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)
_JOINT_ORIGINS_RIGHT = (
    (0.0, -0.178, 0.087),
    *_JOINT_ORIGINS_LEFT[1:],
)
_JOINT_AXES = (
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_AXIS_INDICES = (1, 0, 2, 1, 0, 1, 2)
_LINK_MASSES_LEFT = (
    0.99747353,
    0.13939414,
    1.67349896,
    0.10206150,
    1.24945755,
    0.56828784,
    0.09977741,
)
_LINK_MASSES_RIGHT = (
    0.99747344,
    0.13939414,
    1.67349843,
    0.10206150,
    1.24945755,
    0.56828784,
    0.09977741,
)
_LINK_COMS_LEFT = (
    (-0.02329380, -0.00424861, 0.00174581),
    (0.01044987, 0.0, -0.02650420),
    (-0.00140240, -0.00501388, -0.18798077),
    (0.02791974, 0.01785187, 0.0),
    (-0.12572120, -0.00125474, 0.0),
    (0.00034025, 0.00000493, -0.00154873),
    (0.03004397, 0.0, 0.01513885),
)
_LINK_COMS_RIGHT = (
    (-0.02329380, 0.00338182, 0.00024448),
    (0.01044987, 0.0, -0.02650420),
    (-0.00153940, 0.00501389, -0.18708592),
    (0.02791974, -0.01785187, 0.0),
    (-0.12520740, 0.00074632, 0.0),
    (0.00034019, -0.00112277, -0.00154873),
    (0.03004397, 0.0, 0.01513885),
)


class _ArmChain:
    _DOF = 7

    def __init__(self, origins, masses, centers_of_mass) -> None:
        self._origins = np.asarray(origins, dtype=np.float64)
        self._axes = np.asarray(_JOINT_AXES, dtype=np.float64)
        self._masses = np.asarray(masses, dtype=np.float64)
        self._centers_of_mass = np.asarray(centers_of_mass, dtype=np.float64)

        self._joint_positions = np.empty((self._DOF, 3), dtype=np.float64)
        self._joint_axes = np.empty((self._DOF, 3), dtype=np.float64)
        self._link_com_positions = np.empty((self._DOF, 3), dtype=np.float64)
        self._parent_rotation = np.eye(3, dtype=np.float64)
        self._link_rotation = np.empty((3, 3), dtype=np.float64)
        self._joint_rotation = np.empty((3, 3), dtype=np.float64)
        self._parent_position = np.zeros(3, dtype=np.float64)
        self._identity_rotation = np.eye(3, dtype=np.float64)
        self._vector_buffer = np.empty(3, dtype=np.float64)

    @staticmethod
    def _axis_rotation(axis_index: int, angle: float, output: np.ndarray) -> None:
        cosine = math.cos(angle)
        sine = math.sin(angle)
        output.fill(0.0)
        if axis_index == 0:
            output[0, 0] = 1.0
            output[1, 1] = cosine
            output[1, 2] = -sine
            output[2, 1] = sine
            output[2, 2] = cosine
        elif axis_index == 1:
            output[0, 0] = cosine
            output[0, 2] = sine
            output[1, 1] = 1.0
            output[2, 0] = -sine
            output[2, 2] = cosine
        else:
            output[0, 0] = cosine
            output[0, 1] = -sine
            output[1, 0] = sine
            output[1, 1] = cosine
            output[2, 2] = 1.0

    def _forward_kinematics(self, positions: np.ndarray) -> None:
        self._parent_rotation[:] = self._identity_rotation
        self._parent_position.fill(0.0)

        for index in range(self._DOF):
            joint_position = self._joint_positions[index]
            np.matmul(
                self._parent_rotation,
                self._origins[index],
                out=self._vector_buffer,
            )
            np.add(
                self._parent_position,
                self._vector_buffer,
                out=joint_position,
            )
            np.matmul(
                self._parent_rotation,
                self._axes[index],
                out=self._joint_axes[index],
            )

            self._axis_rotation(
                _AXIS_INDICES[index],
                float(positions[index]),
                self._joint_rotation,
            )
            np.matmul(
                self._parent_rotation,
                self._joint_rotation,
                out=self._link_rotation,
            )
            link_com = self._link_com_positions[index]
            np.matmul(
                self._link_rotation,
                self._centers_of_mass[index],
                out=self._vector_buffer,
            )
            np.add(joint_position, self._vector_buffer, out=link_com)

            self._parent_position[:] = joint_position
            self._parent_rotation[:] = self._link_rotation

    def compute(
        self,
        positions: np.ndarray,
        gravity: np.ndarray,
        output: np.ndarray,
    ) -> None:
        self._forward_kinematics(positions)
        gx, gy, gz = (float(value) for value in gravity)

        for joint_index in range(self._DOF):
            px, py, pz = self._joint_positions[joint_index]
            ax, ay, az = self._joint_axes[joint_index]
            gravity_moment_x = 0.0
            gravity_moment_y = 0.0
            gravity_moment_z = 0.0
            for link_index in range(joint_index, self._DOF):
                rx = self._link_com_positions[link_index, 0] - px
                ry = self._link_com_positions[link_index, 1] - py
                rz = self._link_com_positions[link_index, 2] - pz
                mass = self._masses[link_index]
                gravity_moment_x += mass * (ry * gz - rz * gy)
                gravity_moment_y += mass * (rz * gx - rx * gz)
                gravity_moment_z += mass * (rx * gy - ry * gx)

            # Negate gravity's generalized moment to obtain the actuator
            # feed-forward effort required to hold the current configuration.
            output[joint_index] = -(
                ax * gravity_moment_x
                + ay * gravity_moment_y
                + az * gravity_moment_z
            )

    def potential_energy(self, positions: np.ndarray, gravity: np.ndarray) -> float:
        """Return gravitational potential energy; primarily useful in tests."""
        self._forward_kinematics(positions)
        return -float(
            np.sum(
                self._masses[:, None]
                * self._link_com_positions
                * gravity[None, :]
            )
        )


class ArmGravityModel:
    """Compute feed-forward gravity effort for both ELF3 arm chains."""

    def __init__(self) -> None:
        self._left = _ArmChain(
            _JOINT_ORIGINS_LEFT,
            _LINK_MASSES_LEFT,
            _LINK_COMS_LEFT,
        )
        self._right = _ArmChain(
            _JOINT_ORIGINS_RIGHT,
            _LINK_MASSES_RIGHT,
            _LINK_COMS_RIGHT,
        )

    def compute(
        self,
        arm_positions: np.ndarray,
        gravity_in_torso: np.ndarray,
        output: np.ndarray,
    ) -> np.ndarray:
        if arm_positions.shape != (14,):
            raise ValueError(
                f"arm positions must have shape (14,), got {arm_positions.shape}"
            )
        if gravity_in_torso.shape != (3,):
            raise ValueError(
                "gravity vector must have shape (3,), got "
                f"{gravity_in_torso.shape}"
            )
        if output.shape != (14,):
            raise ValueError(f"gravity output must have shape (14,), got {output.shape}")
        if any(not math.isfinite(float(value)) for value in arm_positions):
            raise ValueError("arm positions contain non-finite values")
        if any(not math.isfinite(float(value)) for value in gravity_in_torso):
            raise ValueError("gravity vector contains non-finite values")

        self._left.compute(arm_positions[:7], gravity_in_torso, output[:7])
        self._right.compute(arm_positions[7:], gravity_in_torso, output[7:])
        return output


__all__ = [
    "ARM_EFFORT_LIMITS",
    "ARM_JOINTS",
    "ArmGravityModel",
    "LEFT_ARM_JOINTS",
    "RIGHT_ARM_JOINTS",
]
