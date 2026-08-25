"""Adapters for the official upper-body runtime used by HOMIE + PICO."""

from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .mixer import ARM_JOINTS


FloatArray = NDArray[np.float32]


def _runtime_package(root: str | Path) -> tuple[str, Path]:
    runtime_root = Path(root).expanduser().resolve()
    if not runtime_root.is_dir():
        raise RuntimeError(f"official upper-body runtime not found: {runtime_root}")
    digest = hashlib.sha256(str(runtime_root).encode("utf-8")).hexdigest()[:12]
    package_name = f"_bxi_upper_body_runtime_{digest}"
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(runtime_root)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    return package_name, runtime_root


def _runtime_module(root: str | Path, leaf: str):
    package_name, runtime_root = _runtime_package(root)
    module_path = runtime_root / f"{leaf}.py"
    if not module_path.is_file():
        raise RuntimeError(f"official upper-body runtime is missing: {module_path}")
    return importlib.import_module(f"{package_name}.{leaf}")


def load_official_gravity(root: str | Path):
    """Return the official gravity model and validated per-arm effort limits."""
    module = _runtime_module(root, "gravity")
    runtime_joints = tuple(getattr(module, "ARM_JOINTS", ()))
    if runtime_joints != ARM_JOINTS:
        raise RuntimeError(
            "official gravity model joint layout does not match this ELF3 build"
        )
    model_type = getattr(module, "ArmGravityModel", None)
    limits = np.asarray(getattr(module, "ARM_EFFORT_LIMITS", ()), dtype=np.float32)
    if model_type is None or limits.shape != (7,) or not np.isfinite(limits).all():
        raise RuntimeError("official gravity model API is incomplete")
    return model_type(), limits.copy()


def load_official_gripper_types(root: str | Path):
    """Return the official gripper config/session types without loading its Mod."""
    module = _runtime_module(root, "gripper_session")
    config_type = getattr(module, "GripperConfig", None)
    session_type = getattr(module, "GripperSession", None)
    if config_type is None or session_type is None:
        raise RuntimeError("official gripper runtime API is incomplete")
    return config_type, session_type


class ArmGravityCompensator:
    """Compute bounded official arm gravity effort from ROS feedback samples."""

    _STANDARD_GRAVITY = 9.80665

    def __init__(
        self,
        model,
        effort_limits: Sequence[float],
        *,
        gravity_scale: float = 1.0,
        torque_limit_scale: float = 0.8,
        sample_timeout_s: float = 0.5,
    ) -> None:
        limits = np.asarray(effort_limits, dtype=np.float32)
        if limits.shape == (7,):
            limits = np.tile(limits, 2)
        if limits.shape != (14,) or not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError("arm effort limits must contain 7 or 14 positive values")
        for name, value, lower, upper in (
            ("gravity_scale", gravity_scale, 0.0, 1.5),
            ("torque_limit_scale", torque_limit_scale, 0.0, 1.0),
        ):
            numeric = float(value)
            if not math.isfinite(numeric) or not lower <= numeric <= upper:
                raise ValueError(f"{name} must be within [{lower}, {upper}]")
        timeout = float(sample_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("sample_timeout_s must be finite and positive")

        self._model = model
        self.gravity_scale = float(gravity_scale)
        self.sample_timeout_s = timeout
        self._limits = limits * float(torque_limit_scale)
        self._negative_limits = -self._limits
        self._positions = np.zeros(14, dtype=np.float64)
        self._quaternion_wxyz = np.zeros(4, dtype=np.float64)
        self._gravity = np.zeros(3, dtype=np.float64)
        self._torques = np.zeros(14, dtype=np.float64)
        self._joint_received_at: float | None = None
        self._imu_received_at: float | None = None

    @staticmethod
    def _timestamp(value: float) -> float:
        timestamp = float(value)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        return timestamp

    def observe_joint_state(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        *,
        received_at: float,
    ) -> None:
        if len(names) != len(positions) or len(set(names)) != len(names):
            raise ValueError("joint feedback names and positions are inconsistent")
        by_name = dict(zip(names, positions))
        try:
            arm_positions = np.asarray(
                [by_name[name] for name in ARM_JOINTS], dtype=np.float64
            )
        except KeyError as exc:
            raise ValueError(f"joint feedback is missing {exc.args[0]}") from exc
        if not np.isfinite(arm_positions).all():
            raise ValueError("arm feedback contains non-finite positions")
        self._positions[:] = arm_positions
        self._joint_received_at = self._timestamp(received_at)

    def observe_orientation_xyzw(
        self, values: Sequence[float], *, received_at: float
    ) -> None:
        quaternion = np.asarray(values, dtype=np.float64)
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise ValueError("IMU orientation must contain four finite values")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1.0e-6:
            raise ValueError("IMU orientation quaternion has zero norm")
        x, y, z, w = quaternion / norm
        self._quaternion_wxyz[:] = w, x, y, z
        self._imu_received_at = self._timestamp(received_at)

    @staticmethod
    def _fresh(received_at: float | None, now: float, timeout_s: float) -> bool:
        return received_at is not None and 0.0 <= now - received_at <= timeout_s

    def compute(self, *, now: float) -> FloatArray | None:
        timestamp = self._timestamp(now)
        if not self._fresh(
            self._joint_received_at, timestamp, self.sample_timeout_s
        ) or not self._fresh(self._imu_received_at, timestamp, self.sample_timeout_s):
            return None

        w, x, y, z = self._quaternion_wxyz
        gravity = -self._STANDARD_GRAVITY * self.gravity_scale
        self._gravity[0] = gravity * (2.0 * (x * z - w * y))
        self._gravity[1] = gravity * (2.0 * (y * z + w * x))
        self._gravity[2] = gravity * (1.0 - 2.0 * (x * x + y * y))
        self._model.compute(self._positions, self._gravity, self._torques)
        if not np.isfinite(self._torques).all():
            raise ValueError("official gravity model returned non-finite effort")
        np.clip(
            self._torques,
            self._negative_limits,
            self._limits,
            out=self._torques,
        )
        return self._torques.astype(np.float32, copy=True)


__all__ = [
    "ArmGravityCompensator",
    "load_official_gravity",
    "load_official_gripper_types",
]
