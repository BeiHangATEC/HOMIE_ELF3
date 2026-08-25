"""Read-only ELF3 hardware checks for the HOMIE/PICO control path."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS


HEAD_JOINTS = ("head_z_joint", "head_y_joint")
HARDWARE_JOINTS = tuple(ELF3_POLICY_JOINTS.names) + HEAD_JOINTS
DEFAULT_MOTOR_DISABLE_MASK = 0x60000000

# The vendor driver indexes each bit in get_motor_recv_flag() with this table.
MOTOR_LOCATIONS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    (4, 5),
    (4, 6),
    (4, 7),
    (0, 4),
    (0, 5),
)


def decode_motor_receive_mask(
    received_mask: int,
    *,
    disabled_mask: int = DEFAULT_MOTOR_DISABLE_MASK,
) -> dict:
    """Decode the vendor driver's received-motor bitset by joint and CAN bus."""

    valid_mask = (1 << len(HARDWARE_JOINTS)) - 1
    received_mask = int(received_mask) & valid_mask
    disabled_mask = int(disabled_mask) & valid_mask
    expected_mask = valid_mask & ~disabled_mask
    missing_mask = expected_mask & ~received_mask

    received = []
    missing = []
    disabled = []
    by_bus: dict[str, dict[str, list[str]]] = {}
    for index, (name, location) in enumerate(zip(HARDWARE_JOINTS, MOTOR_LOCATIONS)):
        bus, can_id = location
        item = f"{name} (id={can_id})"
        bus_entry = by_bus.setdefault(
            f"can{bus}", {"received": [], "missing": [], "disabled": []}
        )
        if disabled_mask & (1 << index):
            disabled.append(name)
            bus_entry["disabled"].append(item)
        elif received_mask & (1 << index):
            received.append(name)
            bus_entry["received"].append(item)
        else:
            missing.append(name)
            bus_entry["missing"].append(item)

    return {
        "ready": missing_mask == 0,
        "received_mask": f"0x{received_mask:08X}",
        "disabled_mask": f"0x{disabled_mask:08X}",
        "expected_count": expected_mask.bit_count(),
        "received_count": (received_mask & expected_mask).bit_count(),
        "missing_count": missing_mask.bit_count(),
        "received": received,
        "missing": missing,
        "disabled": disabled,
        "by_bus": by_bus,
    }


@dataclass
class _TopicSample:
    count: int = 0
    first_at: float | None = None
    last_at: float | None = None
    valid: bool = False
    detail: str = "no messages"

    def observe(self, *, now: float, valid: bool, detail: str) -> None:
        self.count += 1
        if self.first_at is None:
            self.first_at = now
        self.last_at = now
        self.valid = valid
        self.detail = detail

    def rate_hz(self) -> float:
        if self.count < 2 or self.first_at is None or self.last_at is None:
            return 0.0
        span = self.last_at - self.first_at
        return (self.count - 1) / span if span > 0.0 else 0.0


@dataclass
class HardwareReadiness:
    """Accumulate observable ROS health without publishing or changing state."""

    required_joints: tuple[str, ...] = tuple(ELF3_POLICY_JOINTS.names)
    actuator_states: _TopicSample = field(default_factory=_TopicSample)
    imu: _TopicSample = field(default_factory=_TopicSample)
    state_machine: _TopicSample = field(default_factory=_TopicSample)
    actuator_commands: _TopicSample = field(default_factory=_TopicSample)
    current_state: str | None = None

    def observe_actuator_states(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        *,
        now: float,
    ) -> None:
        name_tuple = tuple(names)
        missing = sorted(set(self.required_joints) - set(name_tuple))
        valid = (
            bool(name_tuple)
            and len(name_tuple) == len(set(name_tuple))
            and len(name_tuple) == len(positions)
            and all(math.isfinite(float(value)) for value in positions)
            and not missing
        )
        detail = (
            f"{len(name_tuple)} named joints"
            if valid
            else f"invalid feedback; missing={missing}"
        )
        self.actuator_states.observe(now=now, valid=valid, detail=detail)

    def observe_imu(self, quaternion_xyzw: Iterable[float], *, now: float) -> None:
        values = tuple(float(value) for value in quaternion_xyzw)
        norm = math.sqrt(sum(value * value for value in values)) if len(values) == 4 else 0.0
        valid = len(values) == 4 and all(math.isfinite(value) for value in values)
        valid = valid and 0.5 <= norm <= 1.5
        self.imu.observe(
            now=now,
            valid=valid,
            detail=f"quaternion norm={norm:.6f}",
        )

    def observe_state_machine(self, payload: str, *, now: float) -> None:
        try:
            current = json.loads(payload)["current"]["name"]
            valid = isinstance(current, str) and bool(current)
        except (KeyError, TypeError, json.JSONDecodeError):
            current = None
            valid = False
        if valid:
            self.current_state = current
        self.state_machine.observe(
            now=now,
            valid=valid,
            detail=f"current={current}" if valid else "invalid state JSON",
        )

    def observe_actuator_commands(
        self,
        names: Sequence[str],
        position: Sequence[float],
        kp: Sequence[float],
        kd: Sequence[float],
        velocity: Sequence[float],
        torque: Sequence[float],
        *,
        now: float,
    ) -> None:
        name_tuple = tuple(names)
        arrays = (position, kp, kd, velocity, torque)
        missing = sorted(set(self.required_joints) - set(name_tuple))
        valid = (
            bool(name_tuple)
            and len(name_tuple) == len(set(name_tuple))
            and all(len(values) == len(name_tuple) for values in arrays)
            and all(
                math.isfinite(float(value))
                for values in arrays
                for value in values
            )
            and not missing
        )
        detail = (
            f"{len(name_tuple)} named joints"
            if valid
            else f"invalid command; missing={missing}"
        )
        self.actuator_commands.observe(now=now, valid=valid, detail=detail)

    def report(self, *, now: float, freshness_s: float = 0.5) -> dict:
        minimum_rates = {
            "actuator_states": 20.0,
            "imu": 20.0,
            "state_machine": 2.0,
            "actuator_commands": 20.0,
        }
        topics = {}
        errors = []
        for name, minimum_rate in minimum_rates.items():
            sample = getattr(self, name)
            age = None if sample.last_at is None else now - sample.last_at
            rate = sample.rate_hz()
            ready = (
                sample.valid
                and age is not None
                and 0.0 <= age <= freshness_s
                and rate >= minimum_rate
            )
            topics[name] = {
                "ready": ready,
                "count": sample.count,
                "rate_hz": round(rate, 3),
                "age_s": None if age is None else round(age, 3),
                "detail": sample.detail,
            }
            if not ready:
                errors.append(
                    f"{name} not ready: {sample.detail}, rate={rate:.1f} Hz, age={age}"
                )
        return {
            "ready": not errors,
            "current_state": self.current_state,
            "topics": topics,
            "errors": errors,
        }


def _find_pci_device(vendor: str, device: str) -> list[str]:
    matches = []
    for entry in Path("/sys/bus/pci/devices").glob("*"):
        try:
            entry_vendor = (entry / "vendor").read_text(encoding="ascii").strip().lower()
            entry_device = (entry / "device").read_text(encoding="ascii").strip().lower()
        except OSError:
            continue
        if entry_vendor == vendor.lower() and entry_device == device.lower():
            matches.append(entry.name)
    return sorted(matches)


def static_host_checks(*, runtime_root: str, arm_ik_python: str) -> dict:
    """Check files needed by the already-installed vendor hardware stack."""

    runtime = Path(runtime_root)
    sonic_runtime = runtime.parent / "com.bxi.sonic"
    checks = {
        "dev_mem": Path("/dev/mem").exists() and os.access("/dev/mem", os.R_OK),
        "imu": Path("/dev/ttyIMU").exists(),
        "gamepad": Path("/dev/input/jsBattleDragon").exists(),
        "hardware_driver": Path(
            "/opt/bxi/bxi_ros2_pkg/lib/hardware_elf3/hardware_elf3"
        ).is_file(),
        "upper_body_runtime": runtime.is_dir(),
        "pico_runtime_launcher": (
            runtime / "shared_runtime_launcher.py"
        ).is_file(),
        "arm_bridge_launcher": (runtime / "arm_bridge_launcher.py").is_file(),
        "sonic_runtime": (sonic_runtime / "mod.yaml").is_file(),
        "arm_ik_python": Path(arm_ik_python).is_file()
        and os.access(arm_ik_python, os.X_OK),
    }
    pci_devices = _find_pci_device("0x10ee", "0x7022")
    checks["pci_canfd_10ee_7022"] = bool(pci_devices)
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "pci_devices": pci_devices,
        "runtime_root": str(runtime),
        "arm_ik_python": arm_ik_python,
    }


def _parse_mask(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer mask: {value}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("mask must be non-negative")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only ELF3 HOMIE/PICO hardware readiness check"
    )
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--topic-prefix", default="hardware/")
    parser.add_argument("--motor-recv-mask", type=_parse_mask)
    parser.add_argument(
        "--motor-disable-mask",
        type=_parse_mask,
        default=DEFAULT_MOTOR_DISABLE_MASK,
    )
    parser.add_argument(
        "--runtime-root",
        default=os.environ.get(
            "BXI_UPPER_BODY_MOD_ROOT",
            "/opt/bxi/homie_pico_runtime/com.bxi.upper_body_teleop",
        ),
    )
    parser.add_argument(
        "--arm-ik-python",
        default=os.environ.get("BXI_ARM_IK_PYTHON", sys.executable),
    )
    parser.add_argument("--static-only", action="store_true")
    return parser


def _monitor_ros(*, duration: float, topic_prefix: str) -> dict:
    import rclpy
    from communication.msg import ActuatorCmds, ActuatorStates
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import Imu
    from std_msgs.msg import String

    if not math.isfinite(duration) or duration < 1.0:
        raise ValueError("duration must be finite and at least 1 second")
    prefix = topic_prefix.strip().strip("/")
    prefix = f"/{prefix}/" if prefix else "/"
    readiness = HardwareReadiness()
    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    node = Node("homie_pico_hw_preflight")
    qos = QoSProfile(
        depth=10,
        durability=qos_profile_sensor_data.durability,
        reliability=qos_profile_sensor_data.reliability,
    )

    def now() -> float:
        return time.monotonic()

    node.create_subscription(
        ActuatorStates,
        f"{prefix}actuator_states",
        lambda msg: readiness.observe_actuator_states(
            msg.name, msg.position, now=now()
        ),
        qos,
    )
    node.create_subscription(
        Imu,
        f"{prefix}imu_data",
        lambda msg: readiness.observe_imu(
            (
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ),
            now=now(),
        ),
        qos,
    )
    node.create_subscription(
        String,
        f"{prefix}state_machine_info",
        lambda msg: readiness.observe_state_machine(msg.data, now=now()),
        10,
    )
    node.create_subscription(
        ActuatorCmds,
        f"{prefix}actuators_cmds",
        lambda msg: readiness.observe_actuator_commands(
            msg.actuators_name,
            msg.pos,
            msg.kp,
            msg.kd,
            msg.vel,
            msg.torque,
            now=now(),
        ),
        qos,
    )

    deadline = time.monotonic() + duration
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        return readiness.report(now=time.monotonic())
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    args = _build_parser().parse_args()
    result = {
        "static": static_host_checks(
            runtime_root=args.runtime_root,
            arm_ik_python=args.arm_ik_python,
        )
    }
    if args.motor_recv_mask is not None:
        result["motor_receive"] = decode_motor_receive_mask(
            args.motor_recv_mask,
            disabled_mask=args.motor_disable_mask,
        )
    if not args.static_only:
        result["ros"] = _monitor_ros(
            duration=args.duration,
            topic_prefix=args.topic_prefix,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    ready = result["static"]["ready"] and (
        "ros" not in result or result["ros"]["ready"]
    ) and (
        "motor_receive" not in result or result["motor_receive"]["ready"]
    )
    raise SystemExit(0 if ready else 1)


__all__ = [
    "DEFAULT_MOTOR_DISABLE_MASK",
    "HARDWARE_JOINTS",
    "HardwareReadiness",
    "decode_motor_receive_mask",
    "main",
    "static_host_checks",
]
