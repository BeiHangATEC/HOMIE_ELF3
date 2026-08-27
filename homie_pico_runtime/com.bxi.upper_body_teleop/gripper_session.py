"""State-lifecycle wrapper for the gripper behavior shared with SONIC."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from threading import Lock
from typing import TYPE_CHECKING

import communication.msg as bxi_msg
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32

from .gripper import (
    BxiMotor,
    CalibrationPhase,
    CalibrationSettings,
    GripperCalibrator,
    JointControl,
    MotorFeedback,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import LoggerLike, RobotControlContext


@dataclass(frozen=True, slots=True)
class GripperConfig:
    enabled: bool = False
    enable_interval_s: float = 1.0
    left_bus: int = 5
    right_bus: int = 6
    can_id: int = 1
    master_id: int = 0x11
    kp: float = 20.0
    kd: float = 1.0
    calibration_speed_rad_s: float = 0.2
    calibration_kp: float = 5.0
    calibration_kd: float = 0.5
    contact_torque: float = 2.0
    abort_torque: float = 8.0
    contact_confirm_s: float = 0.25
    stopped_velocity_rad_s: float = 0.1
    tracking_error_rad: float = 0.08
    limit_margin_rad: float = 0.15
    minimum_span_rad: float = 1.0
    maximum_search_travel_rad: float = 7.0
    response_timeout_s: float = 1.0
    feedback_timeout_s: float = 0.3
    phase_timeout_s: float = 45.0
    maximum_mos_temperature_c: int = 80
    maximum_motor_temperature_c: int = 80


class GripperSession:
    """The same enable, calibration, feedback and trigger flow as SONIC."""

    def __init__(self, config: GripperConfig) -> None:
        self.config = config
        self.enabled = bool(config.enabled)
        self._validate_config()
        settings = CalibrationSettings(
            speed_rad_s=float(config.calibration_speed_rad_s),
            contact_torque=float(config.contact_torque),
            abort_torque=float(config.abort_torque),
            contact_confirm_s=float(config.contact_confirm_s),
            stopped_velocity_rad_s=float(config.stopped_velocity_rad_s),
            tracking_error_rad=float(config.tracking_error_rad),
            limit_margin_rad=float(config.limit_margin_rad),
            minimum_span_rad=float(config.minimum_span_rad),
            maximum_search_travel_rad=float(config.maximum_search_travel_rad),
            response_timeout_s=float(config.response_timeout_s),
            feedback_timeout_s=float(config.feedback_timeout_s),
            phase_timeout_s=float(config.phase_timeout_s),
            maximum_mos_temperature_c=int(config.maximum_mos_temperature_c),
            maximum_motor_temperature_c=int(config.maximum_motor_temperature_c),
        )
        self._calibrators = {
            int(config.left_bus): GripperCalibrator("left", settings),
            int(config.right_bus): GripperCalibrator("right", settings),
        }
        self._logger: LoggerLike | None = None
        self._active = False
        self._armed = False
        self._calibrated = False
        self._faulted = False
        self._available = not self.enabled
        self._last_enable_time: float | None = None
        self._left_trigger = 0.0
        self._right_trigger = 0.0
        self._subscriptions = []
        self._publisher = None
        self._feedback_lock = Lock()
        self._feedback: dict[int, MotorFeedback] = {}
        self._phase_snapshot = {
            bus: calibrator.phase for bus, calibrator in self._calibrators.items()
        }
        self._bad_feedback_warned = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def logger(self) -> LoggerLike:
        if self._logger is None:
            raise RuntimeError("gripper logger is not bound")
        return self._logger

    def _validate_config(self) -> None:
        config = self.config
        if config.enable_interval_s <= 0.0:
            raise ValueError("gripper enable_interval_s must be positive")
        ids = (config.left_bus, config.right_bus, config.can_id, config.master_id)
        if min(ids) < 0:
            raise ValueError("gripper bus and CAN IDs must be non-negative")
        if config.left_bus == config.right_bus:
            raise ValueError("left and right gripper buses must differ")
        if max(config.left_bus, config.right_bus) > 0xFF:
            raise ValueError("gripper bus must fit in uint8")
        if max(config.can_id, config.master_id) > 0x7FF:
            raise ValueError("gripper CAN IDs must be standard 11-bit IDs")
        gains = (config.kp, config.kd, config.calibration_kp, config.calibration_kd)
        if not all(math.isfinite(value) and value >= 0.0 for value in gains):
            raise ValueError("gripper gains must be finite and non-negative")

    def bind(self, ctx: RobotControlContext, logger: LoggerLike) -> None:
        self._logger = logger
        if not self.enabled:
            return
        packet_type = getattr(
            bxi_msg,
            "CANFDPacket",
            getattr(bxi_msg, "CanfdPacket", None),
        )
        if packet_type is None:
            self.enabled = False
            self._available = True
            self.logger.warning(
                "半身遥操夹爪已禁用：缺少communication.msg.CANFDPacket；双臂遥操仍可用"
            )
            return
        self._subscriptions = [
            ctx.ros_node.create_subscription(
                Float32,
                "pico/left_trigger",
                self._left_trigger_callback,
                QoSProfile(depth=1),
            ),
            ctx.ros_node.create_subscription(
                Float32,
                "pico/right_trigger",
                self._right_trigger_callback,
                QoSProfile(depth=1),
            ),
            ctx.ros_node.create_subscription(
                packet_type,
                "canfd_packet/rx",
                self._feedback_callback,
                QoSProfile(depth=100),
            ),
        ]
        self._publisher = ctx.ros_node.create_publisher(
            packet_type,
            "canfd_packet/tx",
            QoSProfile(depth=100),
        )
        self._available = True

    def unbind(self, ctx: RobotControlContext) -> None:
        if self._active and self._publisher is not None:
            self._disable()
        for subscription in self._subscriptions:
            ctx.ros_node.destroy_subscription(subscription)
        self._subscriptions.clear()
        with self._feedback_lock:
            self._feedback.clear()
        if self._publisher is not None:
            ctx.ros_node.destroy_publisher(self._publisher)
            self._publisher = None

    def enter(self) -> None:
        if not self.enabled or self._publisher is None:
            return
        now = time.monotonic()
        self._start(now)
        self._publish_enable(now)
        self._armed = True
        self.logger.info(
            "半身遥操夹爪已使能，等待电机响应后执行与SONIC一致的低速限位校准"
        )

    def exit(self) -> None:
        if self.enabled and self._publisher is not None:
            self._disable()
        self._active = False
        self._armed = False
        self._calibrated = False
        self._faulted = False
        self._last_enable_time = None
        with self._feedback_lock:
            self._feedback.clear()

    @staticmethod
    def _valid_trigger(value: object) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result):
            return None
        return min(1.0, max(0.0, result))

    def _left_trigger_callback(self, message: Float32) -> None:
        value = self._valid_trigger(message.data)
        if value is not None and self._active:
            self._left_trigger = value

    def _right_trigger_callback(self, message: Float32) -> None:
        value = self._valid_trigger(message.data)
        if value is not None and self._active:
            self._right_trigger = value

    def _feedback_callback(self, message) -> None:
        if not self._active:
            return
        bus = int(message.bus)
        if bus not in self._calibrators:
            return
        if int(message.frame.can_id) & 0x7FF != int(self.config.master_id):
            return
        try:
            if int(message.frame.len) != 8:
                raise ValueError(f"expected 8 feedback bytes, got {int(message.frame.len)}")
            feedback = BxiMotor.unpack_feedback(
                message.frame.data,
                received_at=time.monotonic(),
            )
            if feedback.motor_id != int(self.config.can_id):
                return
        except (TypeError, ValueError) as exc:
            if not self._bad_feedback_warned:
                self.logger.warning(f"忽略非法夹爪响应帧：{exc}")
                self._bad_feedback_warned = True
            return
        with self._feedback_lock:
            self._feedback[bus] = feedback

    def _start(self, now: float) -> None:
        self._left_trigger = self._right_trigger = 0.0
        self._armed = False
        self._calibrated = False
        self._faulted = False
        self._bad_feedback_warned = False
        self._last_enable_time = None
        with self._feedback_lock:
            self._feedback.clear()
        for bus, calibrator in self._calibrators.items():
            calibrator.reset(now)
            self._phase_snapshot[bus] = calibrator.phase
        self._active = True

    def _publish_enable(self, now: float) -> None:
        assert self._publisher is not None
        for bus in (self.config.left_bus, self.config.right_bus):
            self._publisher.publish(
                BxiMotor.build_motor_packet(
                    int(bus), int(self.config.can_id), BxiMotor.enter_motor_mode()
                )
            )
        self._last_enable_time = now

    def _refresh_enable(self, now: float) -> None:
        if (
            self._last_enable_time is None
            or now - self._last_enable_time >= self.config.enable_interval_s
        ):
            self._publish_enable(now)

    def _disable(self) -> None:
        if self._publisher is None:
            return
        for bus in (self.config.left_bus, self.config.right_bus):
            self._publisher.publish(
                BxiMotor.build_motor_packet(
                    int(bus), int(self.config.can_id), BxiMotor.exit_motor_mode()
                )
            )

    def _publish_target(self, bus: int, target: float, *, kp: float, kd: float) -> None:
        assert self._publisher is not None
        data = BxiMotor.pack_cmd(
            JointControl(p_des=float(target), kp=float(kp), kd=float(kd)),
            p_range=(-12.5, 12.5),
            v_range=(-45.0, 45.0),
            t_range=(-40.0, 40.0),
            kp_range=(0.0, 500.0),
            kd_range=(0.0, 5.0),
        )
        self._publisher.publish(
            BxiMotor.build_motor_packet(bus, int(self.config.can_id), data)
        )

    def _publish_trigger_target(self, bus: int, trigger: float) -> None:
        calibrator = self._calibrators[bus]
        if not calibrator.ready:
            return
        assert calibrator.open_position is not None
        assert calibrator.closed_position is not None
        target = calibrator.closed_position + (1.0 - trigger) * (
            calibrator.open_position - calibrator.closed_position
        )
        self._publish_target(bus, target, kp=self.config.kp, kd=self.config.kd)

    def _log_phase(self, bus: int, calibrator: GripperCalibrator) -> None:
        previous = self._phase_snapshot[bus]
        if calibrator.phase is previous:
            return
        self._phase_snapshot[bus] = calibrator.phase
        side = "左" if bus == self.config.left_bus else "右"
        labels = {
            CalibrationPhase.SETTLING: "收到响应，正在稳定当前位置",
            CalibrationPhase.SEEKING_OPEN: "开始低速寻找张开限位",
            CalibrationPhase.BACKING_OFF_OPEN: "已检测张开限位，正在回退",
            CalibrationPhase.SEEKING_CLOSED: "开始低速寻找闭合限位",
            CalibrationPhase.BACKING_OFF_CLOSED: "已检测闭合限位，正在回退",
            CalibrationPhase.RETURNING_OPEN: "正在低速返回张开位置",
            CalibrationPhase.READY: "校准完成",
        }
        if calibrator.phase in labels:
            self.logger.info(f"半身遥操{side}夹爪：{labels[calibrator.phase]}")

    def _fail(self, reason: str) -> None:
        if self._faulted:
            return
        self._faulted = True
        self._calibrated = False
        self._disable()
        self.logger.error(f"半身遥操夹爪校准失败：{reason}；左右夹爪已退出电机模式")

    def update(self, dt: float) -> None:
        if not self.enabled or not self._active or self._faulted:
            return
        now = time.monotonic()
        if not self._armed:
            self._publish_enable(now)
            self._armed = True
        with self._feedback_lock:
            feedback = dict(self._feedback)

        waiting = tuple(
            bus
            for bus, calibrator in self._calibrators.items()
            if calibrator.phase is CalibrationPhase.WAITING_FEEDBACK
        )
        if waiting and not all(bus in feedback for bus in waiting):
            for bus in waiting:
                calibrator = self._calibrators[bus]
                if bus not in feedback:
                    calibrator.update(None, now, dt)
                if calibrator.failed:
                    self._fail(calibrator.failure_reason or "unknown calibration error")
                    return
            return

        for bus, calibrator in self._calibrators.items():
            target = calibrator.update(feedback.get(bus), now, dt)
            self._log_phase(bus, calibrator)
            if calibrator.failed:
                self._fail(calibrator.failure_reason or "unknown calibration error")
                return
            if target is not None and not calibrator.ready:
                self._publish_target(
                    bus,
                    target,
                    kp=self.config.calibration_kp,
                    kd=self.config.calibration_kd,
                )
        if not all(calibrator.ready for calibrator in self._calibrators.values()):
            return
        if not self._calibrated:
            self._calibrated = True
            details = []
            for bus, calibrator in self._calibrators.items():
                side = "左" if bus == self.config.left_bus else "右"
                details.append(
                    f"{side}[闭={calibrator.closed_position:.3f}, 开={calibrator.open_position:.3f}]"
                )
            self.logger.info(
                "半身遥操夹爪校准完成，PICO trigger开始接管：" + ", ".join(details)
            )
        self._refresh_enable(now)
        self._publish_trigger_target(int(self.config.left_bus), self._left_trigger)
        self._publish_trigger_target(int(self.config.right_bus), self._right_trigger)


__all__ = ["GripperConfig", "GripperSession"]
