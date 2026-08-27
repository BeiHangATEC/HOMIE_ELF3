"""Walking lower body with PICO/Pinocchio upper-body teleoperation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from bxi_example_py_elf3.framework.mod_api import (
    EntryFrameProvider,
    JointCommandComposer,
    JointCommandLayer,
    JointLayout,
    JointTargetBuffer,
    MotorFrame,
    ResourceHandle,
    RobotControlState,
    RunningFrameProvider,
    StateBehavior,
)
from bxi_example_py_elf3.policies import HumanoidGaitPolicyLiteIsaaclab
from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS

from .gravity import ARM_EFFORT_LIMITS, ARM_JOINTS, ArmGravityModel
from .gripper_session import GripperSession

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext


LEFT_ARM_JOINTS = ARM_JOINTS[:7]
RIGHT_ARM_JOINTS = ARM_JOINTS[7:]
ARM_LAYOUT = JointLayout(ARM_JOINTS, label="PICO dual-arm target")
HEAD_JOINTS = ("head_y_joint", "head_z_joint")
HEAD_LAYOUT = JointLayout(HEAD_JOINTS, label="PICO head target")
UPPER_BODY_OUTPUT = JointLayout(
    (*ELF3_POLICY_JOINTS.names, *HEAD_JOINTS),
    label="upper-body teleop state output",
)


@dataclass(frozen=True, slots=True)
class UpperBodyTeleopParams:
    reference_topic: str = "pico_control_joint_commands"
    robot_state_topic: str = "pico_control_joint_states"
    live_reference_timeout_s: float = 0.5
    grip_threshold: float = 0.5
    arm_gain_ramp_s: float = 0.4
    arm_kp_scale: float = 1.0
    gravity_scale: float = 1.0
    torque_limit_scale: float = 0.8
    friction_coulomb: tuple[float, ...] = (0.0,) * 7
    friction_viscous: tuple[float, ...] = (0.0,) * 7
    friction_smoothing_velocity: float = 0.1
    head_control_enabled: bool = False
    head_pitch_limit_rad: float = 0.5
    head_yaw_limit_rad: float = 1.0
    head_pitch_speed_rad_s: float = 1.5
    head_yaw_speed_rad_s: float = 2.0
    head_deadband_rad: float = 0.015

    def __post_init__(self) -> None:
        if not self.reference_topic or not self.robot_state_topic:
            raise ValueError("PICO reference and robot-state topics must be non-empty")
        positive = (
            self.live_reference_timeout_s,
            self.arm_gain_ramp_s,
            self.friction_smoothing_velocity,
            self.head_pitch_limit_rad,
            self.head_yaw_limit_rad,
            self.head_pitch_speed_rad_s,
            self.head_yaw_speed_rad_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("upper-body teleop positive parameters are invalid")
        if not 0.0 <= self.grip_threshold <= 1.0:
            raise ValueError("grip_threshold must be within [0, 1]")
        if not 0.0 <= self.arm_kp_scale <= 2.0:
            raise ValueError("arm_kp_scale must be within [0, 2]")
        if not 0.0 <= self.gravity_scale <= 1.5:
            raise ValueError("gravity_scale must be within [0, 1.5]")
        if not 0.0 < self.torque_limit_scale <= 1.0:
            raise ValueError("torque_limit_scale must be within (0, 1]")
        if not math.isfinite(self.head_deadband_rad) or self.head_deadband_rad < 0.0:
            raise ValueError("head_deadband_rad must be finite and non-negative")
        for name, values in (
            ("friction_coulomb", self.friction_coulomb),
            ("friction_viscous", self.friction_viscous),
        ):
            if len(values) != 7 or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in values
            ):
                raise ValueError(f"{name} must contain seven finite non-negative values")


class UpperBodyTeleopState(
    RobotControlState,
    EntryFrameProvider,
    RunningFrameProvider,
):
    """Half-body teleop: gait policy below, named PICO arms above."""

    _STANDARD_GRAVITY = 9.80665
    _HEAD_KP = 16.747
    _HEAD_KD = 1.066

    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[HumanoidGaitPolicyLiteIsaaclab],
        gravity_model: ResourceHandle[ArmGravityModel],
        params: UpperBodyTeleopParams,
        gripper: GripperSession,
    ) -> None:
        super().__init__(name, state_id, resources=(policy, gravity_model))
        self._policy = policy
        self._gravity_model = gravity_model
        self._params = params
        self._gripper = gripper

        self._reference_lock = Lock()
        self._reference_arm = np.zeros(14, dtype=np.float32)
        self._reference_head = np.zeros(2, dtype=np.float32)
        self._reference_received_at: float | None = None
        self._left_grip = 0.0
        self._right_grip = 0.0
        self._reference_snapshot = np.zeros(14, dtype=np.float32)
        self._head_snapshot = np.zeros(2, dtype=np.float32)
        self._grip_snapshot = np.zeros(2, dtype=np.float32)

        self._arm_target = JointTargetBuffer(ARM_LAYOUT)
        self._head_target = JointTargetBuffer(HEAD_LAYOUT)
        self._head_target.kp.fill(self._HEAD_KP)
        self._head_target.kd.fill(self._HEAD_KD)
        self._zero_head = np.zeros(2, dtype=np.float32)
        self._composer: JointCommandComposer | None = None
        self._policy_arm_indices: np.ndarray | None = None
        self._output_arm_indices: np.ndarray | None = None
        self._robot_arm_indices: np.ndarray | None = None
        self._policy_layout = None
        self._robot_layout = None
        self._tracking_blend = np.zeros(2, dtype=np.float32)
        self._arm_positions = np.empty(14, dtype=np.float64)
        self._arm_torques = np.empty(14, dtype=np.float64)
        self._gravity = np.empty(3, dtype=np.float64)
        self._friction_coulomb = np.tile(
            np.asarray(params.friction_coulomb, dtype=np.float64), 2
        )
        self._friction_viscous = np.tile(
            np.asarray(params.friction_viscous, dtype=np.float64), 2
        )
        self._friction_enabled = bool(
            np.any(self._friction_coulomb) or np.any(self._friction_viscous)
        )
        self._torque_limits = np.tile(ARM_EFFORT_LIMITS, 2)
        self._torque_limits *= params.torque_limit_scale
        self._negative_torque_limits = -self._torque_limits
        self._subscriptions = []
        self._robot_state_publisher = None
        self._robot_state_message: JointState | None = None
        self._last_robot_state_publish = 0.0
        self._last_running_frame: MotorFrame | None = None
        self._reference_was_fresh = False
        self._invalid_reference_warned = False

    @property
    def policy(self) -> HumanoidGaitPolicyLiteIsaaclab:
        return self._policy.get()

    @property
    def gravity_model(self) -> ArmGravityModel:
        return self._gravity_model.get()

    def on_bind(self, ctx: RobotControlContext) -> None:
        qos = QoSProfile(
            depth=1,
            durability=qos_profile_sensor_data.durability,
            reliability=qos_profile_sensor_data.reliability,
        )
        self._subscriptions = [
            ctx.ros_node.create_subscription(
                JointState,
                self._params.reference_topic,
                self._reference_callback,
                qos,
            ),
            ctx.ros_node.create_subscription(
                Float32,
                "pico/left_grip",
                self._left_grip_callback,
                qos,
            ),
            ctx.ros_node.create_subscription(
                Float32,
                "pico/right_grip",
                self._right_grip_callback,
                qos,
            ),
        ]
        self._robot_state_publisher = ctx.ros_node.create_publisher(
            JointState, self._params.robot_state_topic, qos
        )
        self._gripper.bind(ctx, self.logger)

    def on_unbind(self, ctx: RobotControlContext) -> None:
        self._gripper.unbind(ctx)
        for subscription in self._subscriptions:
            ctx.ros_node.destroy_subscription(subscription)
        self._subscriptions.clear()
        if self._robot_state_publisher is not None:
            ctx.ros_node.destroy_publisher(self._robot_state_publisher)
            self._robot_state_publisher = None

    def _reference_callback(self, message: JointState) -> None:
        if (
            len(message.name) != len(message.position)
            or len(set(message.name)) != len(message.name)
        ):
            return
        by_name = dict(zip(message.name, message.position))
        try:
            arm = np.asarray([by_name[name] for name in ARM_JOINTS], dtype=np.float32)
            head = np.asarray(
                [by_name.get(name, 0.0) for name in HEAD_JOINTS], dtype=np.float32
            )
        except (KeyError, TypeError, ValueError):
            if not self._invalid_reference_warned:
                self.logger.warning("忽略缺少具名双臂关节的PICO解算结果")
                self._invalid_reference_warned = True
            return
        if not np.isfinite(arm).all() or not np.isfinite(head).all():
            if not self._invalid_reference_warned:
                self.logger.warning("忽略包含非有限值的PICO解算结果")
                self._invalid_reference_warned = True
            return
        with self._reference_lock:
            np.copyto(self._reference_arm, arm)
            np.copyto(self._reference_head, head)
            self._reference_received_at = time.monotonic()
        self._invalid_reference_warned = False

    @staticmethod
    def _valid_button(value: object) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result):
            return None
        return float(np.clip(result, 0.0, 1.0))

    def _left_grip_callback(self, message: Float32) -> None:
        value = self._valid_button(message.data)
        if value is not None:
            with self._reference_lock:
                self._left_grip = value

    def _right_grip_callback(self, message: Float32) -> None:
        value = self._valid_button(message.data)
        if value is not None:
            with self._reference_lock:
                self._right_grip = value

    def is_available(self, ctx: RobotControlContext) -> bool:
        return self._gripper.available and all(
            name in ctx.robot_layout.names for name in ARM_JOINTS
        )

    def on_prepare(
        self,
        ctx: RobotControlContext,
        from_state: StateBehavior[RobotControlContext],
    ) -> None:
        del from_state
        ctx.preheat_model(self.policy, command=self.get_cmd_vel(ctx))
        self._compile_mappings(self.policy.output.joints.layout, ctx.robot_layout)
        self._prepare_composer()
        self._tracking_blend.fill(0.0)
        self._head_target.position.fill(0.0)
        self._last_running_frame = None
        self._reference_was_fresh = False
        self._prepare_robot_state_message(ctx)

    def _compile_mappings(self, policy_layout, robot_layout) -> None:
        if self._policy_layout is not policy_layout:
            self._policy_arm_indices = np.fromiter(
                (policy_layout.index(name) for name in ARM_JOINTS),
                dtype=np.intp,
                count=14,
            )
            self._policy_layout = policy_layout
        if self._robot_layout is not robot_layout:
            self._robot_arm_indices = np.fromiter(
                (robot_layout.index(name) for name in ARM_JOINTS),
                dtype=np.intp,
                count=14,
            )
            self._robot_layout = robot_layout

    def _prepare_composer(self) -> None:
        output_layout = (
            UPPER_BODY_OUTPUT
            if self._params.head_control_enabled
            else ELF3_POLICY_JOINTS
        )
        layers = [
            JointCommandLayer("without_arm_policy", self.policy.output.joints),
            JointCommandLayer("pico_arms", self._arm_target.view, override=True),
        ]
        if self._params.head_control_enabled:
            layers.append(JointCommandLayer("pico_head", self._head_target.view))
        self._composer = JointCommandComposer(output_layout, tuple(layers))
        self._output_arm_indices = np.fromiter(
            (output_layout.index(name) for name in ARM_JOINTS),
            dtype=np.intp,
            count=14,
        )

    def _prepare_robot_state_message(self, ctx: RobotControlContext) -> None:
        message = JointState()
        message.name = list(ctx.robot_layout.names)
        message.position = [0.0] * ctx.robot_layout.dof_num
        message.velocity = [0.0] * ctx.robot_layout.dof_num
        self._robot_state_message = message
        self._last_robot_state_publish = 0.0

    def on_enter(self, ctx: RobotControlContext) -> None:
        del ctx
        self._gripper.enter()
        self.logger.info(
            "PICO半身遥操已启动：下半身步态持续运行，双臂默认保持PD站立姿态；"
            "PICO按ABXY校准后自动进入POSE，握紧对应grip后接管该侧手臂"
        )

    def on_exit(self, ctx: RobotControlContext) -> None:
        del ctx
        self._gripper.exit()
        self._tracking_blend.fill(0.0)

    def _read_reference(self) -> tuple[bool, bool, bool]:
        with self._reference_lock:
            np.copyto(self._reference_snapshot, self._reference_arm)
            np.copyto(self._head_snapshot, self._reference_head)
            self._grip_snapshot[0] = self._left_grip
            self._grip_snapshot[1] = self._right_grip
            received_at = self._reference_received_at
        fresh = (
            received_at is not None
            and time.monotonic() - received_at <= self._params.live_reference_timeout_s
        )
        return (
            fresh,
            fresh and self._grip_snapshot[0] > self._params.grip_threshold,
            fresh and self._grip_snapshot[1] > self._params.grip_threshold,
        )

    def _project_gravity(self, quaternion_wxyz: np.ndarray) -> None:
        w, x, y, z = (float(value) for value in quaternion_wxyz)
        norm_squared = w * w + x * x + y * y + z * z
        if not math.isfinite(norm_squared) or norm_squared < 1.0e-12:
            raise ValueError("robot orientation quaternion is invalid")
        inverse_norm = 1.0 / math.sqrt(norm_squared)
        w, x, y, z = (
            w * inverse_norm,
            x * inverse_norm,
            y * inverse_norm,
            z * inverse_norm,
        )
        gravity = -self._STANDARD_GRAVITY * self._params.gravity_scale
        self._gravity[0] = gravity * (2.0 * (x * z - w * y))
        self._gravity[1] = gravity * (2.0 * (y * z + w * x))
        self._gravity[2] = gravity * (1.0 - 2.0 * (x * x + y * y))

    def _compute_gravity(self, ctx: RobotControlContext) -> None:
        assert self._robot_arm_indices is not None
        for arm_index, robot_index in enumerate(self._robot_arm_indices):
            self._arm_positions[arm_index] = ctx.robot_joints.position[robot_index]
        self._project_gravity(ctx.current_quat_wxyz)
        self.gravity_model.compute(
            self._arm_positions,
            self._gravity,
            self._arm_torques,
        )
        if self._friction_enabled:
            for arm_index, robot_index in enumerate(self._robot_arm_indices):
                velocity = float(ctx.robot_joints.velocity[robot_index])
                self._arm_torques[arm_index] += (
                    self._friction_coulomb[arm_index]
                    * math.tanh(velocity / self._params.friction_smoothing_velocity)
                    + self._friction_viscous[arm_index] * velocity
                )
        np.clip(
            self._arm_torques,
            self._negative_torque_limits,
            self._torque_limits,
            out=self._arm_torques,
        )

    def _advance_tracking_blend(self, left_active: bool, right_active: bool, dt: float) -> None:
        maximum_step = max(0.0, float(dt)) / self._params.arm_gain_ramp_s
        for index, active in enumerate((left_active, right_active)):
            target = 1.0 if active else 0.0
            delta = float(np.clip(target - self._tracking_blend[index], -maximum_step, maximum_step))
            self._tracking_blend[index] += delta

    def _update_head(self, fresh: bool, dt: float) -> None:
        if not self._params.head_control_enabled:
            return
        desired = self._head_snapshot if fresh else self._zero_head
        clipped = np.clip(
            desired,
            (-self._params.head_pitch_limit_rad, -self._params.head_yaw_limit_rad),
            (self._params.head_pitch_limit_rad, self._params.head_yaw_limit_rad),
        )
        clipped[np.abs(clipped) < self._params.head_deadband_rad] = 0.0
        maximum_step = np.asarray(
            (
                self._params.head_pitch_speed_rad_s * dt,
                self._params.head_yaw_speed_rad_s * dt,
            ),
            dtype=np.float32,
        )
        self._head_target.position += np.clip(
            clipped - self._head_target.position,
            -maximum_step,
            maximum_step,
        )

    def _update_arm_target(self) -> None:
        assert self._policy_arm_indices is not None
        policy_target = self.policy.output.joints
        for side in range(2):
            blend = float(self._tracking_blend[side])
            start = side * 7
            end = start + 7
            for arm_index in range(start, end):
                local_index = arm_index
                policy_index = int(self._policy_arm_indices[arm_index])
                pd_position = float(policy_target.position[policy_index])
                pico_position = float(self._reference_snapshot[arm_index])
                self._arm_target.position[local_index] = pd_position + blend * (
                    pico_position - pd_position
                )
                self._arm_target.kp[local_index] = float(
                    policy_target.kp[policy_index]
                ) * self._params.arm_kp_scale
                self._arm_target.kd[local_index] = float(
                    policy_target.kd[policy_index]
                )

    def _build_frame(self, ctx: RobotControlContext, dt: float, *, advance: bool) -> MotorFrame:
        if self._composer is None or self._output_arm_indices is None:
            raise RuntimeError("upper-body command composer is not prepared")
        fresh, left_active, right_active = self._read_reference()
        self._compute_gravity(ctx)
        if advance:
            self._advance_tracking_blend(left_active, right_active, dt)
            self._update_head(fresh, dt)
            if fresh != self._reference_was_fresh:
                self.logger.info(
                    "PICO双臂参考已恢复"
                    if fresh
                    else "PICO双臂参考断流，已回到PD站立姿态"
                )
                self._reference_was_fresh = fresh
        self._update_arm_target()
        frame = self._composer.compose()
        frame.vel.fill(0.0)
        frame.torque.fill(0.0)
        frame.torque[self._output_arm_indices] = self._arm_torques
        return frame

    def _publish_robot_state(self, ctx: RobotControlContext) -> None:
        if self._robot_state_publisher is None or self._robot_state_message is None:
            return
        now = time.monotonic()
        if now - self._last_robot_state_publish < 0.02:
            return
        self._last_robot_state_publish = now
        message = self._robot_state_message
        message.header.stamp = ctx.ros_node.get_clock().now().to_msg()
        for index in range(ctx.robot_layout.dof_num):
            message.position[index] = float(ctx.robot_joints.position[index])
            message.velocity[index] = float(ctx.robot_joints.velocity[index])
        self._robot_state_publisher.publish(message)

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self._build_frame(ctx, 0.0, advance=False)

    def sample_running_frame(
        self,
        ctx: RobotControlContext,
        dt: float,
        *,
        advance: bool,
    ) -> MotorFrame:
        if not advance:
            return self._last_running_frame or self.get_entry_frame(ctx)
        self.get_cmd_vel(ctx)
        self.policy.step(ctx.inference_frame, dt, advance=True)
        frame = self._build_frame(ctx, dt, advance=True)
        self._publish_robot_state(ctx)
        self._last_running_frame = frame
        return frame

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state(
                "com.bxi.basic_actions/zero_torque",
                trigger="upper_body_teleop_safety",
            )
            return
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))
        self._gripper.update(dt)


__all__ = [
    "ARM_LAYOUT",
    "HEAD_LAYOUT",
    "UpperBodyTeleopParams",
    "UpperBodyTeleopState",
]
