"""ROS bridge that applies official PICO IK targets over HOMIE's arms."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import rclpy
from communication.msg import ActuatorCmds, ActuatorStates
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32, String

from bxi_example_py_elf3.policies.homie import HOMIE_PARAMETERS
from bxi_example_py_elf3.policies.joints import (
    ELF3_ISAAC_PARAMETERS,
    ELF3_POLICY_JOINTS,
)

from .mixer import ARM_JOINTS, HEAD_JOINTS, HomiePicoArmMixer
from .runtime_support import (
    ArmGravityCompensator,
    load_official_gravity,
    load_official_gripper_types,
)


HEAD_KP = (16.747, 16.747)
HEAD_KD = (1.066, 1.066)


class _GripperContext:
    """Small context surface required by the official GripperSession."""

    def __init__(self, ros_node: Node) -> None:
        self.ros_node = ros_node


def _absolute_topic(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("ROS topic must be non-empty")
    return text if text.startswith("/") else f"/{text}"


def _prefix_topic(prefix: str, suffix: str) -> str:
    normalized = str(prefix).strip().strip("/")
    return _absolute_topic(f"{normalized}/{suffix}" if normalized else suffix)


def _teleop_arm_gains(kp_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Select official upper-body gains by name across differing layouts."""

    indices = np.asarray(
        [ELF3_ISAAC_PARAMETERS.layout.index(name) for name in ARM_JOINTS],
        dtype=np.intp,
    )
    return (
        ELF3_ISAAC_PARAMETERS.kp[indices] * float(kp_scale),
        ELF3_ISAAC_PARAMETERS.kd[indices].copy(),
    )


class HomiePicoArmOverrideNode(Node):
    """Publish named arm overrides only while the controller is in HOMIE."""

    def __init__(self) -> None:
        super().__init__("homie_pico_arm_override")
        self.declare_parameter("topic_prefix", "simulation/")
        self.declare_parameter("state_machine_info_topic", "")
        self.declare_parameter("target_state", "com.bxi.homie/homie")
        self.declare_parameter("reference_topic", "pico_control_joint_commands")
        self.declare_parameter("robot_state_topic", "pico_control_joint_states")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("state_timeout_s", 0.5)
        self.declare_parameter("reference_timeout_s", 0.5)
        self.declare_parameter("grip_timeout_s", 0.5)
        self.declare_parameter("grip_threshold", 0.5)
        self.declare_parameter("arm_gain_ramp_s", 1.0)
        self.declare_parameter("arm_kp_scale", 1.0)
        self.declare_parameter("head_control_enabled", True)
        self.declare_parameter("head_pitch_limit_rad", 0.5)
        self.declare_parameter("head_yaw_limit_rad", 1.0)
        self.declare_parameter("head_pitch_speed_rad_s", 1.5)
        self.declare_parameter("head_yaw_speed_rad_s", 2.0)
        self.declare_parameter("head_deadband_rad", 0.015)
        self.declare_parameter("upper_body_mod_root", "")
        self.declare_parameter("gravity_compensation_enabled", True)
        self.declare_parameter("gravity_scale", 1.0)
        self.declare_parameter("torque_limit_scale", 0.8)
        self.declare_parameter("gravity_input_timeout_s", 0.5)
        self.declare_parameter("hardware_gripper", False)
        self.declare_parameter("gripper_left_bus", 5)
        self.declare_parameter("gripper_right_bus", 6)
        self.declare_parameter("gripper_can_id", 1)
        self.declare_parameter("gripper_master_id", 17)
        self.declare_parameter("gripper_kp", 10.0)
        self.declare_parameter("gripper_kd", 0.5)
        self.declare_parameter("gripper_calibration_speed_rad_s", 2.0)
        self.declare_parameter("gripper_calibration_kp", 5.0)
        self.declare_parameter("gripper_calibration_kd", 0.5)
        self.declare_parameter("gripper_contact_torque", 0.5)
        self.declare_parameter("gripper_abort_torque", 1.0)
        self.declare_parameter("gripper_stopped_velocity_rad_s", 0.5)
        self.declare_parameter("gripper_phase_timeout_s", 50.0)
        self.declare_parameter("gripper_maximum_mos_temperature_c", 120)
        self.declare_parameter("gripper_maximum_motor_temperature_c", 120)

        prefix = str(self.get_parameter("topic_prefix").value)
        state_topic = str(self.get_parameter("state_machine_info_topic").value)
        if not state_topic:
            state_topic = _prefix_topic(prefix, "state_machine_info")
        else:
            state_topic = _absolute_topic(state_topic)
        reference_topic = _absolute_topic(
            str(self.get_parameter("reference_topic").value)
        )
        robot_state_topic = _absolute_topic(
            str(self.get_parameter("robot_state_topic").value)
        )
        actuator_state_topic = _prefix_topic(prefix, "actuator_states")
        imu_topic = _prefix_topic(prefix, "imu_data")
        override_topic = _prefix_topic(prefix, "actuators_cmds_override")

        rate_hz = float(self.get_parameter("rate_hz").value)
        arm_kp_scale = float(self.get_parameter("arm_kp_scale").value)
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("rate_hz must be finite and positive")
        if not math.isfinite(arm_kp_scale) or arm_kp_scale < 0.0:
            raise ValueError("arm_kp_scale must be finite and non-negative")

        arm_indices = np.asarray(
            [ELF3_POLICY_JOINTS.index(name) for name in ARM_JOINTS],
            dtype=np.intp,
        )
        arm_kp, arm_kd = _teleop_arm_gains(arm_kp_scale)
        self._mixer = HomiePicoArmMixer(
            target_state=str(self.get_parameter("target_state").value),
            nominal_arm_position=HOMIE_PARAMETERS.default_position[arm_indices],
            # HOMIE's arm gains hold a fixed training pose and are too stiff
            # for a moving operator reference. Reuse the official ELF3
            # upper-body policy gains for teleoperation instead.
            arm_kp=arm_kp,
            arm_kd=arm_kd,
            state_timeout_s=float(self.get_parameter("state_timeout_s").value),
            reference_timeout_s=float(
                self.get_parameter("reference_timeout_s").value
            ),
            grip_timeout_s=float(self.get_parameter("grip_timeout_s").value),
            grip_threshold=float(self.get_parameter("grip_threshold").value),
            arm_gain_ramp_s=float(self.get_parameter("arm_gain_ramp_s").value),
            head_control_enabled=bool(
                self.get_parameter("head_control_enabled").value
            ),
            head_pitch_limit_rad=float(
                self.get_parameter("head_pitch_limit_rad").value
            ),
            head_yaw_limit_rad=float(
                self.get_parameter("head_yaw_limit_rad").value
            ),
            head_pitch_speed_rad_s=float(
                self.get_parameter("head_pitch_speed_rad_s").value
            ),
            head_yaw_speed_rad_s=float(
                self.get_parameter("head_yaw_speed_rad_s").value
            ),
            head_deadband_rad=float(
                self.get_parameter("head_deadband_rad").value
            ),
        )

        runtime_root = str(self.get_parameter("upper_body_mod_root").value).strip()
        self._gravity = None
        if bool(self.get_parameter("gravity_compensation_enabled").value):
            model, effort_limits = load_official_gravity(runtime_root)
            self._gravity = ArmGravityCompensator(
                model,
                effort_limits,
                gravity_scale=float(self.get_parameter("gravity_scale").value),
                torque_limit_scale=float(
                    self.get_parameter("torque_limit_scale").value
                ),
                sample_timeout_s=float(
                    self.get_parameter("gravity_input_timeout_s").value
                ),
            )

        self._gripper = None
        self._gripper_context = None
        self._gripper_active = False
        if bool(self.get_parameter("hardware_gripper").value):
            if prefix.strip("/") != "hardware":
                raise ValueError(
                    "hardware_gripper may only be enabled with topic_prefix=hardware/"
                )
            config_type, session_type = load_official_gripper_types(runtime_root)
            config = config_type(
                enabled=True,
                left_bus=int(self.get_parameter("gripper_left_bus").value),
                right_bus=int(self.get_parameter("gripper_right_bus").value),
                can_id=int(self.get_parameter("gripper_can_id").value),
                master_id=int(self.get_parameter("gripper_master_id").value),
                kp=float(self.get_parameter("gripper_kp").value),
                kd=float(self.get_parameter("gripper_kd").value),
                calibration_speed_rad_s=float(
                    self.get_parameter("gripper_calibration_speed_rad_s").value
                ),
                calibration_kp=float(
                    self.get_parameter("gripper_calibration_kp").value
                ),
                calibration_kd=float(
                    self.get_parameter("gripper_calibration_kd").value
                ),
                contact_torque=float(
                    self.get_parameter("gripper_contact_torque").value
                ),
                abort_torque=float(
                    self.get_parameter("gripper_abort_torque").value
                ),
                stopped_velocity_rad_s=float(
                    self.get_parameter("gripper_stopped_velocity_rad_s").value
                ),
                phase_timeout_s=float(
                    self.get_parameter("gripper_phase_timeout_s").value
                ),
                maximum_mos_temperature_c=int(
                    self.get_parameter("gripper_maximum_mos_temperature_c").value
                ),
                maximum_motor_temperature_c=int(
                    self.get_parameter("gripper_maximum_motor_temperature_c").value
                ),
            )
            self._gripper = session_type(config)
            self._gripper_context = _GripperContext(self)
            self._gripper.bind(self._gripper_context, self.get_logger())

        sensor_qos = QoSProfile(
            depth=1,
            durability=qos_profile_sensor_data.durability,
            reliability=qos_profile_sensor_data.reliability,
        )
        self._override_publisher = self.create_publisher(
            ActuatorCmds, override_topic, sensor_qos
        )
        self._robot_state_publisher = self.create_publisher(
            JointState, robot_state_topic, sensor_qos
        )
        # Keep these references without shadowing rclpy.Node._subscriptions.
        self._input_subscriptions = (
            self.create_subscription(
                ActuatorStates,
                actuator_state_topic,
                self._actuator_state_callback,
                sensor_qos,
            ),
            self.create_subscription(
                Imu,
                imu_topic,
                self._imu_callback,
                sensor_qos,
            ),
            self.create_subscription(
                JointState,
                reference_topic,
                self._reference_callback,
                sensor_qos,
            ),
            self.create_subscription(
                Float32, "/pico/left_grip", self._left_grip_callback, 10
            ),
            self.create_subscription(
                Float32, "/pico/right_grip", self._right_grip_callback, 10
            ),
            self.create_subscription(
                String, state_topic, self._state_callback, 10
            ),
        )
        self._head_supported = False
        self._override_active = False
        self._invalid_reference_warned = False
        self._invalid_state_warned = False
        self._gravity_stale_warned = False
        self._state_name = ""
        self._state_received_at: float | None = None
        self._target_state = str(self.get_parameter("target_state").value)
        self._state_timeout_s = float(self.get_parameter("state_timeout_s").value)
        self._last_tick = time.monotonic()
        self._timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            "HOMIE PICO override ready: "
            f"state={state_topic}, feedback={actuator_state_topic}, "
            f"imu={imu_topic}, reference={reference_topic}, output={override_topic}, "
            f"gravity={'on' if self._gravity is not None else 'off'}, "
            f"gripper={'on' if self._gripper is not None else 'off'}"
        )

    def _actuator_state_callback(self, message: ActuatorStates) -> None:
        names = tuple(message.name)
        if (
            len(names) != len(message.position)
            or len(set(names)) != len(names)
            or not names
        ):
            return
        try:
            position = np.asarray(message.position, dtype=np.float64)
        except (TypeError, ValueError):
            return
        if not np.isfinite(position).all():
            return
        if self._gravity is not None:
            try:
                self._gravity.observe_joint_state(
                    names, position, received_at=time.monotonic()
                )
            except ValueError:
                pass
        velocity = np.asarray(message.velocity, dtype=np.float64)
        if velocity.shape != position.shape or not np.isfinite(velocity).all():
            velocity = np.zeros_like(position)
        output = JointState()
        output.header = message.header
        output.name = list(names)
        output.position = position.tolist()
        output.velocity = velocity.tolist()
        self._robot_state_publisher.publish(output)
        self._head_supported = all(name in names for name in HEAD_JOINTS)

    def _imu_callback(self, message: Imu) -> None:
        if self._gravity is None:
            return
        orientation = message.orientation
        try:
            self._gravity.observe_orientation_xyzw(
                (orientation.x, orientation.y, orientation.z, orientation.w),
                received_at=time.monotonic(),
            )
        except ValueError:
            return

    def _reference_callback(self, message: JointState) -> None:
        try:
            self._mixer.observe_reference(
                message.name,
                message.position,
                received_at=time.monotonic(),
            )
        except ValueError as exc:
            if not self._invalid_reference_warned:
                self.get_logger().warning(f"ignored invalid PICO IK reference: {exc}")
                self._invalid_reference_warned = True
            return
        self._invalid_reference_warned = False

    def _left_grip_callback(self, message: Float32) -> None:
        try:
            self._mixer.observe_grip(
                "left", message.data, received_at=time.monotonic()
            )
        except ValueError:
            return

    def _right_grip_callback(self, message: Float32) -> None:
        try:
            self._mixer.observe_grip(
                "right", message.data, received_at=time.monotonic()
            )
        except ValueError:
            return

    def _state_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            current = payload["current"]
            state_name = current["name"]
            if not isinstance(state_name, str) or not state_name:
                raise ValueError("current state name is missing")
            self._mixer.observe_state(
                state_name,
                received_at=time.monotonic(),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if not self._invalid_state_warned:
                self.get_logger().warning(
                    f"ignored invalid state_machine_info payload: {exc}"
                )
                self._invalid_state_warned = True
            return
        now = time.monotonic()
        self._state_name = state_name
        self._state_received_at = now
        self._set_gripper_active(state_name == self._target_state)
        self._invalid_state_warned = False

    def _set_gripper_active(self, active: bool) -> None:
        if self._gripper is None or active == self._gripper_active:
            return
        if active:
            self._gripper.enter()
        else:
            self._gripper.exit()
        self._gripper_active = active

    def _publish_release(self) -> None:
        message = ActuatorCmds()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "elf3"
        self._override_publisher.publish(message)

    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(max(0.0, now - self._last_tick), 0.1)
        self._last_tick = now
        command = self._mixer.step(now=now, dt=dt)
        state_live = (
            self._state_name == self._target_state
            and self._state_received_at is not None
            and 0.0 <= now - self._state_received_at <= self._state_timeout_s
        )
        self._set_gripper_active(state_live)
        if self._gripper_active:
            self._gripper.update(dt)
        if command is None:
            if self._override_active:
                self._publish_release()
                self._override_active = False
                self.get_logger().info("HOMIE PICO upper-body override released")
            return

        arm_torque = np.zeros(len(ARM_JOINTS), dtype=np.float32)
        if self._gravity is not None:
            try:
                gravity_torque = self._gravity.compute(now=now)
            except ValueError as exc:
                gravity_torque = None
                if not self._gravity_stale_warned:
                    self.get_logger().error(f"invalid arm gravity compensation: {exc}")
            if gravity_torque is None:
                if not self._gravity_stale_warned:
                    self.get_logger().warning(
                        "HOMIE PICO override waiting for fresh arm and IMU feedback"
                    )
                    self._gravity_stale_warned = True
                if self._override_active:
                    self._publish_release()
                    self._override_active = False
                return
            arm_torque[:] = gravity_torque
            self._gravity_stale_warned = False

        names = list(ARM_JOINTS)
        position = command.arm_position.tolist()
        kp = command.arm_kp.tolist()
        kd = command.arm_kd.tolist()
        if self._head_supported:
            names.extend(HEAD_JOINTS)
            position.extend(command.head_position.tolist())
            kp.extend(HEAD_KP)
            kd.extend(HEAD_KD)

        message = ActuatorCmds()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "elf3"
        message.actuators_name = names
        message.pos = position
        message.kp = kp
        message.kd = kd
        message.vel = [0.0] * len(names)
        message.torque = arm_torque.tolist() + [0.0] * (
            len(names) - len(ARM_JOINTS)
        )
        self._override_publisher.publish(message)
        if not self._override_active:
            self._override_active = True
            self.get_logger().info(
                "HOMIE PICO upper-body override active; grip gates each arm"
            )

    def destroy_node(self) -> bool:
        self._set_gripper_active(False)
        if self._gripper is not None and self._gripper_context is not None:
            self._gripper.unbind(self._gripper_context)
        if self._override_active:
            self._publish_release()
            self._override_active = False
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = HomiePicoArmOverrideNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
