"""Resource and state registration for independent half-body teleoperation."""

from __future__ import annotations

from bxi_example_py_elf3.framework.mod_api import (
    ModDefinition,
    ModLoadContext,
    ResourceKey,
    ResourceLoadContext,
    StateBuildContext,
)
from bxi_example_py_elf3.policies import HumanoidGaitPolicyLiteIsaaclab

from .gravity import ArmGravityModel
from .gripper_session import GripperConfig, GripperSession
from .state import UpperBodyTeleopParams, UpperBodyTeleopState


WITHOUT_ARM_POLICY = ResourceKey[HumanoidGaitPolicyLiteIsaaclab](
    "com.bxi.basic_actions/without_arm_policy"
)
GRAVITY_MODEL = ResourceKey[ArmGravityModel](
    "com.bxi.upper_body_teleop/gravity_model"
)


def _load_gravity_model(context: ResourceLoadContext) -> ArmGravityModel:
    del context
    return ArmGravityModel()


def _seven_values(
    state: StateBuildContext,
    name: str,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    raw = state.param(name, list, list(default))
    if len(raw) != 7:
        raise ValueError(f"state '{state.name}' param '{name}' must have seven values")
    values = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"state '{state.name}' param '{name}[{index}]' must be numeric"
            )
        values.append(float(value))
    return tuple(values)


def _load_state_params(state: StateBuildContext) -> UpperBodyTeleopParams:
    defaults = UpperBodyTeleopParams()
    return UpperBodyTeleopParams(
        reference_topic=state.string_param(
            "reference_topic", defaults.reference_topic
        ),
        robot_state_topic=state.string_param(
            "robot_state_topic", defaults.robot_state_topic
        ),
        live_reference_timeout_s=state.float_param(
            "live_reference_timeout_s", defaults.live_reference_timeout_s
        ),
        grip_threshold=state.float_param(
            "grip_threshold", defaults.grip_threshold
        ),
        arm_gain_ramp_s=state.float_param(
            "arm_gain_ramp_s", defaults.arm_gain_ramp_s
        ),
        arm_kp_scale=state.float_param("arm_kp_scale", defaults.arm_kp_scale),
        gravity_scale=state.float_param(
            "gravity_scale", defaults.gravity_scale
        ),
        torque_limit_scale=state.float_param(
            "torque_limit_scale", defaults.torque_limit_scale
        ),
        friction_coulomb=_seven_values(
            state, "friction_coulomb", defaults.friction_coulomb
        ),
        friction_viscous=_seven_values(
            state, "friction_viscous", defaults.friction_viscous
        ),
        friction_smoothing_velocity=state.float_param(
            "friction_smoothing_velocity", defaults.friction_smoothing_velocity
        ),
        head_control_enabled=state.bool_param(
            "head_control_enabled", defaults.head_control_enabled
        ),
        head_pitch_limit_rad=state.float_param(
            "head_pitch_limit_rad", defaults.head_pitch_limit_rad
        ),
        head_yaw_limit_rad=state.float_param(
            "head_yaw_limit_rad", defaults.head_yaw_limit_rad
        ),
        head_pitch_speed_rad_s=state.float_param(
            "head_pitch_speed_rad_s", defaults.head_pitch_speed_rad_s
        ),
        head_yaw_speed_rad_s=state.float_param(
            "head_yaw_speed_rad_s", defaults.head_yaw_speed_rad_s
        ),
        head_deadband_rad=state.float_param(
            "head_deadband_rad", defaults.head_deadband_rad
        ),
    )


def _load_gripper_config(state: StateBuildContext) -> GripperConfig:
    defaults = GripperConfig()
    return GripperConfig(
        enabled=state.bool_param("hardware_gripper", defaults.enabled),
        enable_interval_s=state.float_param(
            "gripper_enable_interval_s", defaults.enable_interval_s
        ),
        left_bus=state.int_param("gripper_left_bus", defaults.left_bus),
        right_bus=state.int_param("gripper_right_bus", defaults.right_bus),
        can_id=state.int_param("gripper_can_id", defaults.can_id),
        master_id=state.int_param("gripper_master_id", defaults.master_id),
        kp=state.float_param("gripper_kp", defaults.kp),
        kd=state.float_param("gripper_kd", defaults.kd),
        calibration_speed_rad_s=state.float_param(
            "gripper_calibration_speed_rad_s", defaults.calibration_speed_rad_s
        ),
        calibration_kp=state.float_param(
            "gripper_calibration_kp", defaults.calibration_kp
        ),
        calibration_kd=state.float_param(
            "gripper_calibration_kd", defaults.calibration_kd
        ),
        contact_torque=state.float_param(
            "gripper_contact_torque", defaults.contact_torque
        ),
        abort_torque=state.float_param(
            "gripper_abort_torque", defaults.abort_torque
        ),
        contact_confirm_s=state.float_param(
            "gripper_contact_confirm_s", defaults.contact_confirm_s
        ),
        stopped_velocity_rad_s=state.float_param(
            "gripper_stopped_velocity_rad_s", defaults.stopped_velocity_rad_s
        ),
        tracking_error_rad=state.float_param(
            "gripper_tracking_error_rad", defaults.tracking_error_rad
        ),
        limit_margin_rad=state.float_param(
            "gripper_limit_margin_rad", defaults.limit_margin_rad
        ),
        minimum_span_rad=state.float_param(
            "gripper_minimum_span_rad", defaults.minimum_span_rad
        ),
        maximum_search_travel_rad=state.float_param(
            "gripper_maximum_search_travel_rad", defaults.maximum_search_travel_rad
        ),
        response_timeout_s=state.float_param(
            "gripper_response_timeout_s", defaults.response_timeout_s
        ),
        feedback_timeout_s=state.float_param(
            "gripper_feedback_timeout_s", defaults.feedback_timeout_s
        ),
        phase_timeout_s=state.float_param(
            "gripper_phase_timeout_s", defaults.phase_timeout_s
        ),
        maximum_mos_temperature_c=state.int_param(
            "gripper_maximum_mos_temperature_c",
            defaults.maximum_mos_temperature_c,
        ),
        maximum_motor_temperature_c=state.int_param(
            "gripper_maximum_motor_temperature_c",
            defaults.maximum_motor_temperature_c,
        ),
    )


def _build_state(
    state: StateBuildContext,
    policy,
    gravity_model,
) -> UpperBodyTeleopState:
    return UpperBodyTeleopState(
        state.name,
        state.state_id,
        policy,
        gravity_model,
        _load_state_params(state),
        GripperSession(_load_gripper_config(state)),
    )


def create_mod(context: ModLoadContext) -> ModDefinition:
    context.register_resource(
        GRAVITY_MODEL,
        _load_gravity_model,
        policy="on_demand",
    )
    policy = context.resource(WITHOUT_ARM_POLICY)
    gravity_model = context.resource(GRAVITY_MODEL)
    return ModDefinition(
        state_factories={
            "upper_body_teleop": lambda state: _build_state(
                state, policy, gravity_model
            )
        }
    )


__all__ = ["GRAVITY_MODEL", "create_mod"]
