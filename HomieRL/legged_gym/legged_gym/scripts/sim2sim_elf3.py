#!/usr/bin/env python3
import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from sim2sim_contract import (  # noqa: E402
    load_contract,
    sha256_file,
    validate_command,
)
from command_slider import SliderCommand, SliderControlPanel  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "resources/robots/elf3_description/mjcf/elf3_fixed_waist.xml"
DEFAULT_METRICS = ROOT / "logs/sim2sim_elf3_metrics.json"
MAX_TILT_DEG = 60.0
MAX_HEIGHT_RMSE_M = 0.10
MAX_LINEAR_VELOCITY_RMSE_MPS = 0.30
MAX_YAW_RATE_RMSE_RADPS = 0.40
MAX_FOOT_YAW_RMSE_DEG = 5.0
MAX_FULL_TOE_OUT_MEAN_ERROR_DEG = 3.0


def slider_limits_from_contract(contract):
    commands = contract["commands"]
    return {
        "x_vel": tuple(commands["velocity_x_mps"]),
        "y_vel": tuple(commands["velocity_y_mps"]),
        "yaw_vel": tuple(commands["yaw_rate_radps"]),
        "height": tuple(commands["height_m"]),
    }


def slider_mode_limits_from_contract(contract):
    modes = contract["commands"].get("modes", {})
    return {
        mode: {
            "x_vel": tuple(limits["velocity_x_mps"]),
            "y_vel": tuple(limits["velocity_y_mps"]),
            "yaw_vel": tuple(limits["yaw_rate_radps"]),
            "height": tuple(limits["height_m"]),
        }
        for mode, limits in modes.items()
    }


def command_from_snapshot(contract, snapshot, mode=None):
    values = validate_command(
        contract,
        snapshot["x_vel"],
        snapshot["y_vel"],
        snapshot["yaw_vel"],
        snapshot["height"],
        mode=mode,
    )
    command = np.asarray(
        [
            values["velocity_x_mps"],
            values["velocity_y_mps"],
            values["yaw_rate_radps"],
        ],
        dtype=np.float32,
    )
    return command, values["height_m"]


def slew_value(current, target, rate, dt):
    max_delta = max(float(rate) * float(dt), 0.0)
    return float(current + np.clip(float(target) - float(current), -max_delta, max_delta))


def named_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"MuJoCo model is missing {name}")
    return object_id


def validate_contract(contract):
    observation = contract["observation"]
    action = contract["action"]
    control = contract["control"]
    reset = contract["reset"]
    if (
        contract["policy"]["format"] != "onnx"
        or observation["history_order"] != "oldest_to_newest"
        or contract["policy"]["input_dim"] != 444
        or contract["policy"]["output_dim"] != 12
        or observation["one_step_dim"] != 74
        or observation["history_length"] != 6
        or len(observation["joint_order"]) != 26
        or len(action["joint_order"]) != 12
    ):
        raise RuntimeError("ELF3 policy contract must be 444 -> 12 with 6 x 74 observations")
    if len(set(observation["joint_order"])) != 26:
        raise RuntimeError("Observation joint names must be unique")
    if len(set(action["joint_order"])) != 12 or not set(action["joint_order"]).issubset(
        observation["joint_order"]
    ):
        raise RuntimeError("Action joints must be 12 unique observation joints")
    for key in ("stiffness", "damping", "effort_limits"):
        if len(control[key]) != 26 or not np.isfinite(control[key]).all():
            raise RuntimeError(f"Control field {key} must contain 26 finite values")
    if len(reset["default_joint_position"]) != 26 or not np.isfinite(
        reset["default_joint_position"]
    ).all():
        raise RuntimeError("Reset default pose must contain 26 finite values")
    expected_policy_dt = control["physics_dt_s"] * control["decimation"]
    if not math.isclose(control["policy_dt_s"], expected_policy_dt, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("Policy timestep does not match physics timestep and decimation")
    commands = contract["commands"]
    modes = commands.get("modes", {})
    if modes:
        default_mode = commands.get("default_mode")
        if default_mode not in modes:
            raise RuntimeError("Command default mode is missing from mode limits")
        required = {
            "velocity_x_mps",
            "velocity_y_mps",
            "yaw_rate_radps",
            "height_m",
        }
        for mode, limits in modes.items():
            if set(limits) != required:
                raise RuntimeError(f"Command mode {mode} has an invalid layout")
            for key in required:
                lower, upper = limits[key]
                envelope_lower, envelope_upper = commands[key]
                if lower > upper or lower < envelope_lower or upper > envelope_upper:
                    raise RuntimeError(f"Command mode {mode} exceeds {key} envelope")
    profile = contract.get("squat_toe_out_profile")
    if profile is not None:
        required = {
            "start_height_m",
            "full_angle_height_m",
            "left_max_angle_deg",
            "right_max_angle_deg",
            "feet_distance_range_m",
        }
        if set(profile) != required:
            raise RuntimeError("Squat toe-out profile has an invalid layout")
        scalar_values = [profile[key] for key in required if key != "feet_distance_range_m"]
        distance_range = profile["feet_distance_range_m"]
        if (
            len(distance_range) != 2
            or not np.isfinite(scalar_values + distance_range).all()
            or profile["full_angle_height_m"] >= profile["start_height_m"]
            or profile["left_max_angle_deg"] < 0.0
            or profile["right_max_angle_deg"] != -profile["left_max_angle_deg"]
            or distance_range[0] > distance_range[1]
        ):
            raise RuntimeError("Squat toe-out profile values are invalid")


def load_policy(path, contract):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ONNX policy does not exist: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != contract["policy"]["sha256"]:
        raise RuntimeError(
            f"ONNX policy SHA-256 mismatch: expected {contract['policy']['sha256']}, got {actual_hash}"
        )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    input_dim = contract["policy"]["input_dim"]
    output_dim = contract["policy"]["output_dim"]
    if model_input.shape[1] != input_dim or model_output.shape[1] != output_dim:
        raise RuntimeError(
            f"ONNX shape mismatch: input={model_input.shape}, output={model_output.shape}"
        )
    output = session.run(
        [model_output.name],
        {model_input.name: np.zeros((1, input_dim), dtype=np.float32)},
    )[0]
    if output.shape != (1, output_dim) or not np.isfinite(output).all():
        raise RuntimeError(f"ONNX zero-input output is invalid: shape={output.shape}")
    return session, model_input.name, model_output.name, actual_hash


def load_model(path, contract):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MuJoCo model does not exist: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != contract["model"]["sha256"]:
        raise RuntimeError(
            f"MuJoCo model SHA-256 mismatch: expected {contract['model']['sha256']}, got {actual_hash}"
        )
    model = mujoco.MjModel.from_xml_path(str(path))
    expected = contract["model"]
    actual_shape = (model.nq, model.nv, model.nu)
    expected_shape = (expected["nq"], expected["nv"], expected["nu"])
    if actual_shape != expected_shape:
        raise RuntimeError(f"MuJoCo model shape mismatch: expected {expected_shape}, got {actual_shape}")
    if not math.isclose(
        model.opt.timestep,
        contract["control"]["physics_dt_s"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(f"MuJoCo timestep mismatch: {model.opt.timestep}")
    if not math.isclose(
        float(model.body_mass.sum()),
        expected["total_body_mass_kg"],
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        raise RuntimeError(f"MuJoCo model mass mismatch: {model.body_mass.sum()}")
    return model, actual_hash


def make_mapping(model, contract):
    joint_order = contract["observation"]["joint_order"]
    action_order = contract["action"]["joint_order"]
    joint_ids = np.asarray(
        [named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_order]
    )
    actuator_ids = np.asarray(
        [named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in joint_order]
    )
    if len(set(joint_ids.tolist())) != 26 or len(set(actuator_ids.tolist())) != 26:
        raise RuntimeError("Joint or actuator name mapping contains duplicates")
    if any(model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE for joint_id in joint_ids):
        raise RuntimeError("All 26 observation joints must be one-DoF hinges")
    if not np.array_equal(model.actuator_trnid[actuator_ids, 0], joint_ids):
        raise RuntimeError("Each named actuator must drive the joint with the same name")
    model_effort = np.max(np.abs(model.actuator_ctrlrange[actuator_ids]), axis=1)
    contract_effort = np.asarray(contract["control"]["effort_limits"], dtype=np.float64)
    if not np.allclose(model_effort, contract_effort, rtol=0.0, atol=1.0e-9):
        raise RuntimeError("MJCF actuator limits do not match the exported contract")
    action_to_full = np.asarray([joint_order.index(name) for name in action_order])
    return {
        "qpos": model.jnt_qposadr[joint_ids],
        "qvel": model.jnt_dofadr[joint_ids],
        "actuator": actuator_ids,
        "action_to_full": action_to_full,
    }


def object_velocity(model, data, object_type, object_id):
    velocity = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, object_type, object_id, velocity, 1)
    return velocity


def make_frame(model, data, mapping, imu_site_id, default, previous_action, command, height, contract):
    observation = contract["observation"]
    imu_velocity = object_velocity(model, data, mujoco.mjtObj.mjOBJ_SITE, imu_site_id)
    projected_gravity = data.site_xmat[imu_site_id].reshape(3, 3).T @ np.asarray(
        [0.0, 0.0, -1.0]
    )
    frame = np.concatenate(
        (
            command * np.asarray(observation["command_scale"]),
            [height],
            imu_velocity[:3] * observation["angular_velocity_scale"],
            projected_gravity,
            (data.qpos[mapping["qpos"]] - default) * observation["joint_position_scale"],
            data.qvel[mapping["qvel"]] * observation["joint_velocity_scale"],
            previous_action,
        )
    ).astype(np.float32)
    if frame.shape != (observation["one_step_dim"],) or not np.isfinite(frame).all():
        raise RuntimeError(f"One-step observation is invalid: shape={frame.shape}")
    return frame


def sole_bottom(model, data, geom_id):
    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
        raise RuntimeError("ELF3 supporting foot geoms must be boxes")
    center = data.geom_xpos[geom_id]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    half_size = model.geom_size[geom_id]
    return min(
        (center + rotation @ (half_size * np.asarray([sx, sy, sz])))[2]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    )


def roll_pitch_degrees(quaternion_wxyz):
    w, x, y, z = quaternion_wxyz
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = math.degrees(
        math.asin(float(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    )
    return roll, pitch


def contact_bodies(model, data, floor_geom_id):
    bodies = set()
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if floor_geom_id not in (contact.geom1, contact.geom2):
            continue
        other_geom = contact.geom2 if contact.geom1 == floor_geom_id else contact.geom1
        bodies.add(int(model.geom_bodyid[other_geom]))
    return bodies


def stable_slice(values, physics_dt, requested_duration):
    if len(values) == 0:
        return np.asarray(values), 0.0
    warmup_seconds = min(5.0, requested_duration * 0.2)
    warmup_steps = min(len(values) - 1, int(round(warmup_seconds / physics_dt)))
    return np.asarray(values[warmup_steps:]), warmup_seconds


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def squat_toe_out_targets(height, profile):
    ratio = np.clip(
        (float(profile["start_height_m"]) - float(height))
        / (
            float(profile["start_height_m"])
            - float(profile["full_angle_height_m"])
        ),
        0.0,
        1.0,
    )
    return np.radians(
        ratio
        * np.asarray(
            [profile["left_max_angle_deg"], profile["right_max_angle_deg"]],
            dtype=np.float64,
        )
    )


def foot_pose_metrics(model, data, torso_body_id, foot_body_ids):
    torso_rotation = data.xmat[torso_body_id].reshape(3, 3)
    torso_position = data.xpos[torso_body_id]
    foot_yaws = []
    foot_positions = []
    for foot_body_id in foot_body_ids:
        foot_rotation = data.xmat[foot_body_id].reshape(3, 3)
        forward_in_torso = torso_rotation.T @ foot_rotation[:, 0]
        foot_yaws.append(math.atan2(forward_in_torso[1], forward_in_torso[0]))
        foot_positions.append(
            torso_rotation.T @ (data.xpos[foot_body_id] - torso_position)
        )
    lateral_distance = abs(foot_positions[0][1] - foot_positions[1][1])
    return np.asarray(foot_yaws, dtype=np.float64), float(lateral_distance)


def simulate(args):
    if args.duration <= 0:
        raise ValueError("--duration must be greater than zero")
    contract_path = Path(args.contract).resolve()
    contract = load_contract(contract_path)
    validate_contract(contract)
    height = args.height
    if height is None:
        height = float(contract["commands"]["height_m"][1])
    command_modes = contract["commands"].get("modes", {})
    mode = getattr(args, "mode", None)
    if command_modes:
        mode = mode or contract["commands"]["default_mode"]
    elif mode is not None:
        raise ValueError("--mode requires a contract with command modes")
    command_values = validate_command(
        contract, args.vx, args.vy, args.yaw, height, mode=mode
    )
    initial_snapshot = {
        "x_vel": command_values["velocity_x_mps"],
        "y_vel": command_values["velocity_y_mps"],
        "yaw_vel": command_values["yaw_rate_radps"],
        "height": command_values["height_m"],
    }
    slider_command = SliderCommand(
        **initial_snapshot,
        limits=slider_limits_from_contract(contract),
        mode_limits=slider_mode_limits_from_contract(contract),
        mode=mode,
    )
    command, height = command_from_snapshot(
        contract, slider_command.snapshot(), slider_command.mode_snapshot()
    )
    requested_height = height
    applied_height = height
    initial_mode = mode

    model, model_hash = load_model(args.model, contract)
    mapping = make_mapping(model, contract)
    session, input_name, output_name, policy_hash = load_policy(args.policy, contract)
    bodies = contract["bodies"]
    imu_site_id = named_id(model, mujoco.mjtObj.mjOBJ_SITE, bodies["imu_site"])
    torso_body_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, bodies["torso_body"])
    foot_body_ids = [
        named_id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in bodies["foot_bodies"]
    ]
    foot_geom_ids = [
        named_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in bodies["foot_geoms"]
    ]
    floor_geom_id = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, bodies["floor_geom"])
    root_joint_id = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_qpos = model.jnt_qposadr[root_joint_id]

    control = contract["control"]
    observation = contract["observation"]
    action_contract = contract["action"]
    default = np.asarray(contract["reset"]["default_joint_position"], dtype=np.float64)
    stiffness = np.asarray(control["stiffness"], dtype=np.float64)
    damping = np.asarray(control["damping"], dtype=np.float64)
    effort = np.asarray(control["effort_limits"], dtype=np.float64)
    physics_dt = float(control["physics_dt_s"])
    decimation = int(control["decimation"])
    policy_dt = float(control["policy_dt_s"])

    data = mujoco.MjData(model)
    data.qpos[root_qpos : root_qpos + 7] = [
        0.0,
        0.0,
        float(contract["reset"]["mujoco_root_world_z_m"]),
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    data.qpos[mapping["qpos"]] = default
    mujoco.mj_forward(model, data)

    previous_action = np.zeros(contract["policy"]["output_dim"], dtype=np.float32)
    initial_frame = make_frame(
        model,
        data,
        mapping,
        imu_site_id,
        default,
        previous_action,
        command,
        height,
        contract,
    )
    history = np.repeat(initial_frame[None, :], observation["history_length"], axis=0)
    policy_steps = int(math.ceil(args.duration / policy_dt))
    expected_physics_steps = policy_steps * decimation
    completed_physics_steps = 0
    wall_start = None
    heights = []
    rolls = []
    pitches = []
    velocities = []
    command_targets = []
    height_targets = []
    actions = []
    torques = []
    foot_slips = []
    foot_yaws = []
    foot_yaw_targets = []
    feet_lateral_distances = []
    nonfoot_contact_steps = 0
    torso_contact_steps = 0
    saturation_count = 0
    saturation_total = 0
    finite = True
    fall_reason = None
    viewer_handle = None
    control_panel = None
    stop_event = threading.Event()
    user_stopped = False
    try:
        if not args.headless:
            control_panel = SliderControlPanel(slider_command)
            control_panel.start(stop_event)
            from mujoco import viewer

            viewer_handle = viewer.launch_passive(model, data)
        wall_start = time.perf_counter()

        for _ in range(policy_steps):
            if control_panel is not None:
                control_panel.raise_if_failed()
                if stop_event.is_set():
                    user_stopped = True
                    break
                if not viewer_handle.is_running():
                    user_stopped = True
                    stop_event.set()
                    break
                command_snapshot, mode = slider_command.state_snapshot()
                command, height = command_from_snapshot(
                    contract, command_snapshot, mode
                )
                requested_height = height
                height = slew_value(
                    applied_height,
                    requested_height,
                    contract["commands"].get("height_slew_rate_mps", math.inf),
                    policy_dt,
                )
                if mode == "walk" and not math.isclose(
                    height, requested_height, rel_tol=0.0, abs_tol=1.0e-9
                ):
                    command = np.zeros(3, dtype=np.float32)
            applied_height = height
            frame = make_frame(
                model,
                data,
                mapping,
                imu_site_id,
                default,
                previous_action,
                command,
                height,
                contract,
            )
            history[:-1] = history[1:]
            history[-1] = frame
            policy_input = np.clip(
                history.reshape(1, contract["policy"]["input_dim"]),
                -observation["clip"],
                observation["clip"],
            ).astype(np.float32)
            action = session.run(
                [output_name], {input_name: np.ascontiguousarray(policy_input)}
            )[0][0]
            if action.shape != previous_action.shape or not np.isfinite(action).all():
                finite = False
                fall_reason = "nonfinite_or_invalid_policy_action"
                break
            previous_action = np.clip(
                action, -action_contract["clip"], action_contract["clip"]
            ).astype(np.float32)
            actions.append(previous_action.copy())
            target = default.copy()
            target[mapping["action_to_full"]] += action_contract["scale"] * previous_action

            for _ in range(decimation):
                raw_torque = (
                    stiffness * (target - data.qpos[mapping["qpos"]])
                    - damping * data.qvel[mapping["qvel"]]
                )
                if not np.isfinite(raw_torque).all():
                    finite = False
                    fall_reason = "nonfinite_control_torque"
                    break
                saturation_count += int(np.count_nonzero(np.abs(raw_torque) >= effort))
                saturation_total += raw_torque.size
                torque = np.clip(raw_torque, -effort, effort)
                torques.append(torque.copy())
                data.ctrl[mapping["actuator"]] = torque
                mujoco.mj_step(model, data)
                completed_physics_steps += 1
                finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
                if not finite:
                    fall_reason = "nonfinite_simulation_state"
                    break

                support_bottom = max(sole_bottom(model, data, geom_id) for geom_id in foot_geom_ids)
                actual_height = float(data.xpos[torso_body_id, 2] - support_bottom)
                heights.append(actual_height)
                height_targets.append(height)
                command_targets.append(command.copy())
                actual_foot_yaws, feet_lateral_distance = foot_pose_metrics(
                    model, data, torso_body_id, foot_body_ids
                )
                profile = contract.get("squat_toe_out_profile")
                target_foot_yaws = (
                    squat_toe_out_targets(actual_height, profile)
                    if profile is not None and mode == "squat"
                    else np.zeros(2, dtype=np.float64)
                )
                foot_yaws.append(actual_foot_yaws)
                foot_yaw_targets.append(target_foot_yaws)
                feet_lateral_distances.append(feet_lateral_distance)
                roll, pitch = roll_pitch_degrees(data.xquat[torso_body_id])
                rolls.append(roll)
                pitches.append(pitch)
                torso_velocity = object_velocity(
                    model, data, mujoco.mjtObj.mjOBJ_BODY, torso_body_id
                )
                velocities.append([torso_velocity[3], torso_velocity[4], torso_velocity[2]])
                floor_bodies = contact_bodies(model, data, floor_geom_id)
                nonfoot_contact_steps += int(
                    any(body_id not in foot_body_ids for body_id in floor_bodies)
                )
                torso_contact_steps += int(torso_body_id in floor_bodies)
                for foot_body_id in set(foot_body_ids).intersection(floor_bodies):
                    foot_velocity = object_velocity(
                        model, data, mujoco.mjtObj.mjOBJ_BODY, foot_body_id
                    )
                    foot_slips.append(float(np.linalg.norm(foot_velocity[3:5])))
                if torso_body_id in floor_bodies:
                    fall_reason = "torso_contacted_floor"
                elif max(abs(roll), abs(pitch)) >= MAX_TILT_DEG:
                    fall_reason = "torso_tilt_exceeded_60deg"
                if viewer_handle is not None:
                    viewer_handle.sync()
                    delay = wall_start + completed_physics_steps * physics_dt - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                if fall_reason is not None:
                    break
            if not finite or fall_reason is not None:
                break
    finally:
        stop_event.set()
        if viewer_handle is not None:
            viewer_handle.close()
        if control_panel is not None:
            control_panel.join()

    if control_panel is not None:
        control_panel.raise_if_failed()

    height_array = np.asarray(heights, dtype=np.float64)
    velocity_array = np.asarray(velocities, dtype=np.float64)
    command_target_array = np.asarray(command_targets, dtype=np.float64)
    height_target_array = np.asarray(height_targets, dtype=np.float64)
    action_array = np.asarray(actions, dtype=np.float64)
    torque_array = np.asarray(torques, dtype=np.float64)
    foot_yaw_array = np.asarray(foot_yaws, dtype=np.float64)
    foot_yaw_target_array = np.asarray(foot_yaw_targets, dtype=np.float64)
    feet_lateral_distance_array = np.asarray(
        feet_lateral_distances, dtype=np.float64
    )
    stable_heights, warmup_seconds = stable_slice(height_array, physics_dt, args.duration)
    stable_velocities, _ = stable_slice(velocity_array, physics_dt, args.duration)
    stable_command_targets, _ = stable_slice(
        command_target_array, physics_dt, args.duration
    )
    stable_height_targets, _ = stable_slice(
        height_target_array, physics_dt, args.duration
    )
    stable_foot_yaws, _ = stable_slice(foot_yaw_array, physics_dt, args.duration)
    stable_foot_yaw_targets, _ = stable_slice(
        foot_yaw_target_array, physics_dt, args.duration
    )
    stable_feet_lateral_distances, _ = stable_slice(
        feet_lateral_distance_array, physics_dt, args.duration
    )
    height_rmse = (
        float(np.sqrt(np.mean((stable_heights - stable_height_targets) ** 2)))
        if stable_heights.size
        else float("nan")
    )
    velocity_rmse = (
        np.sqrt(np.mean((stable_velocities - stable_command_targets) ** 2, axis=0))
        if stable_velocities.size
        else np.full(3, np.nan)
    )
    foot_yaw_rmse = (
        np.sqrt(
            np.mean(
                np.square(wrap_angle(stable_foot_yaws - stable_foot_yaw_targets)),
                axis=0,
            )
        )
        if stable_foot_yaws.size
        else np.full(2, np.nan)
    )
    stable_mean_foot_yaw = (
        np.mean(stable_foot_yaws, axis=0)
        if stable_foot_yaws.size
        else np.full(2, np.nan)
    )
    stable_mean_foot_yaw_target = (
        np.mean(stable_foot_yaw_targets, axis=0)
        if stable_foot_yaw_targets.size
        else np.full(2, np.nan)
    )
    completed = completed_physics_steps == expected_physics_steps
    alive = bool(
        finite
        and completed
        and torso_contact_steps == 0
        and rolls
        and max(np.max(np.abs(rolls)), np.max(np.abs(pitches))) < MAX_TILT_DEG
    )
    if not alive and fall_reason is None:
        fall_reason = "stopped_by_user" if user_stopped else "simulation_incomplete"
    gate_checks = {
        "alive": alive,
        "height_rmse": bool(np.isfinite(height_rmse) and height_rmse <= MAX_HEIGHT_RMSE_M),
        "linear_velocity_rmse": bool(
            np.isfinite(velocity_rmse[:2]).all()
            and np.max(velocity_rmse[:2]) <= MAX_LINEAR_VELOCITY_RMSE_MPS
        ),
        "yaw_rate_rmse": bool(
            np.isfinite(velocity_rmse[2])
            and velocity_rmse[2] <= MAX_YAW_RATE_RMSE_RADPS
        ),
    }
    profile = contract.get("squat_toe_out_profile")
    toe_out_gate = profile is not None and mode == "squat"
    if toe_out_gate:
        distance_minimum, distance_maximum = profile["feet_distance_range_m"]
        gate_checks["foot_yaw_rmse"] = bool(
            np.isfinite(foot_yaw_rmse).all()
            and np.max(np.degrees(foot_yaw_rmse)) <= MAX_FOOT_YAW_RMSE_DEG
        )
        gate_checks["feet_lateral_distance"] = bool(
            stable_feet_lateral_distances.size
            and np.isfinite(stable_feet_lateral_distances).all()
            and stable_feet_lateral_distances.min() >= distance_minimum
            and stable_feet_lateral_distances.max() <= distance_maximum
        )
        if requested_height <= profile["full_angle_height_m"] + 1.0e-9:
            expected_full_yaw = np.asarray(
                [profile["left_max_angle_deg"], profile["right_max_angle_deg"]]
            )
            gate_checks["full_toe_out_mean"] = bool(
                np.isfinite(stable_mean_foot_yaw).all()
                and np.max(
                    np.abs(np.degrees(stable_mean_foot_yaw) - expected_full_yaw)
                )
                <= MAX_FULL_TOE_OUT_MEAN_ERROR_DEG
            )
    gate_passed = all(gate_checks.values())
    velocity_rmse_json = [
        float(value) if np.isfinite(value) else None for value in velocity_rmse
    ]
    metrics = {
        "task": contract["task"],
        "alive": alive,
        "finite": finite,
        "gate_requested": bool(args.gate),
        "gate_passed": gate_passed,
        "gate_checks": gate_checks,
        "fall_reason": None if alive else fall_reason,
        "policy_path": str(Path(args.policy).resolve()),
        "policy_sha256": policy_hash,
        "model_path": str(Path(args.model).resolve()),
        "model_sha256": model_hash,
        "model_classification": contract["model"]["classification"],
        "contract_path": str(contract_path),
        "requested_duration_s": args.duration,
        "simulated_duration_s": completed_physics_steps * physics_dt,
        "completed_physics_steps": completed_physics_steps,
        "expected_physics_steps": expected_physics_steps,
        "warmup_excluded_s": warmup_seconds,
        "interactive_commands": not args.headless,
        "initial_mode": initial_mode,
        "final_mode": mode,
        "initial_command": {
            "velocity_x_mps": float(initial_snapshot["x_vel"]),
            "velocity_y_mps": float(initial_snapshot["y_vel"]),
            "yaw_rate_radps": float(initial_snapshot["yaw_vel"]),
            "height_m": float(initial_snapshot["height"]),
        },
        "final_command": {
            "velocity_x_mps": float(command[0]),
            "velocity_y_mps": float(command[1]),
            "yaw_rate_radps": float(command[2]),
            "height_m": float(height),
        },
        "requested_final_height_m": float(requested_height),
        "command": {
            "velocity_x_mps": float(command[0]),
            "velocity_y_mps": float(command[1]),
            "yaw_rate_radps": float(command[2]),
            "height_m": float(height),
        },
        "height_m": {
            "min": float(height_array.min()),
            "mean": float(height_array.mean()),
            "final": float(height_array[-1]),
            "stable_rmse": height_rmse,
        }
        if height_array.size
        else None,
        "velocity_tracking_rmse": velocity_rmse_json,
        "roll_deg": {
            "max_abs": float(np.max(np.abs(rolls))),
            "final": float(rolls[-1]),
        }
        if rolls
        else None,
        "pitch_deg": {
            "max_abs": float(np.max(np.abs(pitches))),
            "final": float(pitches[-1]),
        }
        if pitches
        else None,
        "action_abs_max": float(np.max(np.abs(action_array))) if action_array.size else None,
        "torque_abs_max_nm": float(np.max(np.abs(torque_array))) if torque_array.size else None,
        "torque_saturation_fraction": (
            saturation_count / saturation_total if saturation_total else None
        ),
        "nonfoot_contact_fraction": (
            nonfoot_contact_steps / completed_physics_steps if completed_physics_steps else None
        ),
        "torso_contact_fraction": (
            torso_contact_steps / completed_physics_steps if completed_physics_steps else None
        ),
        "foot_slip_mps": {
            "mean": float(np.mean(foot_slips)),
            "max": float(np.max(foot_slips)),
        }
        if foot_slips
        else None,
        "foot_yaw_deg": {
            "left_stable_mean": float(np.degrees(stable_mean_foot_yaw[0])),
            "right_stable_mean": float(np.degrees(stable_mean_foot_yaw[1])),
            "left_target_stable_mean": float(
                np.degrees(stable_mean_foot_yaw_target[0])
            ),
            "right_target_stable_mean": float(
                np.degrees(stable_mean_foot_yaw_target[1])
            ),
            "left_stable_rmse": float(np.degrees(foot_yaw_rmse[0])),
            "right_stable_rmse": float(np.degrees(foot_yaw_rmse[1])),
        }
        if stable_foot_yaws.size
        else None,
        "feet_center_lateral_distance_m": {
            "stable_min": float(stable_feet_lateral_distances.min()),
            "stable_mean": float(stable_feet_lateral_distances.mean()),
            "stable_max": float(stable_feet_lateral_distances.max()),
        }
        if stable_feet_lateral_distances.size
        else None,
        "physics_frequency_hz": 1.0 / physics_dt,
        "policy_frequency_hz": 1.0 / policy_dt,
        "observation_contract": "444 = 6 x 74, oldest-to-newest; frame=[cmd3,height1,imu_ang_vel3,gravity3,q26,dq26,previous_action12]",
        "observation_joint_order": observation["joint_order"],
        "action_joint_order": action_contract["joint_order"],
    }
    metrics_path = Path(args.metrics).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False))
    stopped_cleanly = (
        user_stopped
        and not args.gate
        and finite
        and fall_reason == "stopped_by_user"
    )
    if not alive and not stopped_cleanly:
        return 2
    if args.gate and not gate_passed:
        return 2
    return 0


def main():
    parser = argparse.ArgumentParser(description="ELF3 Isaac Gym to MuJoCo Sim2Sim")
    parser.add_argument("--policy", type=Path, required=True, help="Exported ONNX policy")
    parser.add_argument("--contract", type=Path, required=True, help="Policy contract JSON")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument(
        "--mode",
        choices=("walk", "squat"),
        default=None,
        help="Conditional command mode for staged elf3_walk policies",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args()
    raise SystemExit(simulate(args))


if __name__ == "__main__":
    main()
