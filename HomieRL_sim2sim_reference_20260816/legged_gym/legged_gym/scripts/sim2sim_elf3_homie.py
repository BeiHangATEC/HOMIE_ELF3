#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

import hashlib

import mujoco
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "resources/robots/elf3_dof31/mjcf/r1_derived/elf3_homie_urdf_sole.xml"
DEFAULT_METRICS = ROOT / "logs/elf3_dof31/sim2sim_elf3_homie_metrics.json"
JOINT_ORDER = [
    "head_z_joint", "head_y_joint",
    "l_shoulder_y_joint", "l_shoulder_x_joint", "l_shoulder_z_joint", "l_elbow_y_joint",
    "l_wrist_x_joint", "l_wrist_y_joint", "l_wrist_z_joint",
    "r_shoulder_y_joint", "r_shoulder_x_joint", "r_shoulder_z_joint", "r_elbow_y_joint",
    "r_wrist_x_joint", "r_wrist_y_joint", "r_wrist_z_joint",
    "waist_y_joint", "waist_x_joint", "waist_z_joint",
    "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint", "l_knee_y_joint", "l_ankle_y_joint", "l_ankle_x_joint",
    "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint", "r_knee_y_joint", "r_ankle_y_joint", "r_ankle_x_joint",
]
ACTION_JOINT_ORDER = [
    "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint", "l_knee_y_joint", "l_ankle_y_joint", "l_ankle_x_joint",
    "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint", "r_knee_y_joint", "r_ankle_y_joint", "r_ankle_x_joint",
]
DEFAULT_ANGLES = {name: 0.0 for name in JOINT_ORDER}
STIFFNESS = {"hip_z": 90, "hip_x": 90, "hip_y": 100, "knee": 140, "ankle": 35,
             "waist": 180, "head": 20, "shoulder": 80, "elbow": 50, "wrist": 20}
DAMPING = {"hip_z": 2.0, "hip_x": 2.0, "hip_y": 2.5, "knee": 4.0, "ankle": 1.5,
           "waist": 4.0, "head": 0.5, "shoulder": 2.0, "elbow": 1.0, "wrist": 0.5}
PHYSICS_DT = 0.005
DECIMATION = 4
POLICY_DT = PHYSICS_DT * DECIMATION
ACTION_SCALE = 0.25
ACTION_CLIP = 100.0
OBS_CLIP = 100.0


def load_contract(policy_path, contract_path=None):
    path = contract_path or policy_path.with_name("r1_contract.json")
    if not path.is_file():
        raise FileNotFoundError(f"R1 contract 不存在: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    required = ("body_height_command_m", "mujoco_reset_root_world_z_m", "default_pose_rad")
    missing = [key for key in required if key not in contract]
    if missing:
        raise RuntimeError(f"R1 contract 缺少字段: {missing}")
    return path, contract


def named_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"MuJoCo 模型缺少对象: {name}")
    return object_id


def gain(name, table):
    for key, value in table.items():
        if key in name:
            return value
    raise RuntimeError(f"关节 {name} 没有 PD 参数")


def make_mapping(model):
    joint_ids = np.asarray([named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_ORDER])
    actuator_ids = np.asarray([named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_ORDER])
    if len(set(joint_ids.tolist())) != 31 or len(set(actuator_ids.tolist())) != 31:
        raise RuntimeError("关节或 actuator 名称映射存在重复")
    if any(model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE for joint_id in joint_ids):
        raise RuntimeError("31 个活动关节必须全部是单自由度 hinge")
    if not np.array_equal(model.actuator_trnid[actuator_ids, 0], joint_ids):
        raise RuntimeError("motor 名称与其驱动关节不一致")
    qpos = model.jnt_qposadr[joint_ids]
    qvel = model.jnt_dofadr[joint_ids]
    action_to_full = np.asarray([JOINT_ORDER.index(name) for name in ACTION_JOINT_ORDER])
    return joint_ids, qpos, qvel, actuator_ids, action_to_full


def body_velocity(model, data, body_id):
    velocity = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 1)
    return velocity


def rotate_world_to_body(quat_wxyz, vector):
    inverse = np.asarray(quat_wxyz, dtype=np.float64).copy()
    inverse[1:] *= -1
    result = np.empty(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(result, np.asarray(vector, dtype=np.float64), inverse)
    return result


def make_frame(model, data, torso_id, qpos, qvel, default, applied_action, command, height):
    angular_velocity = body_velocity(model, data, torso_id)[:3]
    projected_gravity = rotate_world_to_body(data.xquat[torso_id], [0.0, 0.0, -1.0])
    frame = np.concatenate((
        command * np.asarray([2.0, 2.0, 0.5]),
        [height], angular_velocity * 0.5, projected_gravity,
        data.qpos[qpos] - default, data.qvel[qvel] * 0.05, applied_action,
    )).astype(np.float32)
    if frame.shape != (84,) or not np.isfinite(frame).all():
        raise RuntimeError(f"单帧观测应为有限 84 维，实际 shape={frame.shape}")
    return frame


def load_policy(path):
    if not path.is_file():
        raise FileNotFoundError(f"ONNX policy 不存在: {path}")
    policy = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    model_input = policy.get_inputs()[0]
    model_output = policy.get_outputs()[0]
    if model_input.shape[1] != 504 or model_output.shape[1] != 12:
        raise RuntimeError(f"ONNX policy 契约不兼容: input={model_input.shape}, output={model_output.shape}")
    output = policy.run([model_output.name], {model_input.name: np.zeros((1, 504), dtype=np.float32)})[0]
    if output.shape != (1, 12) or not np.isfinite(output).all():
        raise RuntimeError(f"ONNX policy 对零输入输出无效: shape={output.shape}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return policy, model_input.name, model_output.name, digest


def profile_height(profile, elapsed, fixed_height):
    if profile != "fixed":
        raise ValueError("R1 only supports the fixed calibrated standing height")
    return fixed_height


def roll_pitch(quat_wxyz):
    w, x, y, z = quat_wxyz
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    return math.degrees(roll), math.degrees(pitch)


def simulate(args):
    if args.duration <= 0:
        raise ValueError("--duration 必须大于 0")
    contract_path, contract = load_contract(args.policy, args.contract)
    body_height = args.body_height if args.body_height is not None else float(contract["body_height_command_m"])
    root_height = args.root_height if args.root_height is not None else float(contract["mujoco_reset_root_world_z_m"])
    default_angles = contract["default_pose_rad"]
    for side in ("l", "r"):
        DEFAULT_ANGLES[f"{side}_hip_y_joint"] = default_angles["hip_y"]
        DEFAULT_ANGLES[f"{side}_knee_y_joint"] = default_angles["knee_y"]
        DEFAULT_ANGLES[f"{side}_ankle_y_joint"] = default_angles["ankle_y"]
    model = mujoco.MjModel.from_xml_path(str(args.model))
    if model.nu != 31 or model.nsensor < 3 or not math.isclose(model.opt.timestep, PHYSICS_DT):
        raise RuntimeError(f"模型契约不符: nu={model.nu}, nsensor={model.nsensor}, timestep={model.opt.timestep}")
    _, qpos, qvel, actuator_ids, action_to_full = make_mapping(model)
    torso_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    floor_id = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = {named_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in ("l_ankle_x_link", "r_ankle_x_link")}
    foot_geom_ids = [named_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                     for name in ("l_foot_collision", "r_foot_collision")]
    policy, input_name, output_name, policy_sha256 = load_policy(args.policy)

    default = np.asarray([DEFAULT_ANGLES[name] for name in JOINT_ORDER])
    kp = np.asarray([gain(name, STIFFNESS) for name in JOINT_ORDER])
    kd = np.asarray([gain(name, DAMPING) for name in JOINT_ORDER])
    effort = np.max(np.abs(model.actuator_ctrlrange[actuator_ids]), axis=1)
    data = mujoco.MjData(model)
    root_joint = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_qpos = model.jnt_qposadr[root_joint]
    data.qpos[root_qpos:root_qpos + 7] = [0.0, 0.0, root_height, 1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos] = default
    mujoco.mj_forward(model, data)

    command = np.asarray([args.vx, args.vy, args.yaw], dtype=np.float32)
    applied_action = np.zeros(12, dtype=np.float32)
    initial_frame = make_frame(model, data, torso_id, qpos, qvel, default, applied_action, command, body_height)
    history = np.repeat(initial_frame[None, :], 6, axis=0)
    viewer_handle = None
    if not args.headless:
        from mujoco import viewer
        viewer_handle = viewer.launch_passive(model, data)
    policy_steps = int(math.ceil(args.duration / POLICY_DT))
    wall_start = time.perf_counter()
    heights, rolls, pitches, yaws, velocities, actions, torques = [], [], [], [], [], [], []
    feet_y, feet_distance, foot_slip = [], [], []
    nonfoot_contact_steps = completed_physics_steps = saturation_count = saturation_total = 0
    finite = True
    fall_reason = None
    commanded_heights = []

    for policy_step in range(policy_steps):
        if viewer_handle is not None and not viewer_handle.is_running():
            break
        target_height = profile_height(args.height_profile, policy_step * POLICY_DT, body_height)
        commanded_heights.append(target_height)
        frame = make_frame(model, data, torso_id, qpos, qvel, default, applied_action, command, target_height)
        history[:-1] = history[1:]
        history[-1] = frame
        observation = np.clip(history.reshape(1, 504), -OBS_CLIP, OBS_CLIP).astype(np.float32)
        action = policy.run([output_name], {input_name: np.ascontiguousarray(observation)})[0][0]
        if action.shape != (12,) or not np.isfinite(action).all():
            finite = False
            break
        action = np.clip(action, -ACTION_CLIP, ACTION_CLIP).astype(np.float32)
        applied_action = action.copy()
        actions.extend(action.tolist())
        target = default.copy()
        target[action_to_full] += ACTION_SCALE * action

        for _ in range(DECIMATION):
            raw_torque = kp * (target - data.qpos[qpos]) - kd * data.qvel[qvel]
            if not np.isfinite(raw_torque).all():
                finite = False
                break
            saturation_count += int(np.count_nonzero(np.abs(raw_torque) >= effort))
            saturation_total += raw_torque.size
            torque = np.clip(raw_torque, -effort, effort)
            torques.extend(torque.tolist())
            data.ctrl[actuator_ids] = torque
            mujoco.mj_step(model, data)
            completed_physics_steps += 1
            finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
            if not finite:
                break
            sole_bottoms = []
            for geom_id in foot_geom_ids:
                center = data.geom_xpos[geom_id]
                rotation = data.geom_xmat[geom_id].reshape(3, 3)
                half_size = model.geom_size[geom_id]
                corners = [center + rotation @ (half_size * np.asarray([sx, sy, sz]))
                           for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
                sole_bottoms.append(min(corner[2] for corner in corners))
            heights.append(float(data.xpos[torso_id, 2] - max(sole_bottoms)))
            roll, pitch = roll_pitch(data.xquat[torso_id])
            rolls.append(roll)
            pitches.append(pitch)
            w, x, y, z = data.xquat[torso_id]
            yaws.append(math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
            velocity = body_velocity(model, data, torso_id)
            velocities.append([velocity[3], velocity[4], velocity[2]])
            signed_y = [float(data.xpos[body_id, 1] - data.xpos[torso_id, 1]) for body_id in sorted(foot_ids)]
            feet_y.append(signed_y)
            feet_distance.append(abs(signed_y[0] - signed_y[1]))
            floor_bodies = set()
            contacting_foot_ids = set()
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                if floor_id in (contact.geom1, contact.geom2):
                    other = contact.geom2 if contact.geom1 == floor_id else contact.geom1
                    body_id = int(model.geom_bodyid[other])
                    floor_bodies.add(body_id)
                    if body_id in foot_ids:
                        contacting_foot_ids.add(body_id)
            nonfoot_contact_steps += int(any(body not in foot_ids for body in floor_bodies))
            for body_id in contacting_foot_ids:
                foot_velocity = body_velocity(model, data, body_id)
                foot_slip.append(float(np.linalg.norm(foot_velocity[3:5])))
            if heights[-1] <= 0.4:
                fall_reason = "torso_to_supporting_sole_height_below_0.4m"
            elif abs(roll) >= 60.0:
                fall_reason = "torso_roll_exceeded_60deg"
            elif abs(pitch) >= 60.0:
                fall_reason = "torso_pitch_exceeded_60deg"
            if viewer_handle is not None:
                viewer_handle.sync()
                delay = wall_start + completed_physics_steps * PHYSICS_DT - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        if not finite:
            break

    if viewer_handle is not None:
        viewer_handle.close()
    velocity_array = np.asarray(velocities)
    tracking_rmse = np.sqrt(np.mean((velocity_array - command) ** 2, axis=0)) if len(velocity_array) else np.full(3, np.nan)
    height_array = np.asarray(heights)
    action_array = np.asarray(actions)
    torque_array = np.asarray(torques)
    completed_policy_steps = completed_physics_steps // DECIMATION
    alive = bool(finite and completed_policy_steps == policy_steps and len(heights) and heights[-1] > 0.4
                 and max(np.max(np.abs(rolls)), np.max(np.abs(pitches))) < 60.0)
    if not finite:
        fall_reason = "nonfinite_state_or_action"
    elif not alive and fall_reason is None:
        fall_reason = "simulation_incomplete"
    metrics = {
        "alive": alive,
        "finite": finite,
        "fall_reason": None if alive else fall_reason,
        "policy_path": str(args.policy.resolve()),
        "policy_sha256": policy_sha256,
        "model_path": str(args.model.resolve()),
        "contract_path": str(contract_path.resolve()),
        "reset_root_world_z_m": root_height,
        "requested_duration_s": args.duration,
        "sim_time_s": completed_physics_steps * PHYSICS_DT,
        "completed_policy_steps": completed_policy_steps,
        "base_height_m": {"min": float(height_array.min()), "mean": float(height_array.mean()), "final": float(height_array[-1])} if len(height_array) else None,
        "height_profile": args.height_profile,
        "body_height_command_m": body_height,
        "base_height_error_rmse_m": float(np.sqrt(np.mean((height_array - np.repeat(commanded_heights, DECIMATION)[:len(height_array)]) ** 2))) if len(height_array) else None,
        "roll_deg": {"max_abs": float(np.max(np.abs(rolls))), "final": float(rolls[-1])} if rolls else None,
        "pitch_deg": {"max_abs": float(np.max(np.abs(pitches))), "final": float(pitches[-1])} if pitches else None,
        "yaw_deg": {"max_abs": float(np.max(np.abs(yaws))), "final": float(yaws[-1])} if yaws else None,
        "signed_feet_y_m": {"left_mean": float(np.mean(np.asarray(feet_y)[:, 0])), "right_mean": float(np.mean(np.asarray(feet_y)[:, 1]))} if feet_y else None,
        "feet_distance_m": {"min": float(np.min(feet_distance)), "mean": float(np.mean(feet_distance)), "max": float(np.max(feet_distance))} if feet_distance else None,
        "foot_slip_mps": {"mean": float(np.mean(foot_slip)), "max": float(np.max(foot_slip))} if foot_slip else None,
        "velocity_command": command.tolist(),
        "velocity_tracking_rmse": tracking_rmse.tolist(),
        "action_finite": bool(np.isfinite(action_array).all()),
        "action_abs_max": float(np.max(np.abs(action_array))) if action_array.size else None,
        "torque_finite": bool(np.isfinite(torque_array).all()),
        "torque_abs_max_nm": float(np.max(np.abs(torque_array))) if torque_array.size else None,
        "torque_saturation_fraction": saturation_count / saturation_total if saturation_total else None,
        "nonfoot_contact_fraction": nonfoot_contact_steps / completed_physics_steps if completed_physics_steps else None,
        "physics_frequency_hz": 1.0 / PHYSICS_DT,
        "policy_frequency_hz": 1.0 / POLICY_DT,
        "observation_contract": "504 = 6 frames x 84, frame-major oldest-to-newest; each frame [cmd3,height1,ang_vel3,gravity3,q31,dq31,action12]",
        "action_joint_order": ACTION_JOINT_ORDER,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0 if finite else 2


def main():
    parser = argparse.ArgumentParser(description="ELF3 HOMIE Isaac Gym -> MuJoCo Sim2Sim")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--height", "--body-height", dest="body_height", type=float, default=None, help="policy body-height命令；默认读取导出contract")
    parser.add_argument("--height_profile", choices=("fixed",), default="fixed")
    parser.add_argument("--root-height", type=float, default=None, help="MuJoCo reset root world z；默认读取导出contract，可显式覆盖")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--headless", action="store_true", help="无界面运行（默认建议用于服务器）")
    parser.add_argument("--policy", type=Path, required=True, help="归档导出的原始 ONNX policy")
    parser.add_argument("--contract", type=Path, help="R1 contract JSON；默认使用policy同目录r1_contract.json")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args()
    raise SystemExit(simulate(args))


if __name__ == "__main__":
    main()
