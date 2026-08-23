import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


SCHEMA_VERSION = 1
SUPPORTED_TASKS = ("elf3", "elf3_height", "elf3_walk", "elf3_walk_toeout")
MUJOCO_RESET_ROOT_WORLD_Z_M = 1.05552264
DERIVED_MODEL_CLASSIFICATION = "derived development MJCF; not a vendor-validated final model"
MODEL_CLASSIFICATIONS = {
    "derived": DERIVED_MODEL_CLASSIFICATION,
    "vendor": "vendor-validated final MJCF",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gain_for_joint(joint_name, gains):
    matches = [float(value) for key, value in gains.items() if key in joint_name]
    if len(matches) != 1:
        raise ValueError(f"Joint {joint_name} must match exactly one gain entry, got {len(matches)}")
    return matches[0]


def _joint_effort_limits(urdf_path, joint_names):
    robot = ET.parse(urdf_path).getroot()
    joints = {joint.get("name"): joint for joint in robot.findall("joint")}
    limits = []
    for name in joint_names:
        joint = joints.get(name)
        limit = joint.find("limit") if joint is not None else None
        if joint is None or joint.get("type") != "revolute" or limit is None or limit.get("effort") is None:
            raise ValueError(f"URDF does not define a revolute effort limit for {name}")
        limits.append(float(limit.get("effort")))
    return limits


def command_limits_for_task(task, env_cfg):
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported ELF3 Sim2Sim task: {task}")
    ranges = env_cfg.commands.ranges
    if task == "elf3_height":
        velocity_x = velocity_y = yaw_rate = [0.0, 0.0]
        height = [float(env_cfg.commands.height_min), float(env_cfg.commands.height_max)]
    elif task in ("elf3_walk", "elf3_walk_toeout"):
        velocity_x = [float(value) for value in ranges.lin_vel_x]
        velocity_y = [float(value) for value in ranges.lin_vel_y]
        yaw_rate = [float(value) for value in ranges.ang_vel_yaw]
        height = [float(env_cfg.commands.height_min), float(env_cfg.commands.height_max)]
    else:
        velocity_x = [float(value) for value in ranges.lin_vel_x]
        velocity_y = [float(value) for value in ranges.lin_vel_y]
        yaw_rate = [float(value) for value in ranges.ang_vel_yaw]
        height_target = float(env_cfg.commands.height_target)
        height = [round(height_target + float(value), 12) for value in ranges.height]
    commands = {
        "velocity_x_mps": velocity_x,
        "velocity_y_mps": velocity_y,
        "yaw_rate_radps": yaw_rate,
        "height_m": height,
    }
    if task in ("elf3_height", "elf3_walk", "elf3_walk_toeout"):
        commands["height_slew_rate_mps"] = float(env_cfg.commands.height_slew_rate)
    if task in ("elf3_walk", "elf3_walk_toeout"):
        standing_height = float(env_cfg.commands.height_target)
        commands["default_mode"] = "walk"
        commands["modes"] = {
            "walk": {
                "velocity_x_mps": velocity_x,
                "velocity_y_mps": velocity_y,
                "yaw_rate_radps": yaw_rate,
                "height_m": [standing_height, standing_height],
            },
            "squat": {
                "velocity_x_mps": [0.0, 0.0],
                "velocity_y_mps": [0.0, 0.0],
                "yaw_rate_radps": [0.0, 0.0],
                "height_m": height,
            },
        }
    return commands


def build_contract(
    task,
    env_cfg,
    policy_path,
    model_path,
    urdf_path,
    model_classification=DERIVED_MODEL_CLASSIFICATION,
):
    policy_path = Path(policy_path).resolve()
    model_path = Path(model_path).resolve()
    urdf_path = Path(urdf_path).resolve()
    policy_joints = list(env_cfg.asset.policy_dof_names)
    upper_joints = list(env_cfg.asset.upper_body_dof_names)
    observation_joints = policy_joints + upper_joints
    default_angles = [float(env_cfg.init_state.default_joint_angles[name]) for name in observation_joints]
    stiffness = [_gain_for_joint(name, env_cfg.control.stiffness) for name in observation_joints]
    damping = [_gain_for_joint(name, env_cfg.control.damping) for name in observation_joints]
    effort_limits = _joint_effort_limits(urdf_path, observation_joints)
    one_step = int(env_cfg.env.num_one_step_observations)
    history = int(env_cfg.env.num_actor_history)
    observation_dim = int(env_cfg.env.num_observations)
    action_dim = int(env_cfg.env.num_actions)
    if (len(observation_joints), one_step, history, observation_dim, action_dim) != (26, 74, 6, 444, 12):
        raise ValueError(
            "ELF3 Sim2Sim requires 26 observation joints, 74 one-step observations, "
            "6 history frames, 444 policy inputs and 12 actions"
        )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "policy": {
            "format": "onnx",
            "sha256": sha256_file(policy_path),
            "input_dim": observation_dim,
            "output_dim": action_dim,
        },
        "model": {
            "classification": model_classification,
            "file": model_path.name,
            "sha256": sha256_file(model_path),
            "nq": 33,
            "nv": 32,
            "nu": 26,
            "total_body_mass_kg": 43.222480280000006,
        },
        "observation": {
            "one_step_dim": one_step,
            "history_length": history,
            "history_order": "oldest_to_newest",
            "joint_order": observation_joints,
            "layout": ["command_3", "height_1", "imu_angular_velocity_3", "projected_gravity_3", "joint_position_26", "joint_velocity_26", "previous_action_12"],
            "command_scale": [
                float(env_cfg.normalization.obs_scales.lin_vel),
                float(env_cfg.normalization.obs_scales.lin_vel),
                float(env_cfg.normalization.obs_scales.ang_vel),
            ],
            "angular_velocity_scale": float(env_cfg.normalization.obs_scales.ang_vel),
            "joint_position_scale": float(env_cfg.normalization.obs_scales.dof_pos),
            "joint_velocity_scale": float(env_cfg.normalization.obs_scales.dof_vel),
            "clip": float(env_cfg.normalization.clip_observations),
        },
        "action": {
            "joint_order": policy_joints,
            "scale": float(env_cfg.control.action_scale),
            "clip": float(env_cfg.normalization.clip_actions),
        },
        "control": {
            "physics_dt_s": float(env_cfg.sim.dt),
            "decimation": int(env_cfg.control.decimation),
            "policy_dt_s": float(env_cfg.sim.dt * env_cfg.control.decimation),
            "stiffness": stiffness,
            "damping": damping,
            "effort_limits": effort_limits,
            "upper_body_target": "default_pose",
        },
        "reset": {
            "mujoco_root_world_z_m": MUJOCO_RESET_ROOT_WORLD_Z_M,
            "default_joint_position": default_angles,
        },
        "commands": command_limits_for_task(task, env_cfg),
        "bodies": {
            "imu_site": "imu",
            "torso_body": "torso_link",
            "foot_bodies": ["l_ankle_x_link", "r_ankle_x_link"],
            "foot_geoms": ["l_foot_collision", "r_foot_collision"],
            "floor_geom": "floor",
        },
    }
    if task == "elf3_walk_toeout":
        contract["squat_toe_out_profile"] = {
            "start_height_m": float(env_cfg.commands.toe_out_start_height),
            "full_angle_height_m": float(
                env_cfg.commands.toe_out_full_angle_height
            ),
            "left_max_angle_deg": float(env_cfg.commands.toe_out_max_angle_deg),
            "right_max_angle_deg": -float(env_cfg.commands.toe_out_max_angle_deg),
            "feet_distance_range_m": [
                float(env_cfg.rewards.least_feet_distance_lateral),
                float(env_cfg.rewards.most_feet_distance_lateral),
            ],
        }
    return contract


def write_contract(contract, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_contract(path):
    path = Path(path)
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported Sim2Sim contract schema: {contract.get('schema_version')}")
    if contract.get("task") not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported Sim2Sim contract task: {contract.get('task')}")
    return contract


def validate_command(
    contract,
    velocity_x,
    velocity_y,
    yaw_rate,
    height,
    mode=None,
    tolerance=1.0e-9,
):
    values = {
        "velocity_x_mps": float(velocity_x),
        "velocity_y_mps": float(velocity_y),
        "yaw_rate_radps": float(yaw_rate),
        "height_m": float(height),
    }
    for key, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"Command {key} must be finite")
    commands = contract["commands"]
    modes = commands.get("modes")
    if mode is not None:
        if not modes or mode not in modes:
            raise ValueError(f"Unsupported command mode: {mode}")
        limit_sets = [(mode, modes[mode])]
    elif modes:
        limit_sets = list(modes.items())
    else:
        limit_sets = [(None, commands)]

    for _, limits in limit_sets:
        if all(
            limits[key][0] - tolerance <= value <= limits[key][1] + tolerance
            for key, value in values.items()
        ):
            return values

    if mode is None:
        raise ValueError("Command combination is outside every supported mode")
    details = ", ".join(
        f"{key}={value} in [{limits[key][0]}, {limits[key][1]}]"
        for key, value in values.items()
    )
    raise ValueError(f"Command mode {mode} rejected: {details}")
