"""Canonical ELF3 facts: joint ordering, groups, gains, geometry, mirror spec.

This module is the single source of truth for the ELF3 robot. Every dimension
used by the environment, the policy and the sim2sim bridge is *derived* from
the joint table below -- no dimension is written as a literal anywhere else.

The ordering here is the HOMIE observation/action ordering. It is NOT the
Isaac Sim runtime ordering; the environment builds a permutation between the
two by joint name at construction time.
"""

from __future__ import annotations

import math
from pathlib import Path


ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "elf3"
URDF_PATH = ASSET_ROOT / "elf3.urdf"
USD_PATH = ASSET_ROOT / "elf3.usd"


# --------------------------------------------------------------------------
# Joint table
# --------------------------------------------------------------------------
# ELF3 has 28 actuated joints. `waist_x_joint` (waist roll) is deliberately
# fixed in the URDF: the real robot does not expose it to the policy. Note the
# vendor MuJoCo model *does* model it as a hinge, so the sim2sim bridge must
# lock it at zero. See MUJOCO_LOCKED_JOINTS.

WAIST_JOINT_NAMES = [
    "waist_y_joint",
    "waist_z_joint",
]

LEFT_LEG_JOINT_NAMES = [
    "l_hip_y_joint",
    "l_hip_x_joint",
    "l_hip_z_joint",
    "l_knee_y_joint",
    "l_ankle_y_joint",
    "l_ankle_x_joint",
]

RIGHT_LEG_JOINT_NAMES = [name.replace("l_", "r_", 1) for name in LEFT_LEG_JOINT_NAMES]

LEFT_ARM_JOINT_NAMES = [
    "l_shoulder_y_joint",
    "l_shoulder_x_joint",
    "l_shoulder_z_joint",
    "l_elbow_y_joint",
    "l_wrist_x_joint",
    "l_wrist_y_joint",
    "l_wrist_z_joint",
]

RIGHT_ARM_JOINT_NAMES = [name.replace("l_", "r_", 1) for name in LEFT_ARM_JOINT_NAMES]

#: Canonical 28-DOF ordering used by observations, actions and checkpoints.
JOINT_NAMES = (
    WAIST_JOINT_NAMES
    + LEFT_LEG_JOINT_NAMES
    + RIGHT_LEG_JOINT_NAMES
    + LEFT_ARM_JOINT_NAMES
    + RIGHT_ARM_JOINT_NAMES
)

#: The 12 joints the policy actually commands.
POLICY_JOINT_NAMES = LEFT_LEG_JOINT_NAMES + RIGHT_LEG_JOINT_NAMES

#: Waist + arms: driven by the upper-body pose curriculum, not by the policy.
UPPER_BODY_JOINT_NAMES = [
    name for name in JOINT_NAMES if name not in POLICY_JOINT_NAMES
]

_JOINT_INDEX = {name: index for index, name in enumerate(JOINT_NAMES)}

LOWER_DOF_INDICES = tuple(_JOINT_INDEX[name] for name in POLICY_JOINT_NAMES)
UPPER_DOF_INDICES = tuple(_JOINT_INDEX[name] for name in UPPER_BODY_JOINT_NAMES)


def joint_index(name: str) -> int:
    """Return the canonical index of a joint, raising on unknown names."""
    try:
        return _JOINT_INDEX[name]
    except KeyError as exc:
        raise KeyError(f"{name!r} is not an ELF3 joint") from exc


# --------------------------------------------------------------------------
# Dimensions -- all derived, never hardcoded
# --------------------------------------------------------------------------
NUM_ROBOT_DOFS = len(JOINT_NAMES)
NUM_POLICY_ACTIONS = len(POLICY_JOINT_NAMES)
NUM_UPPER_BODY_DOFS = len(UPPER_BODY_JOINT_NAMES)

#: commands(vx, vy, wyaw) + height + imu angular velocity + projected gravity
NUM_COMMAND_OBS = 4
NUM_IMU_OBS = 6
NUM_OBS_HEAD = NUM_COMMAND_OBS + NUM_IMU_OBS

#: True base linear velocity, visible to the critic only.
NUM_PRIVILEGED_OBS = 3

NUM_ACTOR_HISTORY = 6
NUM_CRITIC_HISTORY = 1

#: HIM estimator output width: predicted base velocity plus its latent.
NUM_ESTIMATOR_VELOCITY = 3
NUM_ESTIMATOR_LATENT = 32


def num_one_step_actor_obs() -> int:
    """Per-frame actor observation width (78 for the 28-DOF ELF3)."""
    return NUM_OBS_HEAD + 2 * NUM_ROBOT_DOFS + NUM_POLICY_ACTIONS


def num_one_step_critic_obs() -> int:
    """Per-frame critic observation width: actor frame plus base velocity."""
    return num_one_step_actor_obs() + NUM_PRIVILEGED_OBS


def num_actor_obs() -> int:
    """Full actor observation: the frame history the policy receives."""
    return num_one_step_actor_obs() * NUM_ACTOR_HISTORY


def num_critic_obs() -> int:
    """Full critic observation."""
    return num_one_step_critic_obs() * NUM_CRITIC_HISTORY


def num_actor_mlp_input() -> int:
    """Actor network input: latest frame plus the estimator's outputs."""
    return (
        num_one_step_actor_obs() + NUM_ESTIMATOR_VELOCITY + NUM_ESTIMATOR_LATENT
    )


# --------------------------------------------------------------------------
# Default pose
# --------------------------------------------------------------------------
DEFAULT_JOINT_POS = {
    "waist_y_joint": 0.0,
    "waist_z_joint": 0.0,
    "l_hip_y_joint": -0.4,
    "l_hip_x_joint": 0.0,
    "l_hip_z_joint": 0.0,
    "l_knee_y_joint": 0.8,
    "l_ankle_y_joint": -0.4,
    "l_ankle_x_joint": 0.0,
    "r_hip_y_joint": -0.4,
    "r_hip_x_joint": 0.0,
    "r_hip_z_joint": 0.0,
    "r_knee_y_joint": 0.8,
    "r_ankle_y_joint": -0.4,
    "r_ankle_x_joint": 0.0,
    "l_shoulder_y_joint": 0.5,
    "l_shoulder_x_joint": 0.3,
    "l_shoulder_z_joint": -0.1,
    "l_elbow_y_joint": -0.2,
    "l_wrist_x_joint": 0.0,
    "l_wrist_y_joint": 0.0,
    "l_wrist_z_joint": 0.0,
    "r_shoulder_y_joint": 0.5,
    "r_shoulder_x_joint": -0.3,
    "r_shoulder_z_joint": 0.1,
    "r_elbow_y_joint": -0.2,
    "r_wrist_x_joint": 0.0,
    "r_wrist_y_joint": 0.0,
    "r_wrist_z_joint": 0.0,
}


# --------------------------------------------------------------------------
# Link geometry
# --------------------------------------------------------------------------
FOOT_BODY_NAMES = ["l_ankle_x_link", "r_ankle_x_link"]
KNEE_BODY_NAMES = [
    "l_knee_y_link",
    "l_hip_y_link",
    "r_knee_y_link",
    "r_hip_y_link",
]
#: Observation reference frame -- the IMU, not the articulation root.
IMU_BODY_NAME = "imu_link"
TORSO_BODY_NAME = "torso_link"
HAND_BODY_NAMES = ["l_wrist_z_link", "r_wrist_z_link"]

#: Kinematic chain lengths read off the URDF (torso -> waist -> hip -> ankle).
TORSO_TO_HIP_Z = 0.2265 + 0.156
THIGH_LENGTH = 0.32
SHIN_LENGTH = 0.32

#: ELF3 has no per-foot contact marker links, so foot-flatness rewards use a
#: virtual rectangular sole projected from the ankle link pose.
SOLE_LENGTH = 0.24
SOLE_WIDTH = 0.08
SOLE_CENTER_OFFSET = (0.03, 0.0, -0.041)

TOTAL_MASS = 43.22


def torso_height_above_soles(
    hip_pitch: float = DEFAULT_JOINT_POS["l_hip_y_joint"],
    knee_pitch: float = DEFAULT_JOINT_POS["l_knee_y_joint"],
) -> float:
    """Vertical distance from the torso origin to the soles, feet flat.

    This is what the height command means, and it is what fixes the spawn
    height. The previous attempt inherited G1's 0.75 m spawn, which buries
    ELF3's feet 26 cm underground and makes every episode terminate on reset.
    """
    return (
        TORSO_TO_HIP_Z
        + THIGH_LENGTH * math.cos(hip_pitch)
        + SHIN_LENGTH * math.cos(hip_pitch + knee_pitch)
        - SOLE_CENTER_OFFSET[2]
    )


#: Spawn height, consistent with the default joint pose by construction.
DEFAULT_BASE_HEIGHT = torso_height_above_soles()

#: Height command bounds. The lower bound keeps the hip pitch within about
#: -1.2 rad; deeper crouches exist kinematically but demand extreme poses.
#: The upper bound admits the default standing pose, just short of the
#: fully-extended 1.0635 m where the knees would be locked straight.
MIN_HEIGHT_COMMAND = 0.75
MAX_HEIGHT_COMMAND = 1.02


# --------------------------------------------------------------------------
# Actuation
# --------------------------------------------------------------------------
# Regex keyed so both sides match a single entry. Effort and velocity limits
# are deliberately absent: they are read from the URDF at load time.
LEG_STIFFNESS = {
    ".*hip_y_joint": 300.0,
    ".*hip_x_joint": 100.0,
    ".*hip_z_joint": 100.0,
    ".*knee_y_joint": 300.0,
    ".*ankle_y_joint": 50.0,
    ".*ankle_x_joint": 50.0,
}

LEG_DAMPING = {
    ".*hip_y_joint": 2.5,
    ".*hip_x_joint": 2.0,
    ".*hip_z_joint": 2.0,
    ".*knee_y_joint": 2.5,
    ".*ankle_y_joint": 2.0,
    ".*ankle_x_joint": 2.0,
}

UPPER_BODY_STIFFNESS = {
    "waist_y_joint": 500.0,
    "waist_z_joint": 300.0,
    ".*shoulder_y_joint": 100.0,
    ".*shoulder_x_joint": 80.0,
    ".*shoulder_z_joint": 80.0,
    ".*elbow_y_joint": 100.0,
    ".*wrist_x_joint": 20.0,
    ".*wrist_y_joint": 20.0,
    ".*wrist_z_joint": 20.0,
}

UPPER_BODY_DAMPING = {
    "waist_y_joint": 3.0,
    "waist_z_joint": 3.0,
    ".*shoulder_y_joint": 2.5,
    ".*shoulder_x_joint": 2.0,
    ".*shoulder_z_joint": 2.0,
    ".*elbow_y_joint": 2.5,
    ".*wrist_x_joint": 1.0,
    ".*wrist_y_joint": 1.0,
    ".*wrist_z_joint": 1.0,
}

#: Rotor inertia per joint, taken from the vendor MuJoCo model. Leaving these
#: at zero is a stiffness-stability risk at kp=300..500 with dt=1/200.
#: Split per actuator group: Isaac requires every pattern in a group's dict to
#: match at least one joint *in that group*.
LEG_ARMATURE = {
    ".*hip_y_joint": 0.044688,
    ".*hip_x_joint": 0.044688,
    ".*hip_z_joint": 0.0137351,
    ".*knee_y_joint": 0.044688,
    ".*ankle_y_joint": 0.00848397,
    ".*ankle_x_joint": 0.00551458,
}

UPPER_BODY_ARMATURE = {
    "waist_y_joint": 0.0274702,
    "waist_z_joint": 0.0412054,
    ".*shoulder_y_joint": 0.0137351,
    ".*shoulder_x_joint": 0.0137351,
    ".*shoulder_z_joint": 0.00551458,
    ".*elbow_y_joint": 0.0137351,
    ".*wrist_x_joint": 0.00551458,
    ".*wrist_y_joint": 0.00551458,
    ".*wrist_z_joint": 0.00551458,
}

ARMATURE = UPPER_BODY_ARMATURE | LEG_ARMATURE

#: Torque limits, transcribed from the URDF's `<limit effort=...>`.
#: The URDF converter does not carry these into the USD drives (Isaac reports
#: 1e9 instead), so they must be declared here or the legs get unbounded
#: torque authority and the hand-computed PD clamp becomes meaningless.
LEG_EFFORT_LIMIT = {
    ".*hip_y_joint": 150.0,
    ".*hip_x_joint": 150.0,
    ".*hip_z_joint": 45.0,
    ".*knee_y_joint": 150.0,
    ".*ankle_y_joint": 50.0,
    ".*ankle_x_joint": 20.0,
}

UPPER_BODY_EFFORT_LIMIT = {
    "waist_y_joint": 90.0,
    "waist_z_joint": 150.0,
    ".*shoulder_y_joint": 45.0,
    ".*shoulder_x_joint": 45.0,
    ".*shoulder_z_joint": 21.0,
    ".*elbow_y_joint": 45.0,
    ".*wrist_x_joint": 21.0,
    ".*wrist_y_joint": 21.0,
    ".*wrist_z_joint": 21.0,
}

EFFORT_LIMIT = UPPER_BODY_EFFORT_LIMIT | LEG_EFFORT_LIMIT

#: Every ELF3 joint shares the same velocity limit in the URDF.
VELOCITY_LIMIT = 20.0

#: Links with no `<inertial>` in the URDF (IMU frame, lidar and two cameras).
#: PhysX gives geometry-less links a default 1 kg each, which silently added
#: 4 kg -- including at imu_link, the observation reference frame.
MASSLESS_LINK_NAMES = (
    "imu_link",
    "mid360_link",
    "d435i_link",
    "torso_front_bottom_d435i_link",
)
MASSLESS_LINK_MASS = 1e-6

ACTION_SCALE = 0.25
SIM_DT = 1.0 / 200.0
DECIMATION = 4
EPISODE_LENGTH_S = 20.0


def policy_dt() -> float:
    """Control period seen by the policy; the sim2sim loop must match it."""
    return SIM_DT * DECIMATION


# --------------------------------------------------------------------------
# Observation scales
# --------------------------------------------------------------------------
LIN_VEL_SCALE = 2.0
ANG_VEL_SCALE = 0.5
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05

#: Command scaling applied to (vx, vy, wyaw). The height command is fed in
#: absolute metres and is deliberately unscaled.
COMMAND_SCALE = (LIN_VEL_SCALE, LIN_VEL_SCALE, ANG_VEL_SCALE)


# --------------------------------------------------------------------------
# Left/right mirror spec
# --------------------------------------------------------------------------
# Derived from the joint names rather than transcribed, so it cannot drift out
# of sync with the joint table. Upstream hardcodes ~200 lines of G1 indices;
# doing that here would silently corrupt training on any joint reordering.


def _mirror_joint_name(name: str) -> str:
    if name.startswith("l_"):
        return "r_" + name[2:]
    if name.startswith("r_"):
        return "l_" + name[2:]
    return name


def _mirror_sign(name: str) -> int:
    """Roll and yaw axes flip under mirroring; pitch is unchanged."""
    return -1 if name.endswith(("_x_joint", "_z_joint")) else 1


def _build_mirror(names: list[str]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    lookup = {name: index for index, name in enumerate(names)}
    indices = tuple(lookup[_mirror_joint_name(name)] for name in names)
    signs = tuple(_mirror_sign(name) for name in names)
    return indices, signs


DOF_MIRROR_INDICES, DOF_MIRROR_SIGNS = _build_mirror(JOINT_NAMES)
ACTION_MIRROR_INDICES, ACTION_MIRROR_SIGNS = _build_mirror(POLICY_JOINT_NAMES)

#: Sign flips for the observation head: commands (vx, vy, wyaw, height),
#: then IMU angular velocity (xyz), then projected gravity (xyz).
#: Mirroring maps y -> -y, so true vectors use diag(1, -1, 1) and the angular
#: velocity pseudovector uses diag(-1, 1, -1).
OBS_HEAD_MIRROR_SIGNS = (1, -1, -1, 1, -1, 1, -1, 1, -1, 1)

#: Sign flips for the critic's appended base linear velocity (xyz).
CRITIC_TAIL_MIRROR_SIGNS = (1, -1, 1)


# --------------------------------------------------------------------------
# sim2sim bridge
# --------------------------------------------------------------------------
# The vendor MuJoCo model exposes 31 actuators: the 28 trained joints plus
# waist roll and two head joints. Those three must be held at zero, and the
# remaining 28 must be mapped *by name* -- the MJCF ordering differs from the
# canonical ordering (waist_x sits between waist_y and waist_z).
MUJOCO_LOCKED_JOINTS = ("waist_x_joint", "head_z_joint", "head_y_joint")
