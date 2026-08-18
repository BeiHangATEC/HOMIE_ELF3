"""Pure-tensor formulas for the 33 ELF3 HOMIE reward terms.

Every function here takes plain tensors and returns a per-environment tensor.
Nothing imports Isaac Lab, so the whole reward surface is unit-testable without
a GPU -- which is how the scale table and the mode shaping stay trustworthy.

The environment is responsible for gathering the inputs (contact forces, body
poses, virtual sole points) and for multiplying by scale * dt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


REWARD_NAMES = (
    "tracking_x_vel",
    "tracking_y_vel",
    "tracking_ang_vel",
    "tracking_base_height",
    "lin_vel_z",
    "ang_vel_xy",
    "orientation",
    "action_rate",
    "deviation_hip_joint",
    "deviation_ankle_joint",
    "deviation_knee_joint",
    "dof_acc",
    "dof_pos_limits",
    "feet_air_time",
    "feet_clearance",
    "feet_distance_lateral",
    "knee_distance_lateral",
    "feet_ground_parallel",
    "feet_parallel",
    "smoothness",
    "joint_power",
    "feet_stumble",
    "torques",
    "dof_vel",
    "dof_vel_limits",
    "torque_limits",
    "no_fly",
    "joint_tracking_error",
    "feet_slip",
    "feet_contact_forces",
    "contact_momentum",
    "action_vanish",
    "stand_still",
)

REWARD_SCALES = {
    "tracking_x_vel": 1.5,
    "tracking_y_vel": 1.0,
    "tracking_ang_vel": 2.0,
    "tracking_base_height": 2.0,
    "lin_vel_z": -0.5,
    "ang_vel_xy": -0.025,
    "orientation": -1.5,
    "action_rate": -0.01,
    "deviation_hip_joint": -0.2,
    "deviation_ankle_joint": -0.5,
    "deviation_knee_joint": -0.75,
    "dof_acc": -2.5e-7,
    "dof_pos_limits": -2.0,
    "feet_air_time": 0.05,
    "feet_clearance": -0.25,
    "feet_distance_lateral": 0.5,
    "knee_distance_lateral": 1.0,
    "feet_ground_parallel": -0.05,
    "feet_parallel": -0.075,
    "smoothness": -0.05,
    "joint_power": -2e-5,
    "feet_stumble": -1.5,
    "torques": -2.5e-6,
    "dof_vel": -1e-4,
    "dof_vel_limits": -2e-3,
    "torque_limits": -0.1,
    "no_fly": 0.75,
    "joint_tracking_error": -0.1,
    "feet_slip": -0.25,
    "feet_contact_forces": -0.00025,
    "contact_momentum": 2.5e-4,
    "action_vanish": -1.0,
    "stand_still": -0.15,
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _indices(indices: torch.Tensor | Sequence[int], like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(indices, dtype=torch.long, device=like.device)


def _select(tensor: torch.Tensor, indices: torch.Tensor | Sequence[int]) -> torch.Tensor:
    return tensor.index_select(-1, _indices(indices, tensor))


def _crouching(modes: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    """1.0 where the commanded mode is a crouch, else 0.0."""
    if modes is None:
        return torch.zeros_like(like)
    return (modes >= 2).to(like.dtype)


def _mode_shape(raw: torch.Tensor, modes: torch.Tensor | None) -> torch.Tensor:
    """Offset velocity tracking down by 1 while crouching.

    Crouch episodes command zero velocity, so the exponential would otherwise
    pay full reward for standing still and swamp the height term.
    """
    return raw - _crouching(modes, raw)


# --------------------------------------------------------------------------
# Command tracking
# --------------------------------------------------------------------------
def tracking_x_vel(
    commands: torch.Tensor,
    base_lin_vel: torch.Tensor,
    tracking_sigma: float,
    modes: torch.Tensor | None = None,
) -> torch.Tensor:
    raw = torch.exp(
        -torch.square(commands[:, 0] - base_lin_vel[:, 0]) / tracking_sigma
    )
    return _mode_shape(raw, modes)


def tracking_y_vel(
    commands: torch.Tensor,
    base_lin_vel: torch.Tensor,
    tracking_sigma: float,
    modes: torch.Tensor | None = None,
) -> torch.Tensor:
    raw = torch.exp(
        -torch.square(commands[:, 1] - base_lin_vel[:, 1]) / tracking_sigma
    )
    return _mode_shape(raw, modes)


def tracking_ang_vel(
    commands: torch.Tensor,
    base_ang_vel: torch.Tensor,
    tracking_sigma: float,
    modes: torch.Tensor | None = None,
) -> torch.Tensor:
    raw = torch.exp(
        -torch.square(commands[:, 2] - base_ang_vel[:, 2]) / tracking_sigma
    )
    return _mode_shape(raw, modes)


def tracking_base_height(
    *,
    root_height: torch.Tensor,
    feet_height: torch.Tensor,
    commanded_height: torch.Tensor,
    ankle_sole_distance: float,
    modes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Track commanded torso-above-soles height.

    `max` over the two feet rather than `min` so a lifted foot cannot make the
    robot look taller than it is.
    """
    base_height = torch.maximum(
        root_height - feet_height[:, 0], root_height - feet_height[:, 1]
    )
    raw = torch.exp(
        -4.0 * torch.abs(base_height - commanded_height + ankle_sole_distance)
    )
    # Crouching is the harder skill, so it is worth double while crouching.
    return raw * (1.0 + _crouching(modes, raw))


# --------------------------------------------------------------------------
# Base stability
# --------------------------------------------------------------------------
def lin_vel_z(
    base_lin_vel: torch.Tensor, high_command_mask: torch.Tensor
) -> torch.Tensor:
    """Penalize bobbing, but only while commanded to stand tall.

    Squatting requires vertical motion, so this is gated off during crouches.
    """
    return torch.square(base_lin_vel[:, 2]) * high_command_mask.to(
        base_lin_vel.dtype
    )


def ang_vel_xy(base_ang_vel: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(base_ang_vel[:, :2]), dim=-1)


def orientation(projected_gravity: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(projected_gravity[:, :2]), dim=-1)


# --------------------------------------------------------------------------
# Action regularity
# --------------------------------------------------------------------------
def action_rate(
    last_policy_actions: torch.Tensor, policy_actions: torch.Tensor
) -> torch.Tensor:
    return torch.sum(torch.square(last_policy_actions - policy_actions), dim=-1)


def smoothness(
    policy_actions: torch.Tensor,
    last_policy_actions: torch.Tensor,
    second_last_policy_actions: torch.Tensor,
) -> torch.Tensor:
    """Second difference of the action sequence: penalizes jerk."""
    return torch.sum(
        torch.square(
            policy_actions - 2.0 * last_policy_actions + second_last_policy_actions
        ),
        dim=-1,
    )


def action_vanish(
    raw_policy_actions: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
) -> torch.Tensor:
    """Penalize action the joint limits would clip away.

    Without this the policy can emit ever-larger actions that the clamp
    discards, decoupling its output from what the robot actually does.
    """
    below = (action_min - raw_policy_actions).clamp(min=0.0)
    above = (raw_policy_actions - action_max).clamp(min=0.0)
    return torch.sum(below + above, dim=-1)


# --------------------------------------------------------------------------
# Joint regularization
# --------------------------------------------------------------------------
def _joint_deviation(
    dof_pos: torch.Tensor,
    default_dof_pos: torch.Tensor,
    joint_indices: Sequence[int],
    mask: torch.Tensor | None,
) -> torch.Tensor:
    deviation = _select(dof_pos - default_dof_pos, joint_indices)
    raw = torch.sum(torch.square(deviation), dim=-1)
    return raw if mask is None else raw * mask.to(raw.dtype)


def deviation_hip_joint(
    dof_pos: torch.Tensor,
    default_dof_pos: torch.Tensor,
    hip_joint_indices: Sequence[int],
    high_command_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return _joint_deviation(
        dof_pos, default_dof_pos, hip_joint_indices, high_command_mask
    )


def deviation_ankle_joint(
    dof_pos: torch.Tensor,
    default_dof_pos: torch.Tensor,
    ankle_roll_joint_indices: Sequence[int],
    high_command_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return _joint_deviation(
        dof_pos, default_dof_pos, ankle_roll_joint_indices, high_command_mask
    )


def deviation_knee_joint(
    *,
    dof_pos: torch.Tensor,
    joint_lower_limits: torch.Tensor,
    joint_upper_limits: torch.Tensor,
    knee_joint_indices: Sequence[int],
    root_height: torch.Tensor,
    commanded_height: torch.Tensor,
) -> torch.Tensor:
    """Couple knee bend to height error, so squatting uses the knees."""
    knee_pos = _select(dof_pos, knee_joint_indices)
    lower = _select(joint_lower_limits, knee_joint_indices)
    upper = _select(joint_upper_limits, knee_joint_indices)
    normalized = (knee_pos - lower) / (upper - lower).clamp(min=1e-6)
    height_error = (root_height - commanded_height).unsqueeze(-1)
    return torch.sum(torch.abs((normalized - 0.5) * height_error), dim=-1)


def dof_acc(
    dof_vel: torch.Tensor, last_dof_vel: torch.Tensor, step_dt: float
) -> torch.Tensor:
    return torch.sum(torch.square((last_dof_vel - dof_vel) / step_dt), dim=-1)


def dof_vel(dof_vel_tensor: torch.Tensor, leg_indices: Sequence[int]) -> torch.Tensor:
    return torch.sum(torch.square(_select(dof_vel_tensor, leg_indices)), dim=-1)


def dof_pos_limits(
    dof_pos: torch.Tensor,
    soft_lower_limits: torch.Tensor,
    soft_upper_limits: torch.Tensor,
) -> torch.Tensor:
    below = (soft_lower_limits - dof_pos).clamp(min=0.0)
    above = (dof_pos - soft_upper_limits).clamp(min=0.0)
    return torch.sum(below + above, dim=-1)


def dof_vel_limits(
    dof_vel_tensor: torch.Tensor,
    velocity_limits: torch.Tensor,
    soft_ratio: float,
    leg_indices: Sequence[int],
) -> torch.Tensor:
    excess = _select(dof_vel_tensor, leg_indices).abs() - soft_ratio * _select(
        velocity_limits, leg_indices
    )
    return torch.sum(excess.clamp(min=0.0), dim=-1)


def joint_tracking_error(
    joint_pos_target: torch.Tensor,
    dof_pos: torch.Tensor,
    leg_indices: Sequence[int],
) -> torch.Tensor:
    error = _select(joint_pos_target, leg_indices) - _select(dof_pos, leg_indices)
    return torch.sum(torch.square(error), dim=-1)


# --------------------------------------------------------------------------
# Torque and power
# --------------------------------------------------------------------------
def torques(
    applied_torque: torch.Tensor,
    stiffness: torch.Tensor,
    leg_indices: Sequence[int],
) -> torch.Tensor:
    """Normalized by stiffness so stiff and soft joints contribute evenly."""
    normalized = _select(applied_torque, leg_indices) / _select(
        stiffness, leg_indices
    ).clamp(min=1e-6)
    return torch.sum(torch.square(normalized), dim=-1)


def torque_limits(
    applied_torque: torch.Tensor,
    effort_limits: torch.Tensor,
    soft_ratio: float,
    leg_indices: Sequence[int],
) -> torch.Tensor:
    excess = _select(applied_torque, leg_indices).abs() - soft_ratio * _select(
        effort_limits, leg_indices
    )
    return torch.sum(excess.clamp(min=0.0), dim=-1)


def joint_power(
    dof_vel_tensor: torch.Tensor,
    applied_torque: torch.Tensor,
    commands: torch.Tensor,
) -> torch.Tensor:
    """Mechanical power, discounted by how much motion was asked for.

    Dividing by the command magnitude keeps the penalty from punishing the
    effort needed to actually walk fast.
    """
    power = torch.sum(torch.abs(dof_vel_tensor * applied_torque), dim=-1)
    demand = torch.sum(torch.square(commands[:, :2]), dim=-1) + 0.2 * torch.square(
        commands[:, 2]
    )
    return power / demand.clamp(min=0.1)


# --------------------------------------------------------------------------
# Foot contact and gait
# --------------------------------------------------------------------------
def feet_air_time(
    air_time: torch.Tensor,
    first_contact: torch.Tensor,
    command_norm: torch.Tensor,
    target_air_time: float = 0.5,
) -> torch.Tensor:
    """Reward steps that spend about `target_air_time` in the air.

    Zeroed when no motion is commanded, so standing still is not rewarded for
    lifting a foot.
    """
    reward = torch.sum(
        (air_time - target_air_time) * first_contact.to(air_time.dtype), dim=-1
    )
    return reward * (command_norm > 0.1).to(reward.dtype)


def no_fly(contact_forces_z: torch.Tensor, stand_mask: torch.Tensor) -> torch.Tensor:
    """Reward single-support, i.e. a proper gait rather than hopping."""
    in_contact = contact_forces_z > 0.5
    single = (torch.sum(in_contact.to(torch.int32), dim=-1) == 1).to(
        contact_forces_z.dtype
    )
    # While standing there is nothing to alternate, so pay it in full.
    return torch.maximum(single, stand_mask.to(single.dtype))


def stand_still(
    contact_forces_z: torch.Tensor, high_command_mask: torch.Tensor
) -> torch.Tensor:
    """Penalize lifting a foot while commanded to stand tall and still."""
    airborne = (contact_forces_z < 0.1).to(contact_forces_z.dtype)
    return torch.sum(airborne, dim=-1) * high_command_mask.to(airborne.dtype)


def feet_stumble(
    contact_forces: torch.Tensor, lateral_ratio: float = 3.0
) -> torch.Tensor:
    """Detect shear-dominated contact, i.e. stubbing a foot into something."""
    lateral = torch.linalg.vector_norm(contact_forces[..., :2], dim=-1)
    vertical = contact_forces[..., 2].abs()
    return torch.any(lateral > lateral_ratio * vertical, dim=-1).to(
        contact_forces.dtype
    )


def feet_contact_forces(
    contact_forces: torch.Tensor, max_contact_force: float
) -> torch.Tensor:
    magnitude = torch.linalg.vector_norm(contact_forces, dim=-1)
    return torch.sum((magnitude - max_contact_force).clamp(min=0.0), dim=-1)


def feet_slip(
    feet_vel_xy: torch.Tensor, contact_forces_z: torch.Tensor
) -> torch.Tensor:
    speed = torch.linalg.vector_norm(feet_vel_xy, dim=-1)
    return torch.sum(speed * (contact_forces_z > 1.0).to(speed.dtype), dim=-1)


def contact_momentum(
    feet_vel_z: torch.Tensor, contact_forces_z: torch.Tensor
) -> torch.Tensor:
    """Reward soft landings: downward speed paired with low contact force."""
    descending = feet_vel_z.clamp(max=0.0)
    excess_force = (contact_forces_z - 50.0).clamp(min=0.0)
    return torch.sum(descending * excess_force, dim=-1)


def feet_clearance(
    feet_height: torch.Tensor,
    feet_vel_xy_b: torch.Tensor,
    clearance_height_target: float,
    high_command_mask: torch.Tensor,
) -> torch.Tensor:
    """Encourage lifting the swing foot to a target height while moving."""
    error = torch.square(feet_height - clearance_height_target)
    speed = torch.linalg.vector_norm(feet_vel_xy_b, dim=-1)
    return torch.sum(error * speed, dim=-1) * high_command_mask.to(error.dtype)


# --------------------------------------------------------------------------
# Stance geometry
# --------------------------------------------------------------------------
def feet_distance_lateral(
    lateral_distance: torch.Tensor, least: float, most: float
) -> torch.Tensor:
    """Reward keeping the feet within a sane lateral band.

    Too narrow and the robot trips over itself; too wide and it waddles.
    """
    return (
        (lateral_distance > least).to(lateral_distance.dtype)
        * (lateral_distance < most).to(lateral_distance.dtype)
    )


def knee_distance_lateral(
    lateral_distance: torch.Tensor, least: float, most: float
) -> torch.Tensor:
    """Same band test as the feet, at twice the width (two knee pairs)."""
    return feet_distance_lateral(lateral_distance, 2.0 * least, 2.0 * most)


def feet_ground_parallel(
    sole_height_variance: torch.Tensor, contact_filter: torch.Tensor
) -> torch.Tensor:
    """Penalize a tilted sole while it is bearing load."""
    return torch.sum(
        sole_height_variance * contact_filter.to(sole_height_variance.dtype), dim=-1
    )


def feet_parallel(sole_point_distances: torch.Tensor) -> torch.Tensor:
    """Penalize the two soles being non-parallel to each other.

    ELF3 has no per-foot contact marker links, so the environment supplies
    corner-to-corner distances from a virtual rectangular sole.
    """
    return torch.var(sole_point_distances, dim=-1)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def scale_reward_terms(
    raw_terms: Mapping[str, torch.Tensor],
    scales: Mapping[str, float],
    step_dt: float,
) -> dict[str, torch.Tensor]:
    """Apply `scale * dt` to every term, requiring an exact name match.

    Both mappings must cover exactly REWARD_NAMES. Adding a term is therefore
    a deliberate two-place edit rather than something that silently no-ops.
    """
    expected = set(REWARD_NAMES)
    missing_raw = expected - set(raw_terms)
    extra_raw = set(raw_terms) - expected
    if missing_raw or extra_raw:
        raise ValueError(
            f"reward terms mismatch: missing {sorted(missing_raw)}, "
            f"unexpected {sorted(extra_raw)}"
        )
    missing_scale = expected - set(scales)
    extra_scale = set(scales) - expected
    if missing_scale or extra_scale:
        raise ValueError(
            f"reward scales mismatch: missing {sorted(missing_scale)}, "
            f"unexpected {sorted(extra_scale)}"
        )
    return {name: raw_terms[name] * scales[name] * step_dt for name in REWARD_NAMES}


def sum_reward_terms(scaled_terms: Mapping[str, torch.Tensor]) -> torch.Tensor:
    total = None
    for name in REWARD_NAMES:
        term = scaled_terms[name]
        total = term.clone() if total is None else total + term
    if total is None:
        raise ValueError("no reward terms to sum")
    return total
