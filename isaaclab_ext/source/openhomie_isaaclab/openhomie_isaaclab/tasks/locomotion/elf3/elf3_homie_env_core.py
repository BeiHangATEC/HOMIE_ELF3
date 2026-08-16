"""Isaac-independent tensor operations for the ELF3 direct environment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from openhomie_isaaclab import elf3_constants as C


def build_name_permutation(runtime_names: Sequence[str], canonical_names: Sequence[str]) -> torch.Tensor:
    """Return canonical-to-runtime indices after validating a bijection."""
    runtime = list(runtime_names)
    canonical = list(canonical_names)
    if len(runtime) != len(set(runtime)) or len(canonical) != len(set(canonical)):
        raise ValueError("joint name lists must not contain duplicates")
    if len(runtime) != len(canonical) or set(runtime) != set(canonical):
        raise ValueError("runtime and canonical joint names must form the same set")
    lookup = {name: index for index, name in enumerate(runtime)}
    return torch.tensor([lookup[name] for name in canonical], dtype=torch.long)


def gather_canonical(runtime_values: torch.Tensor, canonical_to_runtime: torch.Tensor) -> torch.Tensor:
    indices = canonical_to_runtime.to(device=runtime_values.device, dtype=torch.long)
    if runtime_values.shape[-1] <= int(indices.max().item()):
        raise ValueError("permutation exceeds runtime tensor width")
    return runtime_values.index_select(-1, indices)


def scatter_canonical(
    canonical_values: torch.Tensor,
    canonical_to_runtime: torch.Tensor,
    runtime_width: int,
    base: torch.Tensor | None = None,
) -> torch.Tensor:
    indices = canonical_to_runtime.to(device=canonical_values.device, dtype=torch.long)
    if canonical_values.shape[-1] != indices.numel():
        raise ValueError("canonical tensor width must match permutation")
    if runtime_width <= 0 or torch.any((indices < 0) | (indices >= runtime_width)):
        raise ValueError("invalid runtime width or permutation")
    shape = (*canonical_values.shape[:-1], runtime_width)
    if base is None:
        result = canonical_values.new_zeros(shape)
    else:
        if base.shape != shape:
            raise ValueError("base tensor shape does not match runtime shape")
        result = base.clone()
    result.index_copy_(-1, indices, canonical_values)
    return result


def assemble_actor_frame(
    commands: torch.Tensor,
    imu_ang_vel_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    dof_pos: torch.Tensor,
    default_dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    previous_action: torch.Tensor,
) -> torch.Tensor:
    batch = commands.shape[0]
    expected = {
        "commands": (batch, C.NUM_COMMAND_OBS),
        "imu angular velocity": (batch, 3),
        "projected gravity": (batch, 3),
        "joint position": (batch, C.NUM_ROBOT_DOFS),
        "default joint position": (batch, C.NUM_ROBOT_DOFS),
        "joint velocity": (batch, C.NUM_ROBOT_DOFS),
        "previous action": (batch, C.NUM_POLICY_ACTIONS),
    }
    values = (commands, imu_ang_vel_b, projected_gravity_b, dof_pos, default_dof_pos, dof_vel, previous_action)
    for (name, shape), value in zip(expected.items(), values):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    command_scale = commands.new_tensor((*C.COMMAND_SCALE, 1.0))
    frame = torch.cat(
        (
            commands * command_scale,
            imu_ang_vel_b * C.ANG_VEL_SCALE,
            projected_gravity_b,
            (dof_pos - default_dof_pos) * C.DOF_POS_SCALE,
            dof_vel * C.DOF_VEL_SCALE,
            previous_action,
        ),
        dim=-1,
    )
    if frame.shape[-1] != C.num_one_step_actor_obs():
        raise RuntimeError("ELF3 actor frame cursor disagrees with canonical dimensions")
    return frame


def shift_history(history: torch.Tensor, frame: torch.Tensor, reset_env_ids: torch.Tensor | None = None) -> torch.Tensor:
    if history.ndim != 3 or history.shape[1:] != (C.NUM_ACTOR_HISTORY, C.num_one_step_actor_obs()):
        raise ValueError("history has an invalid shape")
    if frame.shape != (history.shape[0], C.num_one_step_actor_obs()):
        raise ValueError("frame has an invalid shape")
    result = torch.roll(history, shifts=-1, dims=1)
    if reset_env_ids is not None:
        ids = reset_env_ids.to(device=history.device, dtype=torch.long)
        result[ids] = 0.0
    result[:, -1] = frame
    return result


def compute_leg_efforts(
    desired_pos: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    *,
    kp: torch.Tensor,
    kd: torch.Tensor,
    kp_factors: torch.Tensor,
    kd_factors: torch.Tensor,
    actuation_offset: torch.Tensor,
    effort_limits: torch.Tensor,
) -> torch.Tensor:
    if desired_pos.shape != joint_pos.shape or desired_pos.shape != joint_vel.shape:
        raise ValueError("desired position, position, and velocity shapes must match")
    effort = kp * kp_factors * (desired_pos - joint_pos) - kd * kd_factors * joint_vel + actuation_offset
    limits = effort_limits.expand_as(effort)
    if not torch.all(torch.isfinite(limits) & (limits > 0)):
        raise ValueError("effort limits must be finite and positive")
    return torch.clamp(effort, -limits, limits)


def apply_control_randomization(
    kp_factors: torch.Tensor,
    kd_factors: torch.Tensor,
    actuation_offset: torch.Tensor,
    env_ids: torch.Tensor,
    *,
    draws: Mapping[str, torch.Tensor],
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
    offset_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if kp_factors.shape != kd_factors.shape or kp_factors.shape != actuation_offset.shape:
        raise ValueError("control randomization tensors must have matching shapes")
    ids = env_ids.to(device=kp_factors.device, dtype=torch.long)
    shape = (ids.numel(), kp_factors.shape[1])
    outputs = []
    for name, target, bounds in (
        ("kp", kp_factors, kp_range),
        ("kd", kd_factors, kd_range),
        ("offset", actuation_offset, offset_range),
    ):
        draw = draws[name].to(device=target.device, dtype=target.dtype)
        if draw.shape != shape or torch.any((draw < 0) | (draw > 1)):
            raise ValueError(f"{name} draws must have shape {shape} and lie in [0, 1]")
        low, high = bounds
        if low > high:
            raise ValueError(f"{name} randomization range is reversed")
        target[ids] = low + draw * (high - low)
        outputs.append(target)
    return tuple(outputs)


def classify_dones(
    failure: torch.Tensor, episode_length: torch.Tensor, max_episode_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if failure.dtype != torch.bool or failure.shape != episode_length.shape:
        raise ValueError("failure must be a bool tensor matching episode lengths")
    if max_episode_length <= 0:
        raise ValueError("max episode length must be positive")
    terminated = failure
    truncated = (episode_length >= max_episode_length) & ~failure
    return terminated, truncated, truncated.clone()


def virtual_sole_corners(ankle_pos_w: torch.Tensor, ankle_quat_wxyz: torch.Tensor) -> torch.Tensor:
    if ankle_pos_w.ndim != 3 or ankle_pos_w.shape[1:] != (2, 3):
        raise ValueError("ankle positions must have shape [num_envs, 2, 3]")
    if ankle_quat_wxyz.shape != (*ankle_pos_w.shape[:-1], 4):
        raise ValueError("ankle quaternions must have shape [num_envs, 2, 4]")
    cx, cy, cz = C.SOLE_CENTER_OFFSET
    local = ankle_pos_w.new_tensor(
        [
            [cx + C.SOLE_LENGTH / 2, cy + C.SOLE_WIDTH / 2, cz],
            [cx + C.SOLE_LENGTH / 2, cy - C.SOLE_WIDTH / 2, cz],
            [cx - C.SOLE_LENGTH / 2, cy + C.SOLE_WIDTH / 2, cz],
            [cx - C.SOLE_LENGTH / 2, cy - C.SOLE_WIDTH / 2, cz],
        ]
    )
    vector = local.view(1, 1, 4, 3)
    w = ankle_quat_wxyz[..., :1].unsqueeze(-2)
    xyz = ankle_quat_wxyz[..., 1:].unsqueeze(-2)
    rotated = vector + 2.0 * (w * torch.cross(xyz, vector, dim=-1) + torch.cross(xyz, torch.cross(xyz, vector, dim=-1), dim=-1))
    return ankle_pos_w.unsqueeze(-2) + rotated
