from dataclasses import dataclass
import math
from typing import Sequence, Tuple, Union

import torch


Device = Union[str, torch.device]


@dataclass(frozen=True)
class HeightCurriculumResult:
    ratio: float
    qualified_windows: int
    ready: int
    survival_rate: float
    mean_height_error: float
    qualified: bool


def squat_toe_out_ratio(
    heights: torch.Tensor,
    start_height: float,
    full_angle_height: float,
) -> torch.Tensor:
    """计算从双脚朝前到完整外八角的线性下蹲比例。"""
    if full_angle_height >= start_height:
        raise ValueError("full-angle height must be lower than start height")
    return torch.clamp(
        (float(start_height) - heights)
        / (float(start_height) - float(full_angle_height)),
        0.0,
        1.0,
    )


def squat_toe_out_targets(
    heights: torch.Tensor,
    start_height: float,
    full_angle_height: float,
    maximum_angle_degrees: float,
) -> torch.Tensor:
    """返回机身坐标系下左正右负的足端偏航目标，单位为弧度。"""
    if maximum_angle_degrees < 0.0:
        raise ValueError("maximum toe-out angle must be non-negative")
    ratio = squat_toe_out_ratio(heights, start_height, full_angle_height)
    angle = ratio * math.radians(float(maximum_angle_degrees))
    return torch.stack((angle, -angle), dim=-1)


def bounded_lateral_distance_reward(
    distances: torch.Tensor,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    """在指定区间内返回零，越界时返回线性负奖励。"""
    if minimum > maximum:
        raise ValueError("minimum lateral distance must not exceed maximum")
    return -torch.relu(float(minimum) - distances) - torch.relu(
        distances - float(maximum)
    )


def sample_height_targets(
    count: int,
    minimum: float,
    maximum: float,
    endpoint_probability: float,
    device: Device,
) -> torch.Tensor:
    """按两端各 endpoint_probability、其余均匀分布采样高度目标。"""
    if minimum > maximum:
        raise ValueError("minimum height must not exceed maximum height")
    if endpoint_probability < 0.0 or endpoint_probability > 0.5:
        raise ValueError("endpoint_probability must be in [0.0, 0.5]")

    selector = torch.rand(count, device=device)
    targets = minimum + (maximum - minimum) * torch.rand(count, device=device)
    targets = torch.where(selector < endpoint_probability, minimum, targets)
    targets = torch.where(selector >= 1.0 - endpoint_probability, maximum, targets)
    return targets


def slew_height_commands(
    current: torch.Tensor,
    target: torch.Tensor,
    slew_rate: float,
    dt: float,
) -> torch.Tensor:
    """以不超过 slew_rate 的速度将当前高度命令逼近目标。"""
    max_delta = max(float(slew_rate) * float(dt), 0.0)
    return current + torch.clamp(target - current, min=-max_delta, max=max_delta)


def sample_walk_commands(
    count: int,
    x_range: Sequence[float],
    y_range: Sequence[float],
    yaw_range: Sequence[float],
    moving_probability: float,
    device: Device,
) -> torch.Tensor:
    """采样全向速度命令，并按概率将整组命令置为原地站立。"""
    if moving_probability < 0.0 or moving_probability > 1.0:
        raise ValueError("moving_probability must be in [0.0, 1.0]")

    ranges = (x_range, y_range, yaw_range)
    commands = torch.empty((count, 3), device=device)
    for index, value_range in enumerate(ranges):
        commands[:, index].uniform_(float(value_range[0]), float(value_range[1]))
    moving = (torch.rand(count, device=device) < moving_probability).unsqueeze(1)
    return commands * moving


def sample_squat_modes(
    count: int,
    squat_probability: float,
    device: Device,
) -> torch.Tensor:
    """按概率选择第二阶段中执行零速度下蹲任务的环境。"""
    if squat_probability < 0.0 or squat_probability > 1.0:
        raise ValueError("squat_probability must be in [0.0, 1.0]")
    return torch.rand(count, device=device) < squat_probability


def sample_stage2_commands(
    squat_mask: torch.Tensor,
    x_range: Sequence[float],
    y_range: Sequence[float],
    yaw_range: Sequence[float],
    moving_probability: float,
    height_minimum: float,
    height_maximum: float,
    height_endpoint_probability: float,
    standing_height: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """为固定任务分组重采样第二阶段速度命令与高度目标。"""
    if squat_mask.ndim != 1 or squat_mask.dtype != torch.bool:
        raise ValueError("squat_mask must be a one-dimensional boolean tensor")

    count = squat_mask.numel()
    device = squat_mask.device
    velocity_commands = sample_walk_commands(
        count,
        x_range,
        y_range,
        yaw_range,
        moving_probability,
        device,
    )
    velocity_commands[squat_mask] = 0.0

    sampled_heights = sample_height_targets(
        count,
        height_minimum,
        height_maximum,
        height_endpoint_probability,
        device,
    )
    standing_heights = torch.full_like(sampled_heights, float(standing_height))
    height_targets = torch.where(squat_mask, sampled_heights, standing_heights)
    return velocity_commands, height_targets


def upper_height_amplification(
    heights: torch.Tensor,
    minimum: float,
    maximum: float,
    maximum_amplification: float,
) -> torch.Tensor:
    """将站立高度的 1 倍扰动线性放大到最低高度的指定倍率。"""
    span = max(float(maximum) - float(minimum), 1.0e-6)
    squat_ratio = torch.clamp((float(maximum) - heights) / span, 0.0, 1.0)
    return 1.0 + squat_ratio * (float(maximum_amplification) - 1.0)


def evaluate_height_curriculum(
    ratio: float,
    successful_episodes: int,
    completed_episodes: int,
    mean_height_error: float,
    qualified_windows: int,
    ratio_step: float,
    survival_threshold: float,
    height_error_threshold: float,
    ready_windows: int,
) -> HeightCurriculumResult:
    """根据一个完整统计窗的存活率和高度误差更新第一阶段课程。"""
    survival_rate = (
        float(successful_episodes) / float(completed_episodes)
        if completed_episodes > 0
        else 0.0
    )
    qualified = (
        completed_episodes > 0
        and survival_rate >= survival_threshold
        and mean_height_error <= height_error_threshold
    )

    new_ratio = float(ratio)
    if qualified:
        new_ratio = min(1.0, round(new_ratio + float(ratio_step), 10))

    if qualified and new_ratio >= 1.0:
        new_qualified_windows = int(qualified_windows) + 1
    else:
        new_qualified_windows = 0
    ready = int(new_qualified_windows >= int(ready_windows))

    return HeightCurriculumResult(
        ratio=new_ratio,
        qualified_windows=new_qualified_windows,
        ready=ready,
        survival_rate=survival_rate,
        mean_height_error=float(mean_height_error),
        qualified=qualified,
    )
