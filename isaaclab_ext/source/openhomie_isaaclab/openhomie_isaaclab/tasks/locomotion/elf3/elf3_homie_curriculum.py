"""ELF3 command sampling and the two HOMIE curricula.

Simulator-independent: all randomness enters as explicit `draws` tensors, so
the sampling logic is testable without a GPU or a physics step.

Two separate curricula live here:

* the **command mode** mix (walk / stand tall / crouch), sampled per episode;
* the **upper-body amplitude** ramp, which slowly widens the arm and waist
  poses the legs must stay upright against.

The height *stage* ladder is a third curriculum, but it is advanced between
processes rather than in-process, so it lives in `elf3_stages.py`.
"""

from __future__ import annotations

import math
from numbers import Real

import torch


# Command modes. Ordering matters: `>= CROUCH_LOW` means "is crouching", which
# is what the reward shaping keys off.
WALK = 0
HIGH_STAND = 1
CROUCH_LOW = 2
CROUCH_FULL = 3

MODE_PROBABILITIES = (0.60, 0.15, 0.15, 0.10)

G1_SINGLE_STAGE_MODE_PROBABILITIES = {
    WALK: 0.50,
    HIGH_STAND: 1.0 / 6.0,
    CROUCH_FULL: 1.0 / 3.0,
}


def sample_modes(draws: torch.Tensor) -> torch.Tensor:
    """Map uniform draws in [0, 1) onto command modes."""
    if torch.any((draws < 0.0) | (draws >= 1.0)):
        raise ValueError("mode draws must lie in [0, 1)")
    cumulative = torch.tensor(
        [
            MODE_PROBABILITIES[0],
            MODE_PROBABILITIES[0] + MODE_PROBABILITIES[1],
            MODE_PROBABILITIES[0] + MODE_PROBABILITIES[1] + MODE_PROBABILITIES[2],
        ],
        dtype=draws.dtype,
        device=draws.device,
    )
    return torch.bucketize(draws, cumulative)


def walk_mask(modes: torch.Tensor) -> torch.Tensor:
    return modes == WALK


def high_mask(modes: torch.Tensor) -> torch.Tensor:
    """True where the robot is commanded to hold a tall pose.

    Gates the terms that only make sense while standing: vertical-velocity
    damping, hip and ankle deviation, foot clearance.
    """
    return modes <= HIGH_STAND


def stand_mask(modes: torch.Tensor) -> torch.Tensor:
    """True where no velocity is commanded."""
    return modes != WALK


def crouch_mask(modes: torch.Tensor) -> torch.Tensor:
    return modes >= CROUCH_LOW


def _uniform(draws: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + draws * (high - low)


def build_commands(
    *,
    modes: torch.Tensor,
    velocity_draws: torch.Tensor,
    height_draws: torch.Tensor,
    stage,
    crouch_min_height: float,
    crouch_focus_max_height: float,
) -> torch.Tensor:
    """Build the 5-wide command tensor [vx, vy, wyaw, unused, height].

    Only walking envs receive a velocity command; the rest are asked to hold
    position, so the height skill is learned without a conflicting objective.
    Column 3 is unused -- it exists so the layout matches the observation's
    4-channel command head plus legacy heading slot.
    """
    if velocity_draws.shape[-1] != 3:
        raise ValueError("velocity draws must have three columns")
    if modes.shape[0] != velocity_draws.shape[0] != height_draws.shape[0]:
        raise ValueError("draws must agree with the number of environments")

    num_envs = modes.shape[0]
    commands = torch.zeros(
        num_envs, 5, dtype=velocity_draws.dtype, device=velocity_draws.device
    )

    walking = walk_mask(modes).to(commands.dtype).unsqueeze(-1)
    velocities = torch.stack(
        (
            _uniform(velocity_draws[:, 0], *stage.lin_vel_x),
            _uniform(velocity_draws[:, 1], *stage.lin_vel_y),
            _uniform(velocity_draws[:, 2], *stage.ang_vel_yaw),
        ),
        dim=-1,
    )
    commands[:, :3] = velocities * walking

    # Heights: tall modes hold the stage height; crouch modes sample a band.
    low_max = min(crouch_focus_max_height, stage.walk_height)
    heights = torch.full_like(height_draws, stage.walk_height)
    low = crouch_mask(modes) & (modes == CROUCH_LOW)
    full = modes == CROUCH_FULL
    heights = torch.where(
        low, _uniform(height_draws, crouch_min_height, low_max), heights
    )
    heights = torch.where(
        full, _uniform(height_draws, crouch_min_height, stage.walk_height), heights
    )
    commands[:, 4] = heights
    return commands


def sample_g1_single_stage_modes(draws: torch.Tensor) -> torch.Tensor:
    """Sample the three command outcomes with G1's exact inequalities.

    G1 assigns height holds for draws below one third and velocity walks for
    draws strictly above one half. The remaining interval is a tall stand.
    Height holds use ``CROUCH_FULL`` so the existing ELF3 crouch reward
    shaping remains active without altering the reward surface.
    """
    if torch.any((draws < 0.0) | (draws >= 1.0)):
        raise ValueError("G1 single-stage mode draws must lie in [0, 1)")
    modes = torch.full_like(draws, HIGH_STAND, dtype=torch.long)
    modes[draws < (1.0 / 3.0)] = CROUCH_FULL
    modes[draws > 0.5] = WALK
    return modes


def _validate_range(name: str, values: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(values, tuple) or len(values) != 2:
        raise TypeError(f"{name} must be a two-element tuple")
    lower, upper = values
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise TypeError(f"{name} bounds must be real numbers")
    lower, upper = float(lower), float(upper)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError(f"{name} bounds must be finite and ordered")
    return lower, upper


def validate_g1_single_stage_spec(
    *,
    lin_vel_x_range: tuple[float, float],
    lin_vel_y_range: tuple[float, float],
    ang_vel_yaw_range: tuple[float, float],
    height_range: tuple[float, float],
    stand_height: float,
) -> None:
    """Validate the complete fixed envelope before a run starts."""
    _validate_range("linear x velocity", lin_vel_x_range)
    _validate_range("linear y velocity", lin_vel_y_range)
    _validate_range("yaw velocity", ang_vel_yaw_range)
    height_low, height_high = _validate_range("height", height_range)
    if isinstance(stand_height, bool) or not isinstance(stand_height, Real):
        raise TypeError("stand height must be a real number")
    stand_height = float(stand_height)
    if not math.isfinite(stand_height) or not height_low <= stand_height <= height_high:
        raise ValueError("stand height must lie inside the height range")


def build_g1_single_stage_commands(
    *,
    modes: torch.Tensor,
    velocity_draws: torch.Tensor,
    height_draws: torch.Tensor,
    lin_vel_x_range: tuple[float, float],
    lin_vel_y_range: tuple[float, float],
    ang_vel_yaw_range: tuple[float, float],
    height_range: tuple[float, float],
    stand_height: float,
) -> torch.Tensor:
    """Build a fixed-envelope G1-proportioned ELF3 command batch."""
    validate_g1_single_stage_spec(
        lin_vel_x_range=lin_vel_x_range,
        lin_vel_y_range=lin_vel_y_range,
        ang_vel_yaw_range=ang_vel_yaw_range,
        height_range=height_range,
        stand_height=stand_height,
    )
    if modes.ndim != 1:
        raise ValueError("G1 single-stage modes must be rank one")
    if velocity_draws.shape != (modes.shape[0], 3):
        raise ValueError("G1 single-stage velocity draws must have three columns")
    if height_draws.shape != (modes.shape[0],):
        raise ValueError("G1 single-stage height draws must have one value per mode")
    if torch.any((velocity_draws < 0.0) | (velocity_draws > 1.0)):
        raise ValueError("G1 single-stage velocity draws must lie in [0, 1]")
    if torch.any((height_draws < 0.0) | (height_draws > 1.0)):
        raise ValueError("G1 single-stage height draws must lie in [0, 1]")
    supported = (modes == WALK) | (modes == HIGH_STAND) | (modes == CROUCH_FULL)
    if not bool(torch.all(supported)):
        raise ValueError("G1 single-stage modes contain an unsupported value")

    x_low, x_high = _validate_range("linear x velocity", lin_vel_x_range)
    y_low, y_high = _validate_range("linear y velocity", lin_vel_y_range)
    yaw_low, yaw_high = _validate_range("yaw velocity", ang_vel_yaw_range)
    height_low, height_high = _validate_range("height", height_range)
    commands = torch.zeros(modes.shape[0], 5, dtype=velocity_draws.dtype, device=velocity_draws.device)
    walking = modes == WALK
    commands[walking, 0] = _uniform(velocity_draws[walking, 0], x_low, x_high)
    commands[walking, 1] = _uniform(velocity_draws[walking, 1], y_low, y_high)
    commands[walking, 2] = _uniform(velocity_draws[walking, 2], yaw_low, yaw_high)
    commands[:, 4] = float(stand_height)
    height_hold = modes == CROUCH_FULL
    commands[height_hold, 4] = _uniform(height_draws[height_hold], height_low, height_high)
    return commands


def sample_upper_body_targets(
    *,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
    curriculum_ratio: float,
    amplitude_draws: torch.Tensor,
    joint_draws: torch.Tensor,
    direction_draws: torch.Tensor,
) -> torch.Tensor:
    """Sample an upper-body pose, biased small early in training.

    The amplitude uses a truncated-exponential inverse CDF whose rate falls as
    the curriculum advances, so early episodes barely move the arms and later
    ones approach the full reachable workspace.
    """
    if not 0.0 <= curriculum_ratio <= 1.0:
        raise ValueError("curriculum ratio must lie in [0, 1]")
    amplitude = _exponential_amplitude(amplitude_draws, curriculum_ratio)
    # A second uniform draw per joint so they do not all move in lockstep.
    scale = amplitude * joint_draws
    toward_min = (direction_draws >= 0.5).to(action_min.dtype)
    bound = action_min * toward_min + action_max * (1.0 - toward_min)
    return bound * scale


def _exponential_amplitude(
    draws: torch.Tensor, curriculum_ratio: float
) -> torch.Tensor:
    """Inverse CDF of an exponential truncated to [0, 1].

    The rate falls from 20 to ~0.2 as the curriculum advances, so the
    distribution goes from strongly favouring near-zero amplitudes to nearly
    uniform over the reachable workspace.
    """
    rate = 20.0 * (1.0 - curriculum_ratio * 0.99)
    return -torch.log((1.0 - draws) + draws * math.exp(-rate)) / rate


def upper_body_interpolation_delta(
    target: torch.Tensor, current: torch.Tensor, steps: int
) -> torch.Tensor:
    """Per-step increment that walks `current` to `target` over `steps`.

    Interpolating rather than stepping keeps the arms from jerking, which would
    otherwise inject an impulse the legs have to reject.
    """
    if steps <= 0:
        raise ValueError("interpolation steps must be positive")
    return (target - current) / steps


def advance_action_curriculum(
    *,
    current_ratio: float,
    tracking_x_episode_sums: torch.Tensor,
    tracking_x_reward_scale: float,
    max_episode_length: float,
    increment: float = 0.05,
    threshold_fraction: float = 0.8,
) -> float:
    """Widen the upper-body workspace once walking is good enough.

    `tracking_x_reward_scale` already includes the dt factor, so the threshold
    is a fraction of the best achievable per-step tracking reward.
    """
    if max_episode_length <= 0:
        raise ValueError("max episode length must be positive")
    mean_tracking = float(tracking_x_episode_sums.mean()) / max_episode_length
    threshold = threshold_fraction * tracking_x_reward_scale
    if mean_tracking > threshold:
        return min(current_ratio + increment, 1.0)
    return current_ratio
