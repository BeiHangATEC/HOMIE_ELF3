"""ELF3 reward formulas: shape, sign, and the behaviour each term encodes.

Pure tensor maths, no Isaac Sim. The point is not to restate the formulas but
to pin the properties that make them correct: which way each term should move,
and which ones are gated by the commanded mode.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from openhomie_isaaclab.tasks.locomotion.elf3 import elf3_homie_rewards as R


STEP_DT = 0.02


def _cmd(vx=0.0, vy=0.0, wyaw=0.0, height=1.013):
    # commands layout: [vx, vy, wyaw, unused, height]
    return torch.tensor([[vx, vy, wyaw, 0.0, height]])


# --------------------------------------------------------------------------
# Table integrity
# --------------------------------------------------------------------------
def test_every_named_term_has_a_scale():
    assert set(R.REWARD_SCALES) == set(R.REWARD_NAMES)
    assert len(R.REWARD_NAMES) == 33
    assert len(set(R.REWARD_NAMES)) == len(R.REWARD_NAMES)


def test_every_named_term_has_an_implementation():
    for name in R.REWARD_NAMES:
        assert callable(getattr(R, name)), f"{name} has no implementation"


def test_tracking_terms_are_rewards_and_regularizers_are_penalties():
    positive = {
        "tracking_x_vel",
        "tracking_y_vel",
        "tracking_ang_vel",
        "tracking_base_height",
        "feet_air_time",
        "feet_distance_lateral",
        "knee_distance_lateral",
        "no_fly",
        "contact_momentum",
    }
    for name, scale in R.REWARD_SCALES.items():
        if name in positive:
            assert scale > 0.0, f"{name} should reward, got {scale}"
        else:
            assert scale < 0.0, f"{name} should penalize, got {scale}"


# --------------------------------------------------------------------------
# Command tracking
# --------------------------------------------------------------------------
def test_velocity_tracking_peaks_when_the_command_is_met():
    vel = torch.tensor([[0.5, 0.0, 0.0]])
    exact = R.tracking_x_vel(_cmd(vx=0.5), vel, 0.25)
    wrong = R.tracking_x_vel(_cmd(vx=-0.5), vel, 0.25)
    assert exact.item() == pytest.approx(1.0)
    assert wrong.item() < exact.item()


def test_velocity_tracking_is_offset_down_while_crouching():
    """Crouch episodes command zero velocity, so a perfect exp() would pay
    full reward for standing still and swamp the height term."""
    vel = torch.zeros(1, 3)
    walking = R.tracking_x_vel(_cmd(), vel, 0.25, modes=torch.tensor([0]))
    crouching = R.tracking_x_vel(_cmd(), vel, 0.25, modes=torch.tensor([2]))
    assert walking.item() == pytest.approx(1.0)
    assert crouching.item() == pytest.approx(0.0)


def test_height_tracking_peaks_at_the_commanded_height():
    root = torch.tensor([1.0])
    feet = torch.zeros(1, 2)
    good = R.tracking_base_height(
        root_height=root,
        feet_height=feet,
        commanded_height=torch.tensor([1.02]),
        ankle_sole_distance=0.02,
    )
    bad = R.tracking_base_height(
        root_height=root,
        feet_height=feet,
        commanded_height=torch.tensor([0.80]),
        ankle_sole_distance=0.02,
    )
    assert good.item() == pytest.approx(1.0, abs=1e-6)
    assert bad.item() < good.item()


def test_height_tracking_doubles_while_crouching():
    kwargs = dict(
        root_height=torch.tensor([1.0]),
        feet_height=torch.zeros(1, 2),
        commanded_height=torch.tensor([1.02]),
        ankle_sole_distance=0.02,
    )
    walk = R.tracking_base_height(**kwargs, modes=torch.tensor([0]))
    crouch = R.tracking_base_height(**kwargs, modes=torch.tensor([2]))
    assert crouch.item() == pytest.approx(2.0 * walk.item())


def test_height_tracking_uses_the_higher_foot():
    """A lifted foot must not make the robot look taller than it stands."""
    reward = R.tracking_base_height(
        root_height=torch.tensor([1.0]),
        feet_height=torch.tensor([[0.0, 0.5]]),
        commanded_height=torch.tensor([1.02]),
        ankle_sole_distance=0.02,
    )
    # max(1.0-0.0, 1.0-0.5) = 1.0, so this is the flat-footed answer.
    assert reward.item() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# Mode gating
# --------------------------------------------------------------------------
def test_lin_vel_z_only_penalizes_while_standing_tall():
    """Squatting needs vertical motion, so bobbing is only penalized when
    the robot has been told to hold a tall pose."""
    vel = torch.tensor([[0.0, 0.0, 0.4]])
    tall = R.lin_vel_z(vel, torch.tensor([1.0]))
    crouch = R.lin_vel_z(vel, torch.tensor([0.0]))
    assert tall.item() == pytest.approx(0.16)
    assert crouch.item() == pytest.approx(0.0)


def test_stand_still_only_fires_while_commanded_tall_and_still():
    airborne = torch.tensor([[0.0, 0.0]])
    assert R.stand_still(airborne, torch.tensor([1.0])).item() == pytest.approx(2.0)
    assert R.stand_still(airborne, torch.tensor([0.0])).item() == pytest.approx(0.0)


def test_feet_air_time_is_zero_without_a_motion_command():
    air = torch.tensor([[0.6, 0.0]])
    contact = torch.tensor([[1.0, 0.0]])
    moving = R.feet_air_time(air, contact, torch.tensor([0.5]))
    still = R.feet_air_time(air, contact, torch.tensor([0.0]))
    assert moving.item() == pytest.approx(0.1)
    assert still.item() == pytest.approx(0.0)


def test_no_fly_rewards_single_support_and_pays_full_while_standing():
    single = torch.tensor([[10.0, 0.0]])
    double = torch.tensor([[10.0, 10.0]])
    not_standing = torch.tensor([0.0])
    assert R.no_fly(single, not_standing).item() == pytest.approx(1.0)
    assert R.no_fly(double, not_standing).item() == pytest.approx(0.0)
    # Standing has nothing to alternate, so it should not be penalized.
    assert R.no_fly(double, torch.tensor([1.0])).item() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Contact terms
# --------------------------------------------------------------------------
def test_feet_stumble_detects_shear_dominated_contact():
    normal = torch.tensor([[[1.0, 0.0, 100.0], [1.0, 0.0, 100.0]]])
    stubbed = torch.tensor([[[50.0, 0.0, 5.0], [1.0, 0.0, 100.0]]])
    assert R.feet_stumble(normal).item() == pytest.approx(0.0)
    assert R.feet_stumble(stubbed).item() == pytest.approx(1.0)


def test_feet_contact_forces_only_penalizes_above_the_cap():
    under = torch.tensor([[[0.0, 0.0, 300.0], [0.0, 0.0, 0.0]]])
    over = torch.tensor([[[0.0, 0.0, 500.0], [0.0, 0.0, 0.0]]])
    assert R.feet_contact_forces(under, 400.0).item() == pytest.approx(0.0)
    assert R.feet_contact_forces(over, 400.0).item() == pytest.approx(100.0)


def test_feet_slip_needs_both_contact_and_motion():
    moving = torch.tensor([[[0.3, 0.4], [0.0, 0.0]]])
    loaded = torch.tensor([[10.0, 10.0]])
    airborne = torch.tensor([[0.0, 0.0]])
    assert R.feet_slip(moving, loaded).item() == pytest.approx(0.5)
    assert R.feet_slip(moving, airborne).item() == pytest.approx(0.0)


def test_contact_momentum_only_counts_descending_feet():
    force = torch.tensor([[150.0, 150.0]])
    descending = R.contact_momentum(torch.tensor([[-1.0, 0.0]]), force)
    ascending = R.contact_momentum(torch.tensor([[1.0, 0.0]]), force)
    assert descending.item() < 0.0
    assert ascending.item() == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Stance geometry
# --------------------------------------------------------------------------
def test_lateral_distance_rewards_only_the_band():
    assert R.feet_distance_lateral(torch.tensor([0.28]), 0.2, 0.35).item() == 1.0
    assert R.feet_distance_lateral(torch.tensor([0.10]), 0.2, 0.35).item() == 0.0
    assert R.feet_distance_lateral(torch.tensor([0.50]), 0.2, 0.35).item() == 0.0


def test_knee_band_is_twice_the_foot_band():
    """Two knee pairs are summed, so the band doubles."""
    assert R.knee_distance_lateral(torch.tensor([0.56]), 0.2, 0.35).item() == 1.0
    assert R.knee_distance_lateral(torch.tensor([0.28]), 0.2, 0.35).item() == 0.0


def test_feet_parallel_is_zero_for_parallel_soles():
    parallel = torch.tensor([[0.2, 0.2, 0.2, 0.2]])
    tilted = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    assert R.feet_parallel(parallel).item() == pytest.approx(0.0)
    assert R.feet_parallel(tilted).item() > 0.0


def test_feet_ground_parallel_ignores_airborne_feet():
    variance = torch.tensor([[0.05, 0.05]])
    loaded = R.feet_ground_parallel(variance, torch.tensor([[1.0, 1.0]]))
    lifted = R.feet_ground_parallel(variance, torch.tensor([[0.0, 0.0]]))
    assert loaded.item() == pytest.approx(0.1)
    assert lifted.item() == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Joint and torque regularization
# --------------------------------------------------------------------------
def test_limit_penalties_are_zero_inside_the_limits():
    pos = torch.tensor([[0.0, 0.5]])
    lower = torch.tensor([-1.0, -1.0])
    upper = torch.tensor([1.0, 1.0])
    assert R.dof_pos_limits(pos, lower, upper).item() == pytest.approx(0.0)
    outside = torch.tensor([[-1.5, 1.25]])
    assert R.dof_pos_limits(outside, lower, upper).item() == pytest.approx(0.75)


def test_torque_limits_use_the_soft_ratio():
    torque = torch.tensor([[100.0]])
    limits = torch.tensor([100.0])
    # 0.95 * 100 = 95, so 5 N m of the applied torque is over the soft limit.
    assert R.torque_limits(torque, limits, 0.95, [0]).item() == pytest.approx(5.0)
    assert R.torque_limits(
        torch.tensor([[90.0]]), limits, 0.95, [0]
    ).item() == pytest.approx(0.0)


def test_action_vanish_penalizes_only_clipped_action():
    lo = torch.tensor([[-1.0, -1.0]])
    hi = torch.tensor([[1.0, 1.0]])
    assert R.action_vanish(torch.tensor([[0.5, -0.5]]), lo, hi).item() == pytest.approx(
        0.0
    )
    # 2.0 exceeds the upper bound by 1.0; -3.0 undershoots by 2.0.
    assert R.action_vanish(torch.tensor([[2.0, -3.0]]), lo, hi).item() == pytest.approx(
        3.0
    )


def test_smoothness_is_zero_for_constant_and_linear_action():
    a = torch.tensor([[1.0]])
    assert R.smoothness(a, a, a).item() == pytest.approx(0.0)
    # A linear ramp has zero second difference.
    assert R.smoothness(
        torch.tensor([[2.0]]), torch.tensor([[1.0]]), torch.tensor([[0.0]])
    ).item() == pytest.approx(0.0)
    # A direction reversal does not.
    assert R.smoothness(
        torch.tensor([[0.0]]), torch.tensor([[1.0]]), torch.tensor([[0.0]])
    ).item() == pytest.approx(4.0)


def test_joint_power_is_discounted_by_the_commanded_effort():
    vel = torch.tensor([[2.0]])
    torque = torch.tensor([[10.0]])
    still = R.joint_power(vel, torque, _cmd())
    fast = R.joint_power(vel, torque, _cmd(vx=1.0))
    assert fast.item() < still.item(), "walking fast should be penalized less"


def test_dof_acc_scales_with_the_timestep():
    slow = R.dof_acc(torch.zeros(1, 1), torch.tensor([[1.0]]), 0.02)
    assert slow.item() == pytest.approx(2500.0)


def test_deviation_knee_couples_bend_to_height_error():
    """Zero height error means no preference about knee angle."""
    kwargs = dict(
        dof_pos=torch.tensor([[1.0]]),
        joint_lower_limits=torch.tensor([0.0]),
        joint_upper_limits=torch.tensor([2.0]),
        knee_joint_indices=[0],
        root_height=torch.tensor([1.0]),
    )
    on_target = R.deviation_knee_joint(**kwargs, commanded_height=torch.tensor([1.0]))
    off_target = R.deviation_knee_joint(**kwargs, commanded_height=torch.tensor([0.8]))
    assert on_target.item() == pytest.approx(0.0)
    # The knee sits mid-range (normalized 0.5), so the term still cancels.
    assert off_target.item() == pytest.approx(0.0)

    bent = R.deviation_knee_joint(
        dof_pos=torch.tensor([[1.8]]),
        joint_lower_limits=torch.tensor([0.0]),
        joint_upper_limits=torch.tensor([2.0]),
        knee_joint_indices=[0],
        root_height=torch.tensor([1.0]),
        commanded_height=torch.tensor([0.8]),
    )
    assert bent.item() > 0.0


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def _raw_terms(num_envs=2):
    return {name: torch.ones(num_envs) for name in R.REWARD_NAMES}


def test_scaling_applies_scale_times_dt():
    scaled = R.scale_reward_terms(_raw_terms(), R.REWARD_SCALES, STEP_DT)
    assert set(scaled) == set(R.REWARD_NAMES)
    expected = R.REWARD_SCALES["tracking_x_vel"] * STEP_DT
    assert scaled["tracking_x_vel"][0].item() == pytest.approx(expected)


def test_scaling_rejects_a_missing_term():
    """Adding a reward must be a deliberate two-place edit, not a silent no-op."""
    incomplete = _raw_terms()
    del incomplete["no_fly"]
    with pytest.raises(ValueError, match="missing.*no_fly"):
        R.scale_reward_terms(incomplete, R.REWARD_SCALES, STEP_DT)


def test_scaling_rejects_an_unknown_term():
    extra = _raw_terms()
    extra["invented_term"] = torch.ones(2)
    with pytest.raises(ValueError, match="unexpected.*invented_term"):
        R.scale_reward_terms(extra, R.REWARD_SCALES, STEP_DT)


def test_scaling_rejects_an_incomplete_scale_table():
    partial = {k: v for k, v in R.REWARD_SCALES.items() if k != "orientation"}
    with pytest.raises(ValueError, match="scales mismatch"):
        R.scale_reward_terms(_raw_terms(), partial, STEP_DT)


def test_sum_matches_the_scale_total_and_preserves_shape():
    scaled = R.scale_reward_terms(_raw_terms(num_envs=4), R.REWARD_SCALES, STEP_DT)
    total = R.sum_reward_terms(scaled)
    assert total.shape == (4,)
    expected = sum(R.REWARD_SCALES.values()) * STEP_DT
    assert total[0].item() == pytest.approx(expected, rel=1e-5)


def test_sum_does_not_mutate_its_input():
    scaled = R.scale_reward_terms(_raw_terms(), R.REWARD_SCALES, STEP_DT)
    before = scaled["orientation"].clone()
    R.sum_reward_terms(scaled)
    assert torch.equal(scaled["orientation"], before)


def test_all_terms_stay_finite_on_degenerate_input():
    """Zero-width joint limits and zero commands must not produce NaN."""
    zero = torch.zeros(1, 1)
    assert torch.isfinite(
        R.deviation_knee_joint(
            dof_pos=zero,
            joint_lower_limits=torch.tensor([0.0]),
            joint_upper_limits=torch.tensor([0.0]),
            knee_joint_indices=[0],
            root_height=torch.tensor([1.0]),
            commanded_height=torch.tensor([1.0]),
        )
    ).all()
    assert torch.isfinite(R.joint_power(zero, zero, _cmd())).all()
    assert torch.isfinite(R.torques(zero, torch.tensor([0.0]), [0])).all()
