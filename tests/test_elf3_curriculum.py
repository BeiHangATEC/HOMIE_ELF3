"""ELF3 curriculum: command sampling, stage ladder, and the amplitude ramp.

No Isaac Sim, no GPU: every source of randomness is an injected `draws`
tensor, so the sampling behaviour is fully determined by the test.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from openhomie_isaaclab import elf3_constants as C
from openhomie_isaaclab.tasks.locomotion.elf3 import elf3_homie_curriculum as K
from openhomie_isaaclab.tasks.locomotion.elf3 import elf3_stages as S


# --------------------------------------------------------------------------
# Stage ladder direction
# --------------------------------------------------------------------------
def test_ladder_descends_in_height_then_widens_velocity():
    """The whole point of the rebuild: start standing, crouch later."""
    heights = [S.STAGE_SPECS[n].walk_height for n in S.STAGE_ORDER if n.startswith("S")]
    assert heights == sorted(heights, reverse=True), "height stages must descend"
    assert heights[0] > heights[-1]

    speeds = [S.STAGE_SPECS[n].lin_vel_x[1] for n in S.STAGE_ORDER if n.startswith("V")]
    assert speeds == sorted(speeds), "velocity stages must widen"


def test_first_stage_is_the_natural_standing_pose():
    first = S.get_stage(S.STAGE_ORDER[0])
    assert first.walk_height == pytest.approx(1.01, abs=0.01)
    assert abs(first.walk_height - C.DEFAULT_BASE_HEIGHT) < 0.05


def test_no_stage_demands_an_unreachable_height():
    """0.34 m -- the old first stage -- needs about -2.88 rad of hip pitch."""
    for name in S.STAGE_ORDER:
        height = S.STAGE_SPECS[name].walk_height
        assert C.MIN_HEIGHT_COMMAND <= height <= C.MAX_HEIGHT_COMMAND, name
    assert min(s.walk_height for s in S.STAGE_SPECS.values()) > 0.34


def test_velocity_ranges_stay_inside_the_aggregate_window():
    for name in S.STAGE_ORDER:
        spec = S.STAGE_SPECS[name]
        assert spec.lin_vel_x[0] < spec.lin_vel_x[1]
        assert spec.lin_vel_y[0] < spec.lin_vel_y[1]
        assert spec.ang_vel_yaw[0] < spec.ang_vel_yaw[1]


def test_stage_navigation():
    assert S.next_stage(S.STAGE_ORDER[0]) == S.STAGE_ORDER[1]
    assert S.next_stage(S.STAGE_ORDER[-1]) is None
    with pytest.raises(ValueError, match="Unknown"):
        S.get_stage("H0")  # the old naming must not silently resolve


def test_crouch_band_is_clamped_to_the_stage_height():
    shallow = S.get_stage("S5")  # 0.80 m
    low, high = S.effective_crouch_low_bounds(shallow)
    assert low <= high <= shallow.walk_height


def test_crouch_band_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="must not exceed"):
        S.effective_crouch_low_bounds("S0", minimum=0.95, nominal_maximum=0.80)


def test_canonical_definitions_are_json_ready():
    definitions = S.canonical_stage_definitions()
    assert [d["name"] for d in definitions] == list(S.STAGE_ORDER)
    assert all(isinstance(d["lin_vel_x"], list) for d in definitions)


# --------------------------------------------------------------------------
# Mode sampling
# --------------------------------------------------------------------------
def test_mode_probabilities_sum_to_one():
    assert sum(K.MODE_PROBABILITIES) == pytest.approx(1.0)
    assert len(K.MODE_PROBABILITIES) == 4


def test_mode_boundaries():
    """`torch.bucketize` is right-open, so an exact boundary falls low."""
    draws = torch.tensor([0.0, 0.60, 0.6001, 0.75, 0.7501, 0.90, 0.9001, 0.999])
    modes = K.sample_modes(draws)
    assert modes.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_mode_proportions_match_the_configured_mix():
    generator = torch.Generator().manual_seed(0)
    modes = K.sample_modes(torch.rand(200_000, generator=generator))
    for mode, expected in enumerate(K.MODE_PROBABILITIES):
        observed = (modes == mode).to(torch.float32).mean().item()
        assert observed == pytest.approx(expected, abs=0.01), mode


def test_mode_sampling_rejects_out_of_range_draws():
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        K.sample_modes(torch.tensor([1.0]))
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        K.sample_modes(torch.tensor([-0.1]))


def test_mode_masks_partition_the_modes():
    modes = torch.tensor([K.WALK, K.HIGH_STAND, K.CROUCH_LOW, K.CROUCH_FULL])
    assert K.walk_mask(modes).tolist() == [True, False, False, False]
    assert K.high_mask(modes).tolist() == [True, True, False, False]
    assert K.stand_mask(modes).tolist() == [False, True, True, True]
    assert K.crouch_mask(modes).tolist() == [False, False, True, True]


def test_crouch_ordering_is_load_bearing():
    """Reward shaping tests `modes >= CROUCH_LOW`, so the order must hold."""
    assert K.WALK < K.HIGH_STAND < K.CROUCH_LOW < K.CROUCH_FULL


# --------------------------------------------------------------------------
# Command building
# --------------------------------------------------------------------------
def _build(modes, vel=0.5, height=0.5, stage="S0"):
    n = modes.shape[0]
    return K.build_commands(
        modes=modes,
        velocity_draws=torch.full((n, 3), vel),
        height_draws=torch.full((n,), height),
        stage=S.get_stage(stage),
        crouch_min_height=S.CROUCH_MIN_HEIGHT,
        crouch_focus_max_height=S.CROUCH_LOW_NOMINAL_MAX_HEIGHT,
    )


def test_only_walking_envs_get_a_velocity_command():
    commands = _build(torch.tensor([K.WALK, K.HIGH_STAND, K.CROUCH_FULL]))
    assert torch.any(commands[0, :3] != 0.0)
    assert torch.all(commands[1, :3] == 0.0)
    assert torch.all(commands[2, :3] == 0.0)


def test_command_layout_is_five_wide_with_an_unused_slot():
    commands = _build(torch.tensor([K.WALK]))
    assert commands.shape == (1, 5)
    assert commands[0, 3].item() == 0.0, "column 3 is the unused heading slot"


def test_tall_modes_hold_the_stage_height():
    stage = S.get_stage("S0")
    commands = _build(torch.tensor([K.WALK, K.HIGH_STAND]), stage="S0")
    assert commands[0, 4].item() == pytest.approx(stage.walk_height)
    assert commands[1, 4].item() == pytest.approx(stage.walk_height)


def test_crouch_modes_sample_below_the_stage_height():
    stage = S.get_stage("S0")
    commands = _build(torch.tensor([K.CROUCH_LOW, K.CROUCH_FULL]), height=0.5)
    for row in range(2):
        assert S.CROUCH_MIN_HEIGHT <= commands[row, 4].item() <= stage.walk_height


def test_full_crouch_can_reach_deeper_than_shallow_crouch():
    """height_draws=1.0 takes each band to its own maximum."""
    commands = _build(torch.tensor([K.CROUCH_LOW, K.CROUCH_FULL]), height=1.0)
    shallow, full = commands[0, 4].item(), commands[1, 4].item()
    assert shallow == pytest.approx(S.CROUCH_LOW_NOMINAL_MAX_HEIGHT)
    assert full == pytest.approx(S.get_stage("S0").walk_height)


def test_every_commanded_height_is_reachable():
    """Sweep the draw range across all stages and modes."""
    for name in S.STAGE_ORDER:
        for draw in (0.0, 0.5, 1.0):
            commands = _build(
                torch.tensor([K.WALK, K.HIGH_STAND, K.CROUCH_LOW, K.CROUCH_FULL]),
                height=draw,
                stage=name,
            )
            heights = commands[:, 4]
            assert torch.all(heights >= C.MIN_HEIGHT_COMMAND - 1e-6), (name, draw)
            assert torch.all(heights <= C.MAX_HEIGHT_COMMAND + 1e-6), (name, draw)


def test_velocity_draws_map_onto_the_stage_range():
    stage = S.get_stage("V3")
    low = _build(torch.tensor([K.WALK]), vel=0.0, stage="V3")
    high = _build(torch.tensor([K.WALK]), vel=1.0, stage="V3")
    assert low[0, 0].item() == pytest.approx(stage.lin_vel_x[0])
    assert high[0, 0].item() == pytest.approx(stage.lin_vel_x[1])


def test_build_rejects_malformed_draws():
    with pytest.raises(ValueError, match="three columns"):
        K.build_commands(
            modes=torch.tensor([K.WALK]),
            velocity_draws=torch.zeros(1, 2),
            height_draws=torch.zeros(1),
            stage=S.get_stage("S0"),
            crouch_min_height=S.CROUCH_MIN_HEIGHT,
            crouch_focus_max_height=S.CROUCH_LOW_NOMINAL_MAX_HEIGHT,
        )


# --------------------------------------------------------------------------
# G1-proportioned single-stage command sampling
# --------------------------------------------------------------------------
def test_g1_single_stage_mode_boundaries_match_g1_sampler():
    draws = torch.tensor([0.0, 0.3334, 0.5, 0.5001])
    modes = K.sample_g1_single_stage_modes(draws)
    assert modes.tolist() == [K.CROUCH_FULL, K.HIGH_STAND, K.HIGH_STAND, K.WALK]


def test_g1_single_stage_mode_proportions_are_exact_for_partitioned_draws():
    draws = torch.tensor([0.1] * 200 + [0.4] * 100 + [0.9] * 300)
    modes = K.sample_g1_single_stage_modes(draws)
    assert (modes == K.WALK).sum().item() == 300
    assert (modes == K.CROUCH_FULL).sum().item() == 200
    assert (modes == K.HIGH_STAND).sum().item() == 100


def test_g1_single_stage_commands_use_the_full_velocity_and_height_ranges():
    modes = torch.tensor([K.WALK, K.CROUCH_FULL, K.HIGH_STAND])
    commands = K.build_g1_single_stage_commands(
        modes=modes,
        velocity_draws=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]),
        height_draws=torch.tensor([0.0, 1.0, 0.5]),
        lin_vel_x_range=(-0.8, 1.2),
        lin_vel_y_range=(-0.5, 0.5),
        ang_vel_yaw_range=(-0.8, 0.8),
        height_range=(0.40, 1.01),
        stand_height=1.01,
    )
    assert commands.shape == (3, 5)
    assert commands[0].tolist() == pytest.approx([-0.8, -0.5, -0.8, 0.0, 1.01])
    assert commands[1].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.01])
    assert commands[2].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.01])


def test_g1_single_stage_height_lower_bound_is_not_clamped():
    commands = K.build_g1_single_stage_commands(
        modes=torch.tensor([K.CROUCH_FULL]),
        velocity_draws=torch.zeros(1, 3),
        height_draws=torch.zeros(1),
        lin_vel_x_range=(-0.8, 1.2),
        lin_vel_y_range=(-0.5, 0.5),
        ang_vel_yaw_range=(-0.8, 0.8),
        height_range=(0.40, 1.01),
        stand_height=1.01,
    )
    assert commands[0, 4].item() == pytest.approx(0.40)


# --------------------------------------------------------------------------
# Upper-body amplitude curriculum
# --------------------------------------------------------------------------
def _amplitude_mean(ratio, n=20_000, seed=0):
    generator = torch.Generator().manual_seed(seed)
    draws = torch.rand(n, generator=generator)
    return float(K._exponential_amplitude(draws, ratio).mean())


def test_amplitude_grows_monotonically_with_the_curriculum():
    means = [_amplitude_mean(r) for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert means == sorted(means), means
    assert means[0] < 0.1, "early training should barely move the arms"
    assert means[-1] > 0.3, "a finished curriculum should use the workspace"


def test_amplitude_stays_in_the_unit_interval():
    for ratio in (0.0, 0.5, 1.0):
        generator = torch.Generator().manual_seed(1)
        amplitude = K._exponential_amplitude(torch.rand(5_000, generator=generator), ratio)
        assert torch.all(amplitude >= 0.0)
        assert torch.all(amplitude <= 1.0 + 1e-6)
        assert torch.isfinite(amplitude).all()


def test_upper_body_targets_respect_the_action_bounds():
    n, joints = 4, 3
    action_min = torch.full((n, joints), -2.0)
    action_max = torch.full((n, joints), 3.0)
    generator = torch.Generator().manual_seed(2)
    target = K.sample_upper_body_targets(
        action_min=action_min,
        action_max=action_max,
        curriculum_ratio=1.0,
        amplitude_draws=torch.rand(n, joints, generator=generator),
        joint_draws=torch.rand(n, joints, generator=generator),
        direction_draws=torch.rand(n, joints, generator=generator),
    )
    assert target.shape == (n, joints)
    assert torch.all(target >= action_min - 1e-6)
    assert torch.all(target <= action_max + 1e-6)


def test_zero_curriculum_keeps_targets_near_the_default_pose():
    n, joints = 8, 4
    generator = torch.Generator().manual_seed(3)
    target = K.sample_upper_body_targets(
        action_min=torch.full((n, joints), -2.0),
        action_max=torch.full((n, joints), 3.0),
        curriculum_ratio=0.0,
        amplitude_draws=torch.rand(n, joints, generator=generator),
        joint_draws=torch.rand(n, joints, generator=generator),
        direction_draws=torch.rand(n, joints, generator=generator),
    )
    assert float(target.abs().max()) < 1.0


def test_sample_rejects_an_out_of_range_ratio():
    kwargs = dict(
        action_min=torch.zeros(1, 1),
        action_max=torch.ones(1, 1),
        amplitude_draws=torch.zeros(1, 1),
        joint_draws=torch.zeros(1, 1),
        direction_draws=torch.zeros(1, 1),
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        K.sample_upper_body_targets(**kwargs, curriculum_ratio=1.5)


def test_interpolation_reaches_the_target_in_exactly_n_steps():
    current = torch.zeros(1, 3)
    target = torch.tensor([[1.0, -2.0, 0.5]])
    steps = 50
    delta = K.upper_body_interpolation_delta(target, current, steps)
    assert torch.allclose(current + delta * steps, target)


def test_interpolation_rejects_non_positive_steps():
    with pytest.raises(ValueError, match="positive"):
        K.upper_body_interpolation_delta(torch.zeros(1), torch.zeros(1), 0)


def test_action_curriculum_advances_only_when_walking_is_good():
    # scale already includes dt; threshold is 0.8 of the achievable per-step sum
    scale = 1.5 * 0.02
    good = torch.full((16,), 0.9 * scale * 1000)
    poor = torch.full((16,), 0.5 * scale * 1000)
    advanced = K.advance_action_curriculum(
        current_ratio=0.0,
        tracking_x_episode_sums=good,
        tracking_x_reward_scale=scale,
        max_episode_length=1000,
    )
    held = K.advance_action_curriculum(
        current_ratio=0.0,
        tracking_x_episode_sums=poor,
        tracking_x_reward_scale=scale,
        max_episode_length=1000,
    )
    assert advanced == pytest.approx(0.05)
    assert held == pytest.approx(0.0)


def test_action_curriculum_saturates_at_one():
    scale = 1.5 * 0.02
    assert K.advance_action_curriculum(
        current_ratio=0.99,
        tracking_x_episode_sums=torch.full((4,), 1000.0 * scale),
        tracking_x_reward_scale=scale,
        max_episode_length=1000,
    ) == pytest.approx(1.0)
