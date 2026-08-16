"""Isaac-independent tensor contract for the M3a environment core."""

from __future__ import annotations

import importlib
import importlib.util

import pytest

torch = pytest.importorskip("torch")

from openhomie_isaaclab import elf3_constants as C


MODULE = (
    "openhomie_isaaclab.tasks.locomotion.elf3.elf3_homie_env_core"
)


def test_environment_core_interface_exists():
    assert importlib.util.find_spec(MODULE) is not None, (
        "M3a interface missing: create elf3_homie_env_core.py"
    )


@pytest.fixture(scope="module")
def core():
    if importlib.util.find_spec(MODULE) is None:
        pytest.skip("environment core interface does not exist yet")
    return importlib.import_module(MODULE)


@pytest.fixture
def runtime_names():
    names = list(C.JOINT_NAMES)
    return names[14:] + names[:2] + names[8:14] + names[2:8]


def test_name_permutation_round_trips_a_nontrivial_runtime_order(core, runtime_names):
    permutation = core.build_name_permutation(runtime_names, C.JOINT_NAMES)
    assert permutation.dtype == torch.long
    assert permutation.device.type == "cpu"
    assert permutation.tolist() != list(range(C.NUM_ROBOT_DOFS))
    runtime = torch.arange(C.NUM_ROBOT_DOFS).reshape(1, -1)
    canonical = core.gather_canonical(runtime, permutation)
    assert canonical.tolist()[0] == [runtime_names.index(name) for name in C.JOINT_NAMES]
    restored = core.scatter_canonical(
        canonical, permutation, runtime_width=C.NUM_ROBOT_DOFS
    )
    assert torch.equal(restored, runtime)


@pytest.mark.parametrize(
    "runtime_names",
    [
        list(C.JOINT_NAMES[:-1]),
        list(C.JOINT_NAMES) + ["unknown_joint"],
        list(C.JOINT_NAMES[:-1]) + [C.JOINT_NAMES[0]],
    ],
)
def test_name_permutation_rejects_missing_extra_and_duplicate_names(core, runtime_names):
    with pytest.raises(ValueError):
        core.build_name_permutation(runtime_names, C.JOINT_NAMES)


def test_policy_scatter_writes_only_named_legs(core, runtime_names):
    full_permutation = core.build_name_permutation(runtime_names, C.JOINT_NAMES)
    permutation = full_permutation[torch.tensor(C.LOWER_DOF_INDICES)]
    actions = torch.arange(C.NUM_POLICY_ACTIONS, dtype=torch.float32).reshape(1, -1)
    base = torch.full((1, C.NUM_ROBOT_DOFS), -7.0)
    scattered = core.scatter_canonical(
        actions, permutation, runtime_width=C.NUM_ROBOT_DOFS, base=base
    )
    for runtime_index, name in enumerate(runtime_names):
        if name in C.POLICY_JOINT_NAMES:
            expected = C.POLICY_JOINT_NAMES.index(name)
            assert scattered[0, runtime_index].item() == expected
        else:
            assert scattered[0, runtime_index].item() == -7.0


def test_actor_frame_exact_layout_and_scales(core):
    n = 2
    commands = torch.tensor([[1.0, 2.0, 3.0, 0.9]]).repeat(n, 1)
    ang_vel = torch.tensor([[4.0, 5.0, 6.0]]).repeat(n, 1)
    gravity = torch.tensor([[7.0, 8.0, 9.0]]).repeat(n, 1)
    default = torch.arange(C.NUM_ROBOT_DOFS, dtype=torch.float32).repeat(n, 1)
    dof_pos = default + 2.0
    dof_vel = torch.full_like(dof_pos, 3.0)
    previous_action = torch.full((n, C.NUM_POLICY_ACTIONS), 4.0)
    frame = core.assemble_actor_frame(
        commands, ang_vel, gravity, dof_pos, default, dof_vel, previous_action
    )
    assert frame.shape == (n, C.num_one_step_actor_obs())
    expected_head = torch.tensor(
        [2.0, 4.0, 1.5, 0.9, 2.0, 2.5, 3.0, 7.0, 8.0, 9.0]
    )
    assert torch.allclose(frame[0, : C.NUM_OBS_HEAD], expected_head)
    cursor = C.NUM_OBS_HEAD
    assert torch.equal(frame[0, cursor : cursor + C.NUM_ROBOT_DOFS], torch.full((C.NUM_ROBOT_DOFS,), 2.0))
    cursor += C.NUM_ROBOT_DOFS
    assert torch.allclose(
        frame[0, cursor : cursor + C.NUM_ROBOT_DOFS],
        torch.full((C.NUM_ROBOT_DOFS,), 0.15),
    )
    cursor += C.NUM_ROBOT_DOFS
    assert torch.equal(frame[0, cursor:], previous_action[0])


def test_history_shift_clears_only_reset_environments(core):
    history = torch.arange(
        3 * C.NUM_ACTOR_HISTORY * C.num_one_step_actor_obs(), dtype=torch.float32
    ).reshape(3, C.NUM_ACTOR_HISTORY, C.num_one_step_actor_obs())
    old = history.clone()
    frame = torch.full((3, C.num_one_step_actor_obs()), 99.0)
    shifted = core.shift_history(history, frame, torch.tensor([1]))
    assert torch.equal(shifted[0, :-1], old[0, 1:])
    assert torch.equal(shifted[2, :-1], old[2, 1:])
    assert torch.count_nonzero(shifted[1, :-1]) == 0
    assert torch.equal(shifted[:, -1], frame)


def test_virtual_sole_corners_use_geometry_constants_and_wxyz(core):
    ankle_pos = torch.zeros(1, 2, 3)
    identity_wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(1, 2, 1)
    corners = core.virtual_sole_corners(ankle_pos, identity_wxyz)
    assert corners.shape == (1, 2, 4, 3)
    cx, cy, cz = C.SOLE_CENTER_OFFSET
    expected_local = torch.tensor(
        [
            [cx + C.SOLE_LENGTH / 2, cy + C.SOLE_WIDTH / 2, cz],
            [cx + C.SOLE_LENGTH / 2, cy - C.SOLE_WIDTH / 2, cz],
            [cx - C.SOLE_LENGTH / 2, cy + C.SOLE_WIDTH / 2, cz],
            [cx - C.SOLE_LENGTH / 2, cy - C.SOLE_WIDTH / 2, cz],
        ]
    )
    assert torch.allclose(corners[0, 0], expected_local)
    assert torch.allclose(corners[0, 1], expected_local)

    half_sqrt_two = 2.0**-0.5
    yaw_90_wxyz = torch.tensor(
        [half_sqrt_two, 0.0, 0.0, half_sqrt_two]
    ).repeat(1, 2, 1)
    rotated = core.virtual_sole_corners(ankle_pos, yaw_90_wxyz)
    expected_rotated = torch.stack(
        (-expected_local[:, 1], expected_local[:, 0], expected_local[:, 2]), dim=-1
    )
    assert torch.allclose(rotated[0, 0], expected_rotated, atol=1e-6)


def test_pd_effort_applies_randomized_gains_offset_and_clamp(core):
    desired = torch.tensor([[2.0, -2.0]])
    position = torch.zeros_like(desired)
    velocity = torch.tensor([[1.0, -1.0]])
    effort = core.compute_leg_efforts(
        desired,
        position,
        velocity,
        kp=torch.tensor([10.0, 10.0]),
        kd=torch.tensor([2.0, 2.0]),
        kp_factors=torch.tensor([[2.0, 0.5]]),
        kd_factors=torch.tensor([[0.5, 2.0]]),
        actuation_offset=torch.tensor([[1.0, -1.0]]),
        effort_limits=torch.tensor([5.0, 6.0]),
    )
    assert torch.equal(effort, torch.tensor([[5.0, -6.0]]))


def test_control_randomization_mutates_selected_ids_only(core):
    kp = torch.ones(4, 3)
    kd = torch.ones(4, 3)
    offset = torch.zeros(4, 3)
    env_ids = torch.tensor([1, 3])
    draws = {
        "kp": torch.tensor([[0.2, 0.4, 0.6], [0.8, 0.5, 0.3]]),
        "kd": torch.tensor([[0.1, 0.3, 0.5], [0.7, 0.9, 0.2]]),
        "offset": torch.tensor([[0.0, 0.5, 1.0], [0.25, 0.75, 0.4]]),
    }
    out_kp, out_kd, out_offset = core.apply_control_randomization(
        kp, kd, offset, env_ids, draws=draws,
        kp_range=(0.8, 1.2), kd_range=(0.7, 1.3), offset_range=(-2.0, 2.0)
    )
    assert torch.equal(out_kp[[0, 2]], torch.ones(2, 3))
    assert torch.equal(out_kd[[0, 2]], torch.ones(2, 3))
    assert torch.equal(out_offset[[0, 2]], torch.zeros(2, 3))
    assert not torch.equal(out_kp[env_ids], torch.ones(2, 3))
    assert not torch.equal(out_kd[env_ids], torch.ones(2, 3))
    assert not torch.equal(out_offset[env_ids], torch.zeros(2, 3))
    assert torch.all((out_kp[env_ids] >= 0.8) & (out_kp[env_ids] <= 1.2))
    assert torch.all((out_kd[env_ids] >= 0.7) & (out_kd[env_ids] <= 1.3))
    assert torch.all((out_offset[env_ids] >= -2.0) & (out_offset[env_ids] <= 2.0))


def test_done_semantics_keep_failure_and_timeout_separate(core):
    failure = torch.tensor([False, True, False, True])
    episode_length = torch.tensor([3, 3, 4, 4])
    terminated, truncated, time_outs = core.classify_dones(
        failure, episode_length, max_episode_length=4
    )
    assert torch.equal(terminated, torch.tensor([False, True, False, True]))
    assert torch.equal(truncated, torch.tensor([False, False, True, False]))
    assert torch.equal(time_outs, truncated)
