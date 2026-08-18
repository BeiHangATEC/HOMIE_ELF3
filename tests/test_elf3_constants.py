"""ELF3 invariants: joint table, derived dimensions, mirror spec, geometry.

These tests are the specification. They require neither Isaac Sim nor a GPU,
so they run in about a second and gate everything downstream.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

from openhomie_isaaclab import elf3_constants as C


ACTUATED_JOINT_TYPES = {"revolute", "continuous", "prismatic"}


@pytest.fixture(scope="module")
def urdf_root() -> ET.Element:
    assert C.URDF_PATH.is_file(), f"missing ELF3 URDF at {C.URDF_PATH}"
    return ET.parse(C.URDF_PATH).getroot()


@pytest.fixture(scope="module")
def urdf_joints(urdf_root: ET.Element) -> list[ET.Element]:
    return list(urdf_root.findall("joint"))


# --------------------------------------------------------------------------
# Joint table
# --------------------------------------------------------------------------
def test_urdf_has_28_actuated_joints(urdf_joints):
    actuated = [j for j in urdf_joints if j.get("type") in ACTUATED_JOINT_TYPES]
    assert len(actuated) == 28


def test_waist_roll_is_fixed(urdf_joints):
    """ELF3 does not expose waist roll to the policy.

    The vendor MuJoCo model *does* model it as a hinge, which is why the
    sim2sim bridge has to lock it; see test_mujoco_locked_joints_are_known.
    """
    waist_x = [j for j in urdf_joints if j.get("name") == "waist_x_joint"]
    assert len(waist_x) == 1
    assert waist_x[0].get("type") == "fixed"


def test_joint_names_match_urdf_order(urdf_joints):
    """The canonical order must equal the URDF's actuated-joint order."""
    urdf_order = [
        j.get("name") for j in urdf_joints if j.get("type") in ACTUATED_JOINT_TYPES
    ]
    assert list(C.JOINT_NAMES) == urdf_order


def test_joint_names_are_unique():
    assert len(set(C.JOINT_NAMES)) == len(C.JOINT_NAMES)


def test_joint_groups_partition_the_robot():
    lower = set(C.POLICY_JOINT_NAMES)
    upper = set(C.UPPER_BODY_JOINT_NAMES)
    assert len(lower) == 12
    assert len(upper) == 16
    assert not lower & upper
    assert lower | upper == set(C.JOINT_NAMES)


def test_leg_indices_are_not_the_first_twelve():
    """Guards the 'first 12 DOFs are the legs' assumption.

    ELF3's waist joints come first, so any code slicing [:12] for the legs is
    wrong. Indices must always be resolved by name.
    """
    assert C.LOWER_DOF_INDICES == tuple(range(2, 14))
    assert C.LOWER_DOF_INDICES != tuple(range(12))


def test_index_groups_agree_with_name_groups():
    by_index = [C.JOINT_NAMES[i] for i in C.LOWER_DOF_INDICES]
    assert by_index == C.POLICY_JOINT_NAMES
    by_index = [C.JOINT_NAMES[i] for i in C.UPPER_DOF_INDICES]
    assert by_index == C.UPPER_BODY_JOINT_NAMES


def test_joint_index_rejects_unknown_names():
    assert C.joint_index("l_knee_y_joint") == C.JOINT_NAMES.index("l_knee_y_joint")
    with pytest.raises(KeyError):
        C.joint_index("nonexistent_joint")


def test_default_pose_covers_every_joint():
    assert set(C.DEFAULT_JOINT_POS) == set(C.JOINT_NAMES)


# --------------------------------------------------------------------------
# Derived dimensions
# --------------------------------------------------------------------------
def test_derived_observation_dimensions():
    assert C.NUM_ROBOT_DOFS == 28
    assert C.NUM_POLICY_ACTIONS == 12
    assert C.num_one_step_actor_obs() == 78
    assert C.num_one_step_critic_obs() == 81
    assert C.num_actor_obs() == 468
    assert C.num_critic_obs() == 81
    assert C.num_actor_mlp_input() == 113


def test_dimensions_track_the_joint_table():
    """Dimensions must be derived, not literals that drift on a DOF change."""
    expected = C.NUM_OBS_HEAD + 2 * C.NUM_ROBOT_DOFS + C.NUM_POLICY_ACTIONS
    assert C.num_one_step_actor_obs() == expected
    assert C.num_one_step_critic_obs() == expected + C.NUM_PRIVILEGED_OBS
    assert C.num_actor_obs() == expected * C.NUM_ACTOR_HISTORY


def test_policy_control_period_is_50hz():
    assert C.policy_dt() == pytest.approx(0.02)
    assert round(C.EPISODE_LENGTH_S / C.policy_dt()) == 1000


# --------------------------------------------------------------------------
# Mirror spec
# --------------------------------------------------------------------------
def _assert_valid_mirror(indices, signs, width):
    assert len(indices) == width
    assert len(signs) == width
    assert sorted(indices) == list(range(width)), "not a permutation"
    for i in range(width):
        assert indices[indices[i]] == i, f"not an involution at {i}"
        assert signs[i] in (-1, 1), f"sign at {i} is not +-1"
        assert signs[i] == signs[indices[i]], f"asymmetric sign at {i}"


def test_dof_mirror_is_a_signed_involution():
    _assert_valid_mirror(C.DOF_MIRROR_INDICES, C.DOF_MIRROR_SIGNS, C.NUM_ROBOT_DOFS)


def test_action_mirror_is_a_signed_involution():
    _assert_valid_mirror(
        C.ACTION_MIRROR_INDICES, C.ACTION_MIRROR_SIGNS, C.NUM_POLICY_ACTIONS
    )


def test_mirror_swaps_left_and_right_joints():
    for index, name in enumerate(C.JOINT_NAMES):
        mirrored = C.JOINT_NAMES[C.DOF_MIRROR_INDICES[index]]
        if name.startswith("l_"):
            assert mirrored == "r_" + name[2:]
        elif name.startswith("r_"):
            assert mirrored == "l_" + name[2:]
        else:
            assert mirrored == name, "waist joints mirror onto themselves"


def test_mirror_negates_roll_and_yaw_only():
    for index, name in enumerate(C.JOINT_NAMES):
        expected = -1 if name.endswith(("_x_joint", "_z_joint")) else 1
        assert C.DOF_MIRROR_SIGNS[index] == expected, name


def test_mirror_head_and_tail_sign_widths():
    assert len(C.OBS_HEAD_MIRROR_SIGNS) == C.NUM_OBS_HEAD
    assert len(C.CRITIC_TAIL_MIRROR_SIGNS) == C.NUM_PRIVILEGED_OBS
    assert all(s in (-1, 1) for s in C.OBS_HEAD_MIRROR_SIGNS)
    assert all(s in (-1, 1) for s in C.CRITIC_TAIL_MIRROR_SIGNS)


def test_lateral_channels_flip_under_mirroring():
    """Mirroring about the sagittal plane maps y to -y.

    True vectors (gravity, base linear velocity) therefore transform as
    diag(1, -1, 1), while pseudovectors (angular velocity) pick up an extra
    sign and transform as diag(-1, 1, -1). Getting these two confused is the
    classic way to silently corrupt symmetry augmentation.
    """
    vx, vy, wyaw, height = C.OBS_HEAD_MIRROR_SIGNS[:4]
    assert (vx, vy, wyaw, height) == (1, -1, -1, 1)
    ang_vel = C.OBS_HEAD_MIRROR_SIGNS[4:7]
    gravity = C.OBS_HEAD_MIRROR_SIGNS[7:10]
    assert ang_vel == (-1, 1, -1), "angular velocity is a pseudovector"
    assert gravity == (1, -1, 1), "projected gravity is a true vector"
    assert C.CRITIC_TAIL_MIRROR_SIGNS == (1, -1, 1)


# --------------------------------------------------------------------------
# Geometry and spawn height
# --------------------------------------------------------------------------
def _urdf_chain_offset(urdf_root, from_link, to_link):
    """Sum the joint origin offsets along the chain, at zero joint angles."""
    child_to_parent = {}
    for joint in urdf_root.findall("joint"):
        origin = joint.find("origin")
        raw = origin.get("xyz") if origin is not None else None
        xyz = tuple(float(v) for v in (raw or "0 0 0").split())
        child_to_parent[joint.find("child").get("link")] = (
            joint.find("parent").get("link"),
            xyz,
        )

    total = [0.0, 0.0, 0.0]
    link = to_link
    while link != from_link:
        assert link in child_to_parent, f"{to_link} does not descend from {from_link}"
        parent, xyz = child_to_parent[link]
        total = [t + v for t, v in zip(total, xyz)]
        link = parent
    return tuple(total)


def test_kinematic_lengths_match_the_urdf(urdf_root):
    """Guards the constants used by torso_height_above_soles()."""
    _, _, dz = _urdf_chain_offset(urdf_root, "torso_link", "l_ankle_x_link")
    straight_leg_drop = C.TORSO_TO_HIP_Z + C.THIGH_LENGTH + C.SHIN_LENGTH
    assert abs(dz) == pytest.approx(straight_leg_drop, abs=1e-6)


def test_thigh_and_shin_lengths(urdf_root):
    _, _, thigh = _urdf_chain_offset(urdf_root, "l_hip_z_link", "l_knee_y_link")
    _, _, shin = _urdf_chain_offset(urdf_root, "l_knee_y_link", "l_ankle_y_link")
    assert abs(thigh) == pytest.approx(C.THIGH_LENGTH, abs=1e-6)
    assert abs(shin) == pytest.approx(C.SHIN_LENGTH, abs=1e-6)


def test_default_pose_torso_height():
    """ELF3 stands ~1.01 m tall at its default joint pose."""
    assert C.torso_height_above_soles() == pytest.approx(1.013, abs=1e-3)


def test_straight_leg_height_is_the_maximum():
    upright = C.torso_height_above_soles(hip_pitch=0.0, knee_pitch=0.0)
    assert upright == pytest.approx(1.0635, abs=1e-3)
    assert upright > C.torso_height_above_soles()


def test_spawn_height_equals_forward_kinematics():
    """The regression guard for the bug that killed the previous attempt.

    G1's 0.75 m spawn was inherited unchanged. ELF3's torso sits 1.013 m above
    its soles at the default pose, so spawning at 0.75 m buries the feet 26 cm
    underground; contact resolution then ejects the robot and every episode
    terminates within ~38 steps regardless of training.
    """
    assert C.DEFAULT_BASE_HEIGHT == pytest.approx(C.torso_height_above_soles())
    assert C.DEFAULT_BASE_HEIGHT > 1.0, "spawn height must clear the legs"
    assert C.DEFAULT_BASE_HEIGHT != pytest.approx(0.75, abs=0.01)


def test_height_command_range_is_reachable():
    """Commands must lie within what the legs can actually reach."""
    lowest = C.torso_height_above_soles(hip_pitch=-1.2, knee_pitch=2.07)
    highest = C.torso_height_above_soles(hip_pitch=0.0, knee_pitch=0.0)
    assert C.MIN_HEIGHT_COMMAND >= lowest - 1e-3
    assert C.MAX_HEIGHT_COMMAND <= highest
    assert C.MIN_HEIGHT_COMMAND < C.MAX_HEIGHT_COMMAND


def test_default_height_is_inside_the_command_range():
    assert C.MIN_HEIGHT_COMMAND <= C.DEFAULT_BASE_HEIGHT <= C.MAX_HEIGHT_COMMAND


def test_height_curriculum_starts_from_standing():
    """The previous ladder began at 0.34 m, the most contorted pose available.

    0.34 m needs about -2.88 rad of hip pitch, so that curriculum started at
    its hardest point and got easier -- backwards. Crouching must be the
    endpoint, not the entry point.
    """
    deep_crouch = 0.34
    assert C.MIN_HEIGHT_COMMAND > deep_crouch


# --------------------------------------------------------------------------
# Assets, actuation, sim2sim
# --------------------------------------------------------------------------
def test_every_referenced_mesh_exists(urdf_root):
    referenced = {
        mesh.get("filename")
        for mesh in urdf_root.iter("mesh")
        if mesh.get("filename")
    }
    assert referenced, "URDF references no meshes"
    for filename in sorted(referenced):
        assert not filename.startswith("/"), f"absolute mesh path: {filename}"
        assert (C.ASSET_ROOT / filename).is_file(), f"missing mesh {filename}"


def test_named_bodies_exist_in_the_urdf(urdf_root):
    links = {link.get("name") for link in urdf_root.findall("link")}
    expected = (
        C.FOOT_BODY_NAMES
        + C.KNEE_BODY_NAMES
        + C.HAND_BODY_NAMES
        + [C.IMU_BODY_NAME, C.TORSO_BODY_NAME]
    )
    for name in expected:
        assert name in links, f"{name} is not a link in the URDF"


def test_joint_limits_are_finite_and_ordered(urdf_root):
    for joint in urdf_root.findall("joint"):
        if joint.get("type") not in ACTUATED_JOINT_TYPES:
            continue
        limit = joint.find("limit")
        assert limit is not None, joint.get("name")
        lower = float(limit.get("lower"))
        upper = float(limit.get("upper"))
        effort = float(limit.get("effort"))
        velocity = float(limit.get("velocity"))
        assert math.isfinite(lower) and math.isfinite(upper)
        assert lower < upper, joint.get("name")
        assert effort > 0.0, joint.get("name")
        assert velocity > 0.0, joint.get("name")


def test_total_mass_matches_the_urdf(urdf_root):
    total = 0.0
    for link in urdf_root.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            total += float(inertial.find("mass").get("value"))
    assert total == pytest.approx(C.TOTAL_MASS, abs=0.01)


def _regex_covers(patterns, names):
    import re

    return {
        name
        for name in names
        if any(re.fullmatch(pattern, name) for pattern in patterns)
    }


def test_gain_patterns_cover_every_joint_exactly_once():
    import re

    stiffness = {**C.UPPER_BODY_STIFFNESS, **C.LEG_STIFFNESS}
    damping = {**C.UPPER_BODY_DAMPING, **C.LEG_DAMPING}
    for table, label in ((stiffness, "stiffness"), (damping, "damping")):
        for name in C.JOINT_NAMES:
            matches = [p for p in table if re.fullmatch(p, name)]
            assert len(matches) == 1, f"{label}: {name} matched {matches}"


def test_leg_and_upper_gain_tables_target_their_own_groups():
    assert _regex_covers(C.LEG_STIFFNESS, C.JOINT_NAMES) == set(C.POLICY_JOINT_NAMES)
    assert _regex_covers(C.UPPER_BODY_STIFFNESS, C.JOINT_NAMES) == set(
        C.UPPER_BODY_JOINT_NAMES
    )


def test_armature_is_set_for_every_joint():
    """Zero rotor inertia at kp=300..500 with dt=1/200 is unstable."""
    import re

    for name in C.JOINT_NAMES:
        matches = [p for p in C.ARMATURE if re.fullmatch(p, name)]
        assert len(matches) == 1, f"{name} matched {matches}"
        assert C.ARMATURE[matches[0]] > 0.0, name


def test_per_group_regex_dicts_match_only_their_own_group():
    """Isaac rejects an actuator whose gain dict has an unmatched pattern.

    Every pattern handed to an actuator group must match at least one joint
    *within that group*, so the leg and upper-body dicts cannot be merged.
    """
    import re

    groups = (
        (C.POLICY_JOINT_NAMES, (C.LEG_STIFFNESS, C.LEG_DAMPING, C.LEG_ARMATURE)),
        (
            C.UPPER_BODY_JOINT_NAMES,
            (C.UPPER_BODY_STIFFNESS, C.UPPER_BODY_DAMPING, C.UPPER_BODY_ARMATURE),
        ),
    )
    for joint_names, tables in groups:
        for table in tables:
            for pattern in table:
                matched = [n for n in joint_names if re.fullmatch(pattern, n)]
                assert matched, f"{pattern!r} matches no joint in this group"
            for name in joint_names:
                matches = [p for p in table if re.fullmatch(p, name)]
                assert len(matches) == 1, f"{name} matched {matches}"


def test_merged_armature_agrees_with_the_split_tables():
    assert C.ARMATURE == {**C.UPPER_BODY_ARMATURE, **C.LEG_ARMATURE}
    assert not set(C.LEG_ARMATURE) & set(C.UPPER_BODY_ARMATURE)


def test_mujoco_locked_joints_are_not_trained_joints():
    """The vendor MJCF has 31 actuators; the 3 extras must be held at zero."""
    assert len(C.MUJOCO_LOCKED_JOINTS) == 3
    assert "waist_x_joint" in C.MUJOCO_LOCKED_JOINTS
    for name in C.MUJOCO_LOCKED_JOINTS:
        assert name not in C.JOINT_NAMES


def test_command_scale_matches_observation_scales():
    """sim2sim must reuse these; upstream's g1.yaml halves them and drifts."""
    assert C.COMMAND_SCALE == (C.LIN_VEL_SCALE, C.LIN_VEL_SCALE, C.ANG_VEL_SCALE)
    assert C.ANG_VEL_SCALE == 0.5, "upstream's 0.25 in g1.yaml is a real bug"
