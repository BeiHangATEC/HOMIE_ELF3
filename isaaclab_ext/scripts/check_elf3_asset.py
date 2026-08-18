"""Load the ELF3 articulation and verify it against the canonical joint table.

This is the first check that touches Isaac Sim. It confirms the converted USD
produces the articulation the environment expects: 28 actuated joints with the
canonical names, positive masses, positive-definite inertias, the named bodies
the rewards rely on, and -- most importantly -- that the robot settles at the
height forward kinematics predicts.

    python isaaclab_ext/scripts/check_elf3_asset.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_SOURCE = REPO_ROOT / "isaaclab_ext" / "source" / "openhomie_isaaclab"
if str(EXTENSION_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXTENSION_SOURCE))

from isaaclab.app import AppLauncher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # In *simulation* steps (dt = 1/200), not policy steps. 400 is 2 s, which
    # is enough to catch a spawn-height error without asserting that open-loop
    # PD can balance the robot -- it cannot, and that is not the asset's fault.
    parser.add_argument("--settle_steps", type=int, default=400)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()
app_launcher = AppLauncher(ARGS)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from openhomie_isaaclab import elf3_constants as C  # noqa: E402
from openhomie_isaaclab.tasks.locomotion.elf3.elf3_articulation import (  # noqa: E402
    build_elf3_articulation_cfg,
)


class CheckFailed(RuntimeError):
    """Raised when the loaded asset disagrees with the canonical table."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def _resolve_gains(
    robot: Articulation, table: dict[str, float], joint_names
) -> torch.Tensor:
    """Expand a regex-keyed gain table into a tensor in `joint_names` order."""
    import re

    values = []
    for name in joint_names:
        matches = [v for pattern, v in table.items() if re.fullmatch(pattern, name)]
        _check(len(matches) == 1, f"{name} matched {len(matches)} gain patterns")
        values.append(matches[0])
    return torch.tensor(
        [values], dtype=torch.float32, device=robot.device
    ).repeat(robot.num_instances, 1)


def check_joint_names(robot: Articulation) -> None:
    runtime = list(robot.joint_names)
    _check(
        len(runtime) == C.NUM_ROBOT_DOFS,
        f"expected {C.NUM_ROBOT_DOFS} joints, articulation has {len(runtime)}",
    )
    _check(
        sorted(runtime) == sorted(C.JOINT_NAMES),
        "joint names differ from the canonical table:\n"
        f"  only in USD:   {sorted(set(runtime) - set(C.JOINT_NAMES))}\n"
        f"  only in table: {sorted(set(C.JOINT_NAMES) - set(runtime))}",
    )
    _check(
        "waist_x_joint" not in runtime,
        "waist_x_joint is actuated in the USD but ELF3 keeps it fixed",
    )
    print(f"joint names match the canonical table ({len(runtime)} DOFs)")

    if runtime == list(C.JOINT_NAMES):
        print("  runtime order equals canonical order")
        return
    print("  runtime order differs from canonical; the env must permute:")
    for canonical, name in enumerate(C.JOINT_NAMES):
        runtime_index = runtime.index(name)
        if canonical != runtime_index:
            print(f"    canonical[{canonical:2d}] {name:20s} <- runtime[{runtime_index:2d}]")


def check_actuator_groups(robot: Articulation) -> None:
    """Legs must receive no Isaac-side drive; mixed control computes it."""
    legs = robot.actuators["legs"]
    upper = robot.actuators["upper_body"]
    _check(
        len(legs.joint_names) == C.NUM_POLICY_ACTIONS,
        f"leg group has {len(legs.joint_names)} joints, expected {C.NUM_POLICY_ACTIONS}",
    )
    _check(
        len(upper.joint_names) == C.NUM_UPPER_BODY_DOFS,
        f"upper group has {len(upper.joint_names)}, expected {C.NUM_UPPER_BODY_DOFS}",
    )
    _check(
        not set(legs.joint_names) & set(upper.joint_names),
        "actuator groups overlap",
    )
    _check(
        float(legs.stiffness.abs().max()) == 0.0
        and float(legs.damping.abs().max()) == 0.0,
        "leg actuators must have zero stiffness and damping so the environment"
        " can apply its own PD torques without Isaac fighting it",
    )
    _check(
        float(upper.stiffness.min()) > 0.0,
        "upper-body actuators need a positive position-drive stiffness",
    )
    print(
        f"actuator groups: {len(legs.joint_names)} legs (no Isaac drive) + "
        f"{len(upper.joint_names)} upper body (position drive)"
    )


def check_bodies(robot: Articulation) -> None:
    names = list(robot.body_names)
    required = (
        list(C.FOOT_BODY_NAMES)
        + list(C.KNEE_BODY_NAMES)
        + list(C.HAND_BODY_NAMES)
        + [C.IMU_BODY_NAME, C.TORSO_BODY_NAME]
    )
    missing = [name for name in required if name not in names]
    _check(not missing, f"articulation is missing bodies: {missing}")
    print(f"all {len(required)} named bodies present ({len(names)} bodies total)")


def check_physics(robot: Articulation) -> None:
    masses = robot.root_physx_view.get_masses()[0]
    _check(bool(torch.all(masses >= 0.0)), "found a negative link mass")

    phantom = {
        robot.body_names[i]: float(masses[i])
        for i in range(len(robot.body_names))
        if robot.body_names[i] in C.MASSLESS_LINK_NAMES
        and float(masses[i]) > 10 * C.MASSLESS_LINK_MASS
    }
    _check(
        not phantom,
        "PhysX assigned default mass to links the URDF leaves massless: "
        f"{phantom}. These are sensor frames (one of them is imu_link, the "
        "observation reference frame); the environment must zero them via "
        "massless_link_event().",
    )

    total = float(masses.sum())
    _check(
        abs(total - C.TOTAL_MASS) < 0.5,
        f"total mass {total:.3f} kg disagrees with the URDF's {C.TOTAL_MASS} kg",
    )
    print(f"total mass {total:.3f} kg (URDF says {C.TOTAL_MASS} kg)")

    inertias = robot.root_physx_view.get_inertias()[0].reshape(-1, 3, 3)
    for index, inertia in enumerate(inertias):
        if float(masses[index]) == 0.0:
            continue  # massless sensor frames legitimately carry no inertia
        symmetric = 0.5 * (inertia + inertia.T)
        eigenvalues = torch.linalg.eigvalsh(symmetric.double())
        _check(
            bool(torch.all(eigenvalues > -1e-9)),
            f"inertia of {robot.body_names[index]} is not positive semidefinite:"
            f" eigenvalues {eigenvalues.tolist()}",
        )
    print("all inertia tensors positive semidefinite")


def check_limits(robot: Articulation) -> None:
    limits = robot.root_physx_view.get_dof_limits()[0]
    _check(bool(torch.all(torch.isfinite(limits))), "non-finite joint limit")
    _check(bool(torch.all(limits[:, 1] > limits[:, 0])), "joint limit upper <= lower")
    print("joint position limits finite and ordered")

    # Read the effort limits from the actuators, not from the PhysX view: the
    # URDF converter leaves the USD drive maxForce at 1e9, so the view reports
    # that instead of the URDF's real values.
    urdf_max = max(C.EFFORT_LIMIT.values())
    for group_name, group in robot.actuators.items():
        effort = group.effort_limit
        _check(
            bool(torch.all(torch.isfinite(effort))),
            f"{group_name}: non-finite effort limit",
        )
        _check(bool(torch.all(effort > 0.0)), f"{group_name}: non-positive effort")
        _check(
            float(effort.max()) <= urdf_max + 1e-6,
            f"{group_name}: effort limit {float(effort.max()):.1f} N m exceeds the "
            f"URDF maximum {urdf_max:.1f} N m -- the limit did not come from the "
            "URDF and the legs have effectively unbounded torque",
        )
        print(
            f"  {group_name}: effort limits in "
            f"[{float(effort.min()):.1f}, {float(effort.max()):.1f}] N m"
        )


def check_standing_height(
    robot: Articulation, sim: SimulationContext, settle_steps: int
) -> None:
    """Confirm the spawn height matches forward kinematics.

    What this checks: the feet start on the ground rather than inside it, and
    the measured torso-above-soles distance agrees with FK. That is what caught
    the previous attempt's inherited 0.75 m spawn, which buried the feet 26 cm
    underground and pinned every episode at ~38 steps.

    What this does NOT check: that the robot stays standing. ELF3's default pose
    is only marginally stable open-loop -- the CoM sits about 3.5 mm from the
    sole centre and the soles are narrow -- so PD-holding the default pose
    topples after roughly a second in both Isaac Lab and MuJoCo. Staying upright
    is the policy's job, not the asset's, so asserting it here would be wrong.
    """
    torso_id = robot.body_names.index(C.TORSO_BODY_NAME)
    foot_ids = [robot.body_names.index(name) for name in C.FOOT_BODY_NAMES]

    leg_ids, _ = robot.find_joints(list(C.POLICY_JOINT_NAMES), preserve_order=True)
    leg_ids = torch.tensor(leg_ids, dtype=torch.long, device=robot.device)
    kp = _resolve_gains(robot, C.LEG_STIFFNESS, C.POLICY_JOINT_NAMES)
    kd = _resolve_gains(robot, C.LEG_DAMPING, C.POLICY_JOINT_NAMES)
    effort_cap = _resolve_gains(robot, C.LEG_EFFORT_LIMIT, C.POLICY_JOINT_NAMES)

    default_pos = robot.data.default_joint_pos.clone()
    leg_target = default_pos[:, leg_ids]
    dt = sim.get_physics_dt()

    # Sample early, while the pose is still the one we are validating.
    early_measurement = None
    early_at = min(20, settle_steps)

    for step in range(settle_steps):
        torque = kp * (leg_target - robot.data.joint_pos[:, leg_ids]) - kd * (
            robot.data.joint_vel[:, leg_ids]
        )
        robot.set_joint_effort_target(
            torch.clamp(torque, -effort_cap, effort_cap), joint_ids=leg_ids
        )
        robot.set_joint_position_target(default_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        if step + 1 == early_at:
            torso_z = float(robot.data.body_pos_w[0, torso_id, 2])
            foot_z = float(robot.data.body_pos_w[0, foot_ids, 2].min())
            early_measurement = (torso_z, foot_z)

    torso_z, foot_z = early_measurement
    predicted = C.torso_height_above_soles()
    measured = torso_z - (foot_z + C.SOLE_CENTER_OFFSET[2])
    final_z = float(robot.data.body_pos_w[0, torso_id, 2])

    print(f"after {early_at} sim steps ({early_at * dt:.2f} s) of PD-holding:")
    print(f"  torso height above world    : {torso_z:.4f} m")
    print(f"  lowest ankle link           : {foot_z:.4f} m")
    print(f"  torso above soles (measured): {measured:.4f} m")
    print(f"  torso above soles (FK)      : {predicted:.4f} m")
    print(f"  spawn height configured     : {C.DEFAULT_BASE_HEIGHT:.4f} m")
    print(
        f"after {settle_steps} sim steps ({settle_steps * dt:.2f} s): "
        f"torso at {final_z:.4f} m"
    )
    if final_z < 0.5 * predicted:
        print(
            "  (expected: open-loop PD cannot balance this pose, so it topples;"
            " balancing is the policy's job)"
        )

    _check(
        foot_z > -0.02,
        f"feet are {-foot_z:.4f} m below the ground plane -- the robot is "
        "spawning inside the floor (this is the bug that broke the previous "
        "attempt; check init_state.pos against torso_height_above_soles())",
    )
    _check(
        abs(measured - predicted) < 0.05,
        f"standing height {measured:.4f} m disagrees with forward kinematics "
        f"{predicted:.4f} m by more than 5 cm",
    )
    print("spawn height agrees with forward kinematics")


def _zero_massless_links(robot: Articulation) -> None:
    """Undo PhysX's default 1 kg on links the URDF leaves without inertial."""
    view = robot.root_physx_view
    masses = view.get_masses()
    indices = [
        robot.body_names.index(name)
        for name in C.MASSLESS_LINK_NAMES
        if name in robot.body_names
    ]
    if not indices:
        return
    masses[:, indices] = C.MASSLESS_LINK_MASS
    view.set_masses(masses, torch.arange(robot.num_instances))


def main() -> int:
    sim = SimulationContext(
        SimulationCfg(dt=C.SIM_DT, device=ARGS.device, gravity=(0.0, 0.0, -9.81))
    )
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2000.0)
    )

    robot = Articulation(build_elf3_articulation_cfg(prim_path="/World/Robot"))
    sim.reset()

    # The environment does this through an event term; standalone there is no
    # event manager, so apply the same correction directly.
    _zero_massless_links(robot)

    checks = (
        ("joint table", lambda: check_joint_names(robot)),
        ("actuator groups", lambda: check_actuator_groups(robot)),
        ("bodies", lambda: check_bodies(robot)),
        ("physics", lambda: check_physics(robot)),
        ("limits", lambda: check_limits(robot)),
        (
            "standing height",
            lambda: check_standing_height(robot, sim, ARGS.settle_steps),
        ),
    )

    failures = []
    for name, check in checks:
        print(f"\n--- {name} ---")
        try:
            check()
        except CheckFailed as exc:
            failures.append(name)
            print(f"FAILED: {exc}")

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED checks: {', '.join(failures)}")
        return 1
    print("all ELF3 asset checks passed")
    return 0


if __name__ == "__main__":
    # Isaac's simulation_app.close() terminates the process directly, which
    # would discard both the traceback and the exit code. Report before it.
    import traceback

    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        exit_code = 1
    finally:
        print(f"\ncheck_elf3_asset exit code: {exit_code}", flush=True)
        simulation_app.close()
    sys.exit(exit_code)
