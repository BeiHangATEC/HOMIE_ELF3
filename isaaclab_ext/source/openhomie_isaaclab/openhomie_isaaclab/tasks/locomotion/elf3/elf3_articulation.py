"""The ELF3 `ArticulationCfg`, built from the canonical constants.

Two actuator groups, and the split is deliberate: HOMIE's mixed control drives
the 12 leg joints by hand-computed effort, so those joints must receive no
drive from Isaac itself (stiffness = damping = 0), while the 16 upper-body
joints use a real implicit position drive fed by the pose curriculum.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg

from openhomie_isaaclab import elf3_constants as C


def massless_link_event() -> EventTermCfg:
    """Strip the phantom mass PhysX assigns to the sensor frames.

    `imu_link`, `mid360_link` and the two camera links have no `<inertial>` in
    the URDF, so PhysX gives each a default 1 kg -- 4 kg of mass the real robot
    does not have, one of it sitting on the IMU frame the observations use.
    """
    import isaaclab.envs.mdp as mdp

    return EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=list(C.MASSLESS_LINK_NAMES)
            ),
            "mass_distribution_params": (C.MASSLESS_LINK_MASS, C.MASSLESS_LINK_MASS),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": False,
        },
    )



def elf3_spawn_cfg() -> sim_utils.UsdFileCfg:
    """Spawn from the pre-converted USD.

    Converting the URDF at spawn time (as the previous attempt did) re-runs a
    34-mesh convex decomposition on every launch. Run
    `isaaclab_ext/scripts/convert_elf3_usd.py` to produce the USD.
    """
    if not C.USD_PATH.is_file():
        raise FileNotFoundError(
            f"ELF3 USD not found at {C.USD_PATH}.\n"
            "Run: python isaaclab_ext/scripts/convert_elf3_usd.py"
        )
    return sim_utils.UsdFileCfg(
        usd_path=str(C.USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    )


def build_elf3_articulation_cfg(
    prim_path: str = "/World/envs/env_.*/Robot",
) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=elf3_spawn_cfg(),
        init_state=ArticulationCfg.InitialStateCfg(
            # Derived from forward kinematics of the default joint pose, so the
            # feet start exactly on the ground rather than inside it.
            pos=(0.0, 0.0, C.DEFAULT_BASE_HEIGHT),
            joint_pos=dict(C.DEFAULT_JOINT_POS),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.975,
        actuators={
            # Effort and velocity limits are declared explicitly: the URDF
            # converter drops the URDF's <limit effort=...> values, so leaving
            # these as None yields a 1e9 N m limit and unbounded leg torque.
            "legs": IdealPDActuatorCfg(
                joint_names_expr=list(C.POLICY_JOINT_NAMES),
                stiffness=0.0,
                damping=0.0,
                armature=dict(C.LEG_ARMATURE),
                effort_limit=dict(C.LEG_EFFORT_LIMIT),
                velocity_limit=C.VELOCITY_LIMIT,
            ),
            "upper_body": ImplicitActuatorCfg(
                joint_names_expr=list(C.UPPER_BODY_JOINT_NAMES),
                stiffness=dict(C.UPPER_BODY_STIFFNESS),
                damping=dict(C.UPPER_BODY_DAMPING),
                armature=dict(C.UPPER_BODY_ARMATURE),
                effort_limit=dict(C.UPPER_BODY_EFFORT_LIMIT),
                velocity_limit=C.VELOCITY_LIMIT,
            ),
        },
    )
