"""Configuration for the 28-DOF ELF3 HOMIE direct environment."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from openhomie_isaaclab import elf3_constants as C

from .elf3_articulation import build_elf3_articulation_cfg, massless_link_event
from .elf3_homie_rewards import REWARD_SCALES


@configclass
class Elf3HomieEventCfg:
    massless_link_mass = massless_link_event()
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 3.0),
            "dynamic_friction_range": (0.1, 3.0),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 256,
            "make_consistent": True,
        },
    )
    non_torso_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names="^(?!torso_link$)(?!imu_link$)(?!mid360_link$)(?!d435i_link$)(?!torso_front_bottom_d435i_link$).*$",
            ),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    torso_payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=C.TORSO_BODY_NAME),
            "mass_distribution_params": (-5.0, 10.0),
            "operation": "add",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    hand_payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=C.HAND_BODY_NAMES),
            "mass_distribution_params": (-0.1, 0.3),
            "operation": "add",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=C.TORSO_BODY_NAME),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class Elf3HomieEnvCfg(DirectRLEnvCfg):
    episode_length_s = C.EPISODE_LENGTH_S
    decimation = C.DECIMATION
    action_space = C.NUM_POLICY_ACTIONS
    observation_space = C.num_actor_obs()
    state_space = C.num_critic_obs()

    sim: SimulationCfg = SimulationCfg(
        dt=C.SIM_DT,
        render_interval=C.DECIMATION,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True
    )
    robot = build_elf3_articulation_cfg()
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=C.SIM_DT,
        track_air_time=True,
    )
    events: Elf3HomieEventCfg | None = Elf3HomieEventCfg()

    clip_actions = 100.0
    clip_observations = 100.0
    add_noise = True
    noise_level = 1.0
    ang_vel_noise = 0.5
    gravity_noise = 0.05
    dof_pos_noise = 0.02
    dof_vel_noise = 2.0

    command_stage = "S0"
    command_resampling_time_s = 4.0
    use_random_commands = True
    crouch_min_height = C.MIN_HEIGHT_COMMAND
    crouch_focus_max_height = 0.90
    upper_body_resampling_time_s = 1.0
    upper_body_interpolation_steps = 50
    action_curriculum_ratio = 0.0
    freeze_upper_body = False

    enable_action_delay = True
    max_action_delay_substeps = C.DECIMATION - 1
    randomize_control = True
    joint_injection_range = (-0.05, 0.05)
    actuation_offset_range = (-0.05, 0.05)
    kp_factor_range = (0.9, 1.1)
    kd_factor_range = (0.9, 1.1)
    randomize_initial_state = True
    initial_joint_pos_scale = (0.8, 1.2)
    initial_joint_pos_offset = (-0.1, 0.1)
    initial_root_velocity_range = (-0.5, 0.5)

    reward_scales = dict(REWARD_SCALES)
    tracking_sigma = 0.25
    soft_dof_vel_limit = 0.80
    soft_torque_limit = 0.95
    max_contact_force = 400.0
    least_feet_distance_lateral = 0.2
    most_feet_distance_lateral = 0.35
    least_knee_distance_lateral = 0.2
    most_knee_distance_lateral = 0.35
    clearance_height_target = 0.14
    ankle_sole_distance = 0.02
    joint_power_history_length = 100
    torso_contact_threshold = 10.0
    terminate_on_torso_contact = True
    gravity_xy_termination = 0.8

    base_height_target = C.DEFAULT_BASE_HEIGHT
    lin_vel_x_range = (-0.8, 1.2)
    lin_vel_y_range = (-0.5, 0.5)
    ang_vel_yaw_range = (-0.8, 0.8)
    height_range = (C.MIN_HEIGHT_COMMAND - C.DEFAULT_BASE_HEIGHT, 0.0)
