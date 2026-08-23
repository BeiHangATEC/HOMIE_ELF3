# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Elf3Dof31Cfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.96]
        default_joint_angles = {
            "waist_z_joint": 0.0,
            "waist_y_joint": 0.0,
            "waist_x_joint": 0.0,
            "head_z_joint": 0.0,
            "head_y_joint": 0.0,

            "l_hip_z_joint": 0.0,
            "l_hip_x_joint": 0.0,
            "l_hip_y_joint": -0.10,
            "l_knee_y_joint": 0.30,
            "l_ankle_y_joint": -0.20,
            "l_ankle_x_joint": 0.0,

            "r_hip_z_joint": 0.0,
            "r_hip_x_joint": 0.0,
            "r_hip_y_joint": -0.10,
            "r_knee_y_joint": 0.30,
            "r_ankle_y_joint": -0.20,
            "r_ankle_x_joint": 0.0,

            "l_shoulder_z_joint": 0.0,
            "l_shoulder_x_joint": 0.0,
            "l_shoulder_y_joint": 0.0,
            "l_elbow_y_joint": 0.0,
            "l_wrist_z_joint": 0.0,
            "l_wrist_y_joint": 0.0,
            "l_wrist_x_joint": 0.0,

            "r_shoulder_z_joint": 0.0,
            "r_shoulder_x_joint": 0.0,
            "r_shoulder_y_joint": 0.0,
            "r_elbow_y_joint": 0.0,
            "r_wrist_z_joint": 0.0,
            "r_wrist_y_joint": 0.0,
            "r_wrist_x_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = "M"
        stiffness = {
            "hip_z": 90,
            "hip_x": 90,
            "hip_y": 100,
            "knee": 140,
            "ankle": 35,
            "waist": 180,
            "head": 20,
            "shoulder": 80,
            "elbow": 50,
            "wrist": 20,
        }
        damping = {
            "hip_z": 2.0,
            "hip_x": 2.0,
            "hip_y": 2.5,
            "knee": 4.0,
            "ankle": 1.5,
            "waist": 4.0,
            "head": 0.5,
            "shoulder": 2.0,
            "elbow": 1.0,
            "wrist": 0.5,
        }
        action_scale = 0.25
        decimation = 4
        hip_reduction = 1.0

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 0.5
        num_commands = 5
        resampling_time = 4.0
        heading_command = False
        heading_to_ang_vel = False
        independent_height_velocity_sampling = True
        command_deadzone = 0.05
        stage1_curriculum = False
        stage1_total_policy_steps = 1
        stage1_curriculum_segments = [
            (0.20, 0.0, 0.0),
            (0.50, -0.08, 0.05),
            (0.80, -0.18, 0.10),
            (1.00, -0.18, 0.15),
        ]
        stage6_curriculum = False
        stage6_total_policy_steps = 1
        stage6_cycle_period_s = 10.0
        stage6_curriculum_segments = [
            (0.20, 1.00, 0.00, 0.05),
            (0.50, 0.70, 0.15, 0.05),
            (0.80, 0.50, 0.25, 0.08),
            (1.00, 0.40, 0.30, 0.08),
        ]
        stage7_curriculum = False
        stage7_total_policy_steps = 1
        stage7_cycle_period_s = 10.0
        stage7_fixed_heights = [0.974, 0.884, 0.794]
        stage7_cycle_fraction = 0.70
        stage7_curriculum_segments = [
            (0.20, 0.00, 0.03),
            (0.50, 0.00, 0.03),
            (1.00, 0.70, 0.05),
        ]
        stage8_curriculum = False
        stage8_total_policy_steps = 1
        stage8_cycle_period_s = 10.0
        stage8_initial_height = 0.945
        stage8_height_transition_end = 0.50
        stage8_curriculum_segments = [
            (0.30, 0.00, 0.00, 0.00, 0.00),
            (0.50, 0.50, 0.00, 0.00, 0.04),
            (0.70, 0.65, 0.10, 0.20, 0.09),
            (0.85, 0.60, 0.20, 0.35, 0.14),
            (1.00, 0.50, 0.30, 0.50, 0.18),
        ]
        stage9_curriculum = False
        stage9_total_policy_steps = 1
        stage9_cycle_period_s = 10.0
        stage9_curriculum_segments = [
            (0.50, 0.00),
            (0.75, 0.074),
            (1.00, 0.18),
        ]
        stage9_reward_ramp_end = 0.60
        stage9_reset_velocity_segments = [
            (0.50, 0.03),
            (0.75, 0.05),
            (1.00, 0.10),
        ]

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.0, 0.3]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [-3.14, 3.14]
            height = [-0.18, 0.0]

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/elf3_dof31/urdf/elf3.urdf"
        name = "elf3_dof31"
        foot_name = "ankle_x_link"
        left_foot_name = "l_ankle_x_link"
        right_foot_name = "r_ankle_x_link"
        left_hand_name = "l_wrist_x_link"
        right_hand_name = "r_wrist_x_link"
        action_joints = [
            "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint", "l_knee_y_joint", "l_ankle_y_joint", "l_ankle_x_joint",
            "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint", "r_knee_y_joint", "r_ankle_y_joint", "r_ankle_x_joint",
        ]
        penalize_contacts_on = ["hip", "knee", "waist", "shoulder", "elbow", "wrist", "head"]
        terminate_after_contacts_on = ["torso"]
        curriculum_joints = []
        left_leg_joints = ["l_hip_z_joint", "l_hip_x_joint", "l_hip_y_joint", "l_knee_y_joint", "l_ankle_y_joint", "l_ankle_x_joint"]
        right_leg_joints = ["r_hip_z_joint", "r_hip_x_joint", "r_hip_y_joint", "r_knee_y_joint", "r_ankle_y_joint", "r_ankle_x_joint"]
        left_hip_joints = ["l_hip_x_joint", "l_hip_y_joint", "l_hip_z_joint"]
        right_hip_joints = ["r_hip_x_joint", "r_hip_y_joint", "r_hip_z_joint"]
        hip_pitch_joints = ["r_hip_y_joint", "l_hip_y_joint"]
        knee_joints = ["l_knee_y_joint", "r_knee_y_joint"]
        ankle_joints = ["l_ankle_x_joint", "r_ankle_x_joint"]
        upper_body_link = "torso_link"
        imu_link = "torso_link"
        pelvis_body_name = "waist_z_link"
        knee_names = ["l_knee_y_link", "l_hip_y_link", "r_knee_y_link", "r_hip_y_link"]
        static_knee_names = ["l_knee_y_link", "r_knee_y_link"]
        self_collisions = 1
        flip_visual_attachments = False
        ankle_sole_distance = 0.02

    class domain_rand(LeggedRobotCfg.domain_rand):
        use_random = False

        randomize_joint_injection = use_random
        joint_injection_range = [-0.03, 0.03]

        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.03, 0.03]

        randomize_payload_mass = use_random
        payload_mass_range = [-2.0, 5.0]
        hand_payload_mass_range = [-0.1, 0.2]

        randomize_com_displacement = False
        com_displacement_range = [-0.05, 0.05]

        randomize_body_displacement = use_random
        body_displacement_range = [-0.03, 0.03]

        randomize_link_mass = use_random
        link_mass_range = [0.9, 1.1]

        randomize_friction = use_random
        friction_range = [0.4, 1.5]

        randomize_restitution = use_random
        restitution_range = [0.0, 0.5]

        randomize_kp = use_random
        kp_range = [0.9, 1.1]

        randomize_kd = use_random
        kd_range = [0.9, 1.1]

        randomize_initial_joint_pos = use_random
        initial_joint_pos_scale = [0.9, 1.1]
        initial_joint_pos_offset = [-0.05, 0.05]

        push_robots = use_random
        push_interval_s = 4
        upper_interval_s = 1
        max_push_vel_xy = 0.3

        init_upper_ratio = 0.0
        randomize_upper_actions = False
        delay = use_random

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            termination = -10.0
            collision = -1.0
            tracking_x_vel = 3.0
            tracking_y_vel = 2.0
            tracking_ang_vel = 3.0
            lin_vel_z = -1.2
            ang_vel_xy = -0.08
            orientation = -4.0
            action_rate = -0.015
            tracking_base_height = 4.0
            deviation_hip_joint = -0.2
            deviation_ankle_joint = -0.3
            deviation_knee_joint = 0.0
            dof_acc = -5e-7
            dof_pos_limits = -2.0
            feet_air_time = 0.05
            feet_clearance = -0.25
            feet_distance_lateral = 0.5
            knee_distance_lateral = 1.0
            feet_ground_parallel = -1.2
            feet_parallel = -2.0
            smoothness = -0.04
            joint_power = -2e-5
            feet_stumble = -1.5
            torques = -2.5e-6
            dof_vel = -1e-4
            dof_vel_limits = -2e-3
            torque_limits = -0.1
            no_fly = 0.75
            joint_tracking_error = -0.1
            feet_slip = -1.0
            feet_contact_forces = -0.00025
            contact_momentum = 2.5e-4
            action_vanish = -1.0
            stand_still = -1.0
            tracking_pelvis_height = 0.0
            upright_static = 0.0
            leg_symmetry_static = 0.0
            knee_distance_static = 0.0
            stand_posture = 0.0
            feet_position_lock = 0.0
            crouch_feet_xy_velocity = 0.0
            natural_stance = 0.0
            feet_spacing_static = 0.0

        only_positive_rewards = False
        tracking_sigma = 0.25
        tracking_base_height_exp = 12.0
        tracking_pelvis_height_exp = 16.0
        pelvis_height_command_offset = 0.2194
        stand_still_command_threshold = 1e-3
        upright_waist_weights = [1.0, 1.0]
        leg_symmetry_weights = [1.0, 0.5, 0.5, 1.0, 1.0, 0.5]
        knee_distance_height_range = [0.794, 0.974]
        knee_distance_min_range = [0.18, 0.24]
        knee_distance_max = 0.45
        stand_posture_fade_height = [0.894, 0.974]
        stand_posture_weights = [1.0, 0.25, 0.25, 1.0, 1.0, 0.25]
        stage9_height_gate = [0.894, 0.954]
        natural_stance_joint_names = [
            "l_hip_x_joint", "l_hip_z_joint",
            "r_hip_x_joint", "r_hip_z_joint",
        ]
        natural_stance_weights = [1.0, 0.5, 1.0, 0.5]
        feet_spacing_soft_max = 0.48
        soft_dof_pos_limit = 0.975
        soft_dof_vel_limit = 0.80
        soft_torque_limit = 0.95
        base_height_target = 0.974
        crouch_height_threshold = 0.884
        max_contact_force = 400.0
        least_feet_distance = 0.18
        least_feet_distance_lateral = 0.20
        most_feet_distance_lateral = 0.35
        most_knee_distance_lateral = 0.35
        least_knee_distance_lateral = 0.20
        clearance_height_target = 0.10

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 12
        num_dofs = 31
        num_one_step_observations = 2 * num_dofs + 10 + num_actions
        num_one_step_privileged_obs = num_one_step_observations + 3
        num_actor_history = 6
        num_critic_history = 1
        num_observations = num_actor_history * num_one_step_observations
        num_privileged_obs = num_critic_history * num_one_step_privileged_obs
        action_curriculum = False
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.02
            dof_vel = 2.0
            lin_vel = 0.1
            ang_vel = 0.5
            gravity = 0.05
            height_measurements = 0.1


class Elf3Dof31CfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        use_flip = False
        entropy_coef = 0.01
        symmetry_scale = 0.0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "HIMActorCritic"
        algorithm_class_name = "HIMPPO"
        save_interval = 200
        num_steps_per_env = 50
        max_iterations = 1000
        run_name = ""
        experiment_name = "elf3_dof31"
        wandb_project = ""
        logger = "tensorboard"
        wandb_user = ""
