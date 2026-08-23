from legged_gym.envs.g1.elf3_dof31_config import Elf3Dof31Cfg, Elf3Dof31CfgPPO


class Elf3Dof31R1Contract:
    body_height_definition = "torso_link/root origin above supporting sole-bottom plane"
    command_semantics = "commands[:, 4] is the absolute body-height command in metres"
    command_calibration_source = "ATEC validated policy interface"
    body_height_command_m = 0.974
    natural_standing_height_command_m = body_height_command_m
    calibrated_standing_height_m = body_height_command_m
    geometric_isaac_gym_diagnostic_m = 1.0555
    geometric_mujoco_diagnostic_m = 0.8545
    isaac_reset_root_world_z_m = 0.96
    mujoco_reset_root_world_z_m = 1.05552264
    isaac_reset_root_height_m = isaac_reset_root_world_z_m
    mujoco_reset_root_height_m = mujoco_reset_root_world_z_m
    ankle_sole_distance_m = 0.0411461020
    default_pose_rad = {"hip_y": -0.1, "knee_y": 0.3, "ankle_y": -0.2, "all_other_joints": 0.0}
    feet_distance_soft_window_m = [0.20, 0.35]
    knee_distance_soft_window_m = [0.20, 0.35]


class Elf3Dof31R1Cfg(Elf3Dof31Cfg):
    class init_state(Elf3Dof31Cfg.init_state):
        pos = [0.0, 0.0, Elf3Dof31R1Contract.isaac_reset_root_world_z_m]

    class asset(Elf3Dof31Cfg.asset):
        ankle_sole_distance = Elf3Dof31R1Contract.ankle_sole_distance_m

    class commands(Elf3Dof31Cfg.commands):
        curriculum = False
        stage1_curriculum = False
        stage6_curriculum = False
        stage7_curriculum = False
        stage8_curriculum = False
        stage9_curriculum = False
        fixed_absolute_body_height = True
        fixed_body_height_m = Elf3Dof31R1Contract.calibrated_standing_height_m
        reset_velocity = 0.03

        class ranges(Elf3Dof31Cfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            height = [0.0, 0.0]

    class domain_rand(Elf3Dof31Cfg.domain_rand):
        use_random = False
        randomize_joint_injection = False
        randomize_actuation_offset = False
        randomize_payload_mass = False
        randomize_com_displacement = False
        randomize_body_displacement = False
        randomize_link_mass = False
        randomize_friction = True
        friction_range = [0.7, 1.1]
        randomize_restitution = False
        randomize_kp = False
        randomize_kd = False
        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.98, 1.02]
        initial_joint_pos_offset = [-0.01, 0.01]
        push_robots = False
        randomize_upper_actions = False
        delay = False

    class rewards(Elf3Dof31Cfg.rewards):
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
            dof_acc = -5e-7
            dof_pos_limits = -2.0
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

        base_height_target = Elf3Dof31R1Contract.calibrated_standing_height_m
        pelvis_height_command_offset = 0.0
        least_feet_distance_lateral = Elf3Dof31R1Contract.feet_distance_soft_window_m[0]
        most_feet_distance_lateral = Elf3Dof31R1Contract.feet_distance_soft_window_m[1]
        least_knee_distance_lateral = Elf3Dof31R1Contract.knee_distance_soft_window_m[0]
        most_knee_distance_lateral = Elf3Dof31R1Contract.knee_distance_soft_window_m[1]

    class noise(Elf3Dof31Cfg.noise):
        add_noise = True
        noise_level = 0.2

    class env(Elf3Dof31Cfg.env):
        num_envs = 512


class Elf3Dof31R1CfgPPO(Elf3Dof31CfgPPO):
    class runner(Elf3Dof31CfgPPO.runner):
        resume = False
        save_interval = 100
        max_iterations = 200
        run_name = "r1_from_scratch"
        experiment_name = "elf3_dof31_r1"
