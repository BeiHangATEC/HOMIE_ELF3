# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class Elf3RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.01] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'l_hip_z_joint' : 0. ,
           'l_hip_x_joint' : 0,
           'l_hip_y_joint' : -0.1,
           'l_knee_y_joint' : 0.3,
           'l_ankle_y_joint' : -0.2,
           'l_ankle_x_joint' : 0,
           'r_hip_z_joint' : 0.,
           'r_hip_x_joint' : 0,
           'r_hip_y_joint' : -0.1,
           'r_knee_y_joint' : 0.3,
           'r_ankle_y_joint': -0.2,
           'r_ankle_x_joint' : 0,
            "waist_y_joint": 0.,
            "waist_z_joint": 0.,
            "l_shoulder_y_joint": 0.,
            "l_shoulder_x_joint": 0.,
            "l_shoulder_z_joint": 0.,
            "l_elbow_y_joint": 0.,
            "l_wrist_x_joint": 0.,
            "l_wrist_y_joint": 0.,
            "l_wrist_z_joint": 0.,
            "r_shoulder_y_joint": 0.,
            "r_shoulder_x_joint": -0.,#-0.3
            "r_shoulder_z_joint": 0.,
            "r_elbow_y_joint": 0.,#0.8
            "r_wrist_x_joint": 0.,
            "r_wrist_y_joint": 0.,
            "r_wrist_z_joint": 0.,
           
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'M'
          # PD Drive parameters:
        stiffness = {'hip_z': 100,
                     'hip_x': 100,
                     'hip_y': 100,
                     'knee': 150,
                     'ankle': 40,
                     
                     "waist": 300,
                     "shoulder": 200,
                     "wrist": 20,
                     "elbow": 100,
                     "hand": 10
                    
                     }  # [N*m/rad]
        damping = {  'hip_z': 2,
                     'hip_x': 2,
                     'hip_y': 2,
                     'knee': 4,
                     'ankle': 2,
                     "waist": 5,
                     "shoulder": 4,
                     "wrist": 0.5,
                     "elbow": 1,
                     "hand": 2
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0

    class commands( LeggedRobotCfg.commands ):
        curriculum = False # NOTE set True later
        max_curriculum = 1.4
        num_commands = 5 # lin_vel_x, lin_vel_y, ang_vel_yaw, heading, height
        resampling_time = 4. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        heading_to_ang_vel = False
        height_target = 1.01
        use_task_distribution = True
        height_tracking_probability = 0.5
        velocity_tracking_probability = 1.0 / 3.0
        standing_probability = 1.0 / 6.0
        height_sampling_bands = [
            [0.30, 0.50, 0.5],
            [0.50, 1.01, 0.5],
        ]
        max_abs_velocity_command = 0.5
        class ranges( LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
            heading = [-3.14, 3.14]
            height = [-0.71, 0.0]

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/elf3_description/urdf/elf3.urdf'
        name = "elf3"
        foot_name = "ankle_x"
        left_foot_name = "l_ankle_x_link"
        right_foot_name = "r_ankle_x_link"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ['torso']
        curriculum_joints = []
        policy_dof_names = [
            'l_hip_y_joint', 'l_hip_x_joint', 'l_hip_z_joint', 'l_knee_y_joint', 'l_ankle_y_joint', 'l_ankle_x_joint',
            'r_hip_y_joint', 'r_hip_x_joint', 'r_hip_z_joint', 'r_knee_y_joint', 'r_ankle_y_joint', 'r_ankle_x_joint',
        ]
        upper_body_dof_names = [
            'waist_y_joint', 'waist_z_joint',
            'l_shoulder_y_joint', 'l_shoulder_x_joint', 'l_shoulder_z_joint', 'l_elbow_y_joint', 'l_wrist_x_joint', 'l_wrist_y_joint', 'l_wrist_z_joint',
            'r_shoulder_y_joint', 'r_shoulder_x_joint', 'r_shoulder_z_joint', 'r_elbow_y_joint', 'r_wrist_x_joint', 'r_wrist_y_joint', 'r_wrist_z_joint',
        ]
        left_leg_joints = ['l_hip_z_joint', 'l_hip_x_joint', 'l_hip_y_joint', 'l_knee_y_joint', 'l_ankle_y_joint']
        right_leg_joints = ['r_hip_z_joint', 'r_hip_x_joint', 'r_hip_y_joint', 'r_knee_y_joint', 'r_ankle_y_joint']
        left_hip_joints = ['l_hip_x_joint', "l_hip_y_joint", "l_hip_z_joint"]
        right_hip_joints = ['r_hip_x_joint', "r_hip_y_joint", "r_hip_z_joint"]
        hip_pitch_joints = ['r_hip_y_joint', 'l_hip_y_joint']
        knee_joints = ['l_knee_y_joint', 'r_knee_y_joint']
        ankle_joints = ["l_ankle_x_joint", "r_ankle_x_joint"]
        upper_body_link = "torso_link"
        left_hand_name = "l_wrist_z_link"
        right_hand_name = "r_wrist_z_link"
        imu_link = "imu_link"
        knee_names = ["l_knee_y_link", "l_hip_z_link", "r_knee_y_link", "r_hip_z_link"]
        self_collisions = 1
        flip_visual_attachments = False
        ankle_sole_distance = 0.02

        
    class domain_rand(LeggedRobotCfg.domain_rand):
        upper_body_joint_position_ranges = {
            "waist_y_joint": [-0.50, 0.50],
        }
        upper_body_locked_joint_positions = {
            "waist_z_joint": 0.0,
        }

        use_random = True
        
        randomize_joint_injection = use_random
        joint_injection_range = [-0.05, 0.05]
        
        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.05, 0.05]

        randomize_payload_mass = use_random
        payload_mass_range = [-5, 10]
        
        hand_payload_mass_range = [-0.1, 0.3]

        randomize_com_displacement = False
        com_displacement_range = [-0.1, 0.1]
        
        randomize_body_displacement = use_random
        body_displacement_range = [-0.1, 0.1]

        randomize_link_mass = use_random
        link_mass_range = [0.8, 1.2]
        
        randomize_friction = use_random
        friction_range = [0.1, 3.0]
        
        randomize_restitution = use_random
        restitution_range = [0.0, 1.0]
        
        randomize_kp = use_random
        kp_range = [0.9, 1.1]
        
        randomize_kd = use_random
        kd_range = [0.9, 1.1]
        
        randomize_initial_joint_pos = use_random
        initial_joint_pos_scale = [0.8, 1.2]
        initial_joint_pos_offset = [-0.1, 0.1]
        
        push_robots = use_random
        push_interval_s = 4
        upper_interval_s = 1
        max_push_vel_xy = 0.5
        
        init_upper_ratio = 0.
        delay = use_random

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            tracking_x_vel = 1.5
            tracking_y_vel = 1.
            tracking_ang_vel = 2.
            lin_vel_z = -0.5
            ang_vel_xy = -0.025
            orientation = -1.5
            action_rate = -0.01
            tracking_base_height = 4.
            deviation_hip_joint = -0.2
            deviation_ankle_joint = -0.5
            deviation_knee_joint = -0.75
            dof_acc = -2.5e-7
            dof_pos_limits = -2.
            feet_air_time = 0.05
            feet_clearance = -0.25
            feet_distance_lateral = 0.5
            knee_distance_lateral = 1.0
            feet_ground_parallel = -2.0
            feet_parallel = -3.0
            smoothness = -0.05
            joint_power = -2e-5
            feet_stumble = -1.5
            torques = -2.5e-6
            dof_vel = -1e-4
            dof_vel_limits = -2e-3
            torque_limits = -0.1
            no_fly = 0.75
            joint_tracking_error = -0.1
            feet_slip = -0.25
            feet_contact_forces = -0.00025
            contact_momentum = 2.5e-4
            action_vanish = -1.0
            stand_still = -0.15    
        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.975
        soft_dof_vel_limit = 0.80
        soft_torque_limit = 0.95
        base_height_target = 1.01
        max_contact_force = 400.
        least_feet_distance = 0.2
        least_feet_distance_lateral = 0.2
        most_feet_distance_lateral = 0.35
        most_knee_distance_lateral = 0.35
        least_knee_distance_lateral = 0.2
        clearance_height_target = 0.181  # 0.14 + 0.041
        
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_actions = 12
        num_dofs = 28
        num_one_step_observations = 2 * num_dofs + 10 + num_actions # 56 + 10 + 12 = 78
        num_one_step_privileged_obs = num_one_step_observations + 3
        num_actor_history = 6
        num_critic_history = 1
        num_observations = num_actor_history * num_one_step_observations
        num_privileged_obs = num_critic_history * num_one_step_privileged_obs
        action_curriculum = True
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20
        
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'

    class noise( LeggedRobotCfg.noise ):
        add_noise = True
        noise_level = 1.0
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            dof_pos = 0.02
            dof_vel = 2.0
            lin_vel = 0.1
            ang_vel = 0.5
            gravity = 0.05
            height_measurement = 0.1

class Elf3RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        use_flip = True
        entropy_coef = 0.01
        symmetry_scale = 1.0
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        save_interval = 200
        num_steps_per_env = 50
        max_iterations = 100000
        run_name = ''
        experiment_name = ''
        swanlab_project = "HomieRL-ELF3"
        swanlab_workspace = ""
        swanlab_mode = "cloud"
        logger = "swanlab"
        # logger = "tensorboard"
        # logger = "wandb"
        wandb_project = ""
        wandb_user = "" # enter your own wandb user name here
