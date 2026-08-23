from legged_gym.envs.g1.elf3_dof31_r1_config import (
    Elf3Dof31R1Cfg,
    Elf3Dof31R1CfgPPO,
    Elf3Dof31R1Contract,
)


class Elf3Dof31R2Contract(Elf3Dof31R1Contract):
    contract_stage = "R2"
    standing_height_command_m = Elf3Dof31R1Contract.natural_standing_height_command_m
    max_relative_crouch_depth_m = 0.05
    minimum_height_command_m = standing_height_command_m - max_relative_crouch_depth_m
    initialization = "weights-only from selected R1 model_2500.pt; fresh optimizer; preserve checkpoint policy std"


class Elf3Dof31R2Cfg(Elf3Dof31R1Cfg):
    class commands(Elf3Dof31R1Cfg.commands):
        fixed_absolute_body_height = False
        stage1_curriculum = False
        stage6_curriculum = False
        stage7_curriculum = False
        stage8_curriculum = False
        stage9_curriculum = False
        reset_velocity = 0.03

        class ranges(Elf3Dof31R1Cfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            height = [-Elf3Dof31R2Contract.max_relative_crouch_depth_m, 0.0]

    class rewards(Elf3Dof31R1Cfg.rewards):
        base_height_target = Elf3Dof31R2Contract.standing_height_command_m
        crouch_height_threshold = Elf3Dof31R2Contract.standing_height_command_m


class Elf3Dof31R2CfgPPO(Elf3Dof31R1CfgPPO):
    class runner(Elf3Dof31R1CfgPPO.runner):
        resume = False
        save_interval = 100
        max_iterations = 500
        run_name = "r2_height_0_to_005_from_r1_model_2500"
        experiment_name = "elf3_dof31_r2"
