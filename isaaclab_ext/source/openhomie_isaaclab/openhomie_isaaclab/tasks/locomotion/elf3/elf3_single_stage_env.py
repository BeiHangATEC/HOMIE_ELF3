"""Dedicated ELF3 environment for G1-proportioned single-stage training."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from . import elf3_homie_curriculum as curriculum
from .elf3_homie_env import Elf3HomieEnv
from .elf3_single_stage_env_cfg import Elf3SingleStageEnvCfg


class Elf3SingleStageEnv(Elf3HomieEnv):
    """Keep the staged task frozen while replacing only command sampling."""

    cfg: Elf3SingleStageEnvCfg

    def _allocate_state(self):
        super()._allocate_state()
        curriculum.validate_g1_single_stage_spec(
            lin_vel_x_range=self.cfg.lin_vel_x_range,
            lin_vel_y_range=self.cfg.lin_vel_y_range,
            ang_vel_yaw_range=self.cfg.ang_vel_yaw_range,
            height_range=self.cfg.single_stage_height_range,
            stand_height=self.cfg.single_stage_stand_height,
        )

    def set_evaluation_command(
        self, command: Sequence[float | None], mode: int
    ) -> None:
        values = tuple(command)
        if len(values) != 4:
            raise ValueError("evaluation command must contain vx, vy, yaw rate, and height")
        height = self.cfg.single_stage_stand_height if values[3] is None else values[3]
        super().set_evaluation_command((values[0], values[1], values[2], height), mode)

    def _resample_commands(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        if self._evaluation_command is not None:
            self._commands[env_ids] = self._evaluation_command
            self._modes[env_ids] = self._evaluation_mode
            return
        count = env_ids.numel()
        if self.cfg.use_random_commands:
            modes = curriculum.sample_g1_single_stage_modes(
                torch.rand(count, device=self.device)
            )
            commands = curriculum.build_g1_single_stage_commands(
                modes=modes,
                velocity_draws=torch.rand(count, 3, device=self.device),
                height_draws=torch.rand(count, device=self.device),
                lin_vel_x_range=self.cfg.lin_vel_x_range,
                lin_vel_y_range=self.cfg.lin_vel_y_range,
                ang_vel_yaw_range=self.cfg.ang_vel_yaw_range,
                height_range=self.cfg.single_stage_height_range,
                stand_height=self.cfg.single_stage_stand_height,
            )
        else:
            modes = torch.full(
                (count,), curriculum.HIGH_STAND, device=self.device, dtype=torch.long
            )
            commands = torch.zeros(count, 5, device=self.device)
            commands[:, 4] = self.cfg.single_stage_stand_height
        self._modes[env_ids] = modes
        self._commands[env_ids] = commands
