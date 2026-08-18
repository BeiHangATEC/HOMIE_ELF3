"""Configuration isolated from the frozen staged ELF3 training task."""

from __future__ import annotations

from isaaclab.utils import configclass

from .elf3_homie_env_cfg import Elf3HomieEnvCfg


@configclass
class Elf3SingleStageEnvCfg(Elf3HomieEnvCfg):
    """Use the G1 command mix over ELF3's full requested height range."""

    command_profile = "g1_single_stage"
    single_stage_height_range = (0.40, 1.01)
    single_stage_stand_height = 1.01
