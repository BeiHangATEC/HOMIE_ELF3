"""ELF3 HIM agent configuration entry points."""

from .him_ppo_cfg import Elf3HIMRunnerCfg

HIM_PPO_CFG_ENTRY_POINT = (
    "openhomie_isaaclab.tasks.locomotion.elf3.agents."
    "him_ppo_cfg:Elf3HIMRunnerCfg"
)

__all__ = ["Elf3HIMRunnerCfg", "HIM_PPO_CFG_ENTRY_POINT"]
