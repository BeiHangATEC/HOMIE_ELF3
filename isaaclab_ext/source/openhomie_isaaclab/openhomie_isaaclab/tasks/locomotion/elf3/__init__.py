"""Gymnasium registration for the ELF3 HOMIE direct task."""

import gymnasium as gym

TASK_ID = "OpenHomie-Elf3-Homie-Direct-v0"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="openhomie_isaaclab.tasks.locomotion.elf3.elf3_homie_env:Elf3HomieEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "openhomie_isaaclab.tasks.locomotion.elf3."
                "elf3_homie_env_cfg:Elf3HomieEnvCfg"
            )
        },
    )
