"""Independent Isaac Lab acceptance harness for M3a.

This is intentionally outside pytest discovery. Run it under the pinned homie
environment after the production environment and rollout entry point exist.
"""

from __future__ import annotations

import argparse
import faulthandler
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_SOURCE = REPO_ROOT / "isaaclab_ext/source/openhomie_isaaclab"
TASK_ID = "OpenHomie-Elf3-Homie-Direct-v0"


def _preflight() -> None:
    required = [
        EXTENSION_SOURCE / "openhomie_isaaclab/tasks/locomotion/elf3/elf3_homie_env.py",
        EXTENSION_SOURCE / "openhomie_isaaclab/tasks/locomotion/elf3/elf3_homie_env_cfg.py",
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("M3a production interfaces missing: " + ", ".join(missing))


def main() -> int:
    faulthandler.enable(all_threads=True)
    _preflight()
    sys.path.insert(0, str(EXTENSION_SOURCE))

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    simulation_app = launcher.app
    exit_code = 1
    try:
        import gymnasium as gym
        import torch

        import openhomie_isaaclab.tasks.locomotion.elf3  # noqa: F401
        from openhomie_isaaclab import elf3_constants as C
        from openhomie_isaaclab.tasks.locomotion.elf3 import elf3_homie_rewards as R
        from openhomie_isaaclab.tasks.locomotion.elf3.elf3_homie_env_cfg import (
            Elf3HomieEnvCfg,
        )

        spec = gym.spec(TASK_ID)
        assert spec is not None
        print("task registration: PASS")

        cfg = Elf3HomieEnvCfg()
        cfg.scene.num_envs = args.num_envs
        assert cfg.events is not None
        assert cfg.events.massless_link_mass is not None
        for event_name in (
            "physics_material",
            "non_torso_link_mass",
            "torso_payload",
            "hand_payload",
            "torso_com",
            "push_robot",
        ):
            setattr(cfg.events, event_name, None)
        cfg.add_noise = False
        cfg.enable_action_delay = False
        cfg.randomize_control = False
        cfg.randomize_initial_state = False
        print("diagnostic boundary: entering gym.make", flush=True)
        env = gym.make(TASK_ID, cfg=cfg, render_mode=None)
        print("diagnostic boundary: gym.make returned", flush=True)
        unwrapped = env.unwrapped
        robot = unwrapped.scene.articulations["robot"]
        obs, _ = env.reset(seed=42)
        assert robot.num_joints == C.NUM_ROBOT_DOFS
        assert set(robot.joint_names) == set(C.JOINT_NAMES)
        expected_mapping = torch.tensor(
            [robot.joint_names.index(name) for name in C.JOINT_NAMES],
            dtype=torch.long,
            device=unwrapped.device,
        )
        assert torch.equal(unwrapped.canonical_to_runtime_dof_indices, expected_mapping)
        assert not torch.equal(
            expected_mapping,
            torch.arange(C.NUM_ROBOT_DOFS, device=unwrapped.device),
        )
        print("runtime joint mapping: PASS (28 joints)")

        assert obs["policy"].shape == (args.num_envs, C.num_actor_obs())
        assert obs["critic"].shape == (args.num_envs, C.num_critic_obs())
        actions = torch.zeros(
            (args.num_envs, C.NUM_POLICY_ACTIONS),
            dtype=torch.float32,
            device=unwrapped.device,
        )
        next_obs, rewards, terminated, truncated, extras = env.step(actions)
        assert actions.shape == (args.num_envs, C.NUM_POLICY_ACTIONS)
        assert next_obs["policy"].shape == (args.num_envs, C.num_actor_obs())
        assert next_obs["critic"].shape == (args.num_envs, C.num_critic_obs())
        expected_device = torch.device(unwrapped.device)
        for value in (obs["policy"], obs["critic"], next_obs["policy"], next_obs["critic"]):
            assert value.dtype == torch.float32
            assert value.device == expected_device
        tensors = [rewards, terminated, truncated]
        tensors.extend(value for value in next_obs.values() if isinstance(value, torch.Tensor))
        assert all(torch.isfinite(value).all() for value in tensors)
        print("reset/step shapes and finiteness: PASS")

        leg_ids = torch.tensor(C.LOWER_DOF_INDICES, device=unwrapped.device)
        assert unwrapped.applied_torques_canonical.shape == (
            args.num_envs, C.NUM_ROBOT_DOFS
        )
        assert unwrapped.effort_limits_canonical.shape == (
            args.num_envs, C.NUM_ROBOT_DOFS
        )
        assert torch.all(
            unwrapped.applied_torques_canonical[:, leg_ids].abs()
            <= unwrapped.effort_limits_canonical[:, leg_ids] + 1e-6
        )
        print("effort limits and mixed control: PASS")

        assert tuple(unwrapped.reward_term_names) == tuple(R.REWARD_NAMES)
        assert unwrapped.reward_scales == R.REWARD_SCALES
        assert set(unwrapped.last_raw_reward_terms) == set(R.REWARD_NAMES)
        assert set(unwrapped.last_scaled_reward_terms) == set(R.REWARD_NAMES)
        for name in R.REWARD_NAMES:
            expected = (
                unwrapped.last_raw_reward_terms[name]
                * R.REWARD_SCALES[name]
                * C.policy_dt()
            )
            assert torch.allclose(unwrapped.last_scaled_reward_terms[name], expected)
        print("reward wiring and single dt scaling: PASS")

        before = unwrapped.episode_length_buf.clone()
        unwrapped._reset_idx(torch.tensor([0], device=unwrapped.device))
        assert unwrapped.episode_length_buf[0] == 0
        assert torch.equal(unwrapped.episode_length_buf[1:], before[1:])
        print("subset reset isolation: PASS")

        total_mass = robot.root_physx_view.get_masses().sum(dim=1)[0].item()
        assert abs(total_mass - C.TOTAL_MASS) < 0.05, total_mass
        print(f"runtime mass: PASS ({total_mass:.3f} kg)")

        assert not torch.any(extras["time_outs"] & terminated)
        assert torch.equal(extras["time_outs"], truncated)
        print("done separation for observed step: PASS")
        env.close()
        print("M3a Isaac integration: PASS")
        exit_code = 0
    except BaseException:
        traceback.print_exc()
        print("M3a Isaac integration: FAIL", flush=True)
        exit_code = 1
    finally:
        print(f"check_elf3_m3a_env exit code: {exit_code}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
