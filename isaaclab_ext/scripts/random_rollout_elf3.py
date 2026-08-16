"""Headless numerical-stability rollout for the registered ELF3 task."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

EXTENSION_SOURCE = Path(__file__).resolve().parents[1] / "source/openhomie_isaaclab"
TASK_ID = "OpenHomie-Elf3-Homie-Direct-v0"


def main() -> int:
    sys.path.insert(0, str(EXTENSION_SOURCE))
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-length-s", type=float)
    parser.add_argument("--disable-failure-termination", action="store_true")
    parser.add_argument("--induce-failure", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    simulation_app = AppLauncher(args).app
    exit_code = 1
    env = None
    try:
        import gymnasium as gym
        import torch
        import openhomie_isaaclab.tasks.locomotion.elf3  # noqa: F401
        from openhomie_isaaclab import elf3_constants as C
        from openhomie_isaaclab.tasks.locomotion.elf3.elf3_homie_env_cfg import Elf3HomieEnvCfg

        if args.steps <= 0 or args.num_envs <= 0:
            raise ValueError("steps and num-envs must be positive")
        torch.manual_seed(args.seed)
        cfg = Elf3HomieEnvCfg()
        cfg.scene.num_envs = args.num_envs
        if args.episode_length_s is not None:
            if args.episode_length_s <= 0:
                raise ValueError("episode length must be positive")
            cfg.episode_length_s = args.episode_length_s
        if args.disable_failure_termination:
            cfg.terminate_on_torso_contact = False
            cfg.gravity_xy_termination = float("inf")
        if args.induce_failure:
            cfg.terminate_on_torso_contact = False
            cfg.gravity_xy_termination = -1.0
        env = gym.make(TASK_ID, cfg=cfg, render_mode=None)
        obs, _ = env.reset(seed=args.seed)
        terminated_count = 0
        timeout_count = 0
        for _ in range(args.steps):
            actions = 2.0 * torch.rand(args.num_envs, C.NUM_POLICY_ACTIONS, device=env.unwrapped.device) - 1.0
            obs, rewards, terminated, truncated, extras = env.step(actions)
            time_outs = extras["time_outs"]
            tensors = [actions, rewards, terminated, truncated, time_outs]
            tensors.extend(obs.values())
            tensors.extend((env.unwrapped.applied_torques_canonical, env.unwrapped.effort_limits_canonical))
            if not all(torch.isfinite(value).all() for value in tensors):
                raise RuntimeError("rollout produced a non-finite tensor")
            if not torch.equal(time_outs, truncated) or torch.any(time_outs & terminated):
                raise RuntimeError("termination and truncation semantics diverged")
            terminated_count += int(terminated.count_nonzero())
            timeout_count += int(time_outs.count_nonzero())
        if args.disable_failure_termination and timeout_count == 0:
            raise RuntimeError("controlled timeout run observed no timeouts")
        if args.induce_failure and terminated_count == 0:
            raise RuntimeError("controlled failure run observed no terminations")
        print(f"random rollout steps: {args.steps}")
        print(f"observed terminated: {terminated_count}")
        print(f"observed time_outs: {timeout_count}")
        exit_code = 0
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                traceback.print_exc()
                exit_code = 1
        if exit_code == 0:
            print("M3a random rollout: PASS", flush=True)
        else:
            print("M3a random rollout: FAIL", flush=True)
        print(f"random_rollout_elf3 exit code: {exit_code}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
