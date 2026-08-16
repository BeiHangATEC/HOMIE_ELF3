"""HIM on-policy runner with transactional checkpoints."""

from __future__ import annotations

import copy
import math
import os
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups

from .actor_critic import HIMActorCritic
from .ppo import HIMPPO


TIMEOUT_SCALAR_TAG = "Episode_Termination/time_out"
TIMEOUT_SCALAR_UNIT = "rollout_transition_fraction"


def _rollout_timeout_fraction(
    timeout_count: int, *, num_envs: int, num_steps_per_env: int
) -> float:
    """Return the fraction of rollout transitions that ended by timeout."""
    for value, name in (
        (timeout_count, "timeout_count"),
        (num_envs, "num_envs"),
        (num_steps_per_env, "num_steps_per_env"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    if num_envs <= 0 or num_steps_per_env <= 0:
        raise ValueError("rollout dimensions must be positive")
    transition_count = int(num_envs) * int(num_steps_per_env)
    if timeout_count < 0 or timeout_count > transition_count:
        raise ValueError("timeout_count must lie within the rollout")
    fraction = float(timeout_count) / transition_count
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise RuntimeError("timeout fraction is invalid")
    return fraction


class HIMOnPolicyRunner(OnPolicyRunner):
    """Minimal rsl-rl runner specialized to the approved HIM classes."""

    CHECKPOINT_SCHEMA_VERSION = 1

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = copy.deepcopy(train_cfg)
        self.alg_cfg = copy.deepcopy(self.cfg["algorithm"])
        self.policy_cfg = copy.deepcopy(self.cfg["policy"])
        self.device = device
        self.env = env
        self.log_dir = log_dir
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        obs = self.env.get_observations().to(device)
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], ["critic"])
        self.alg = self._construct_him_algorithm(obs)
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.writer = None
        self.disable_logs = log_dir is None
        self._logging_prepared = False
        self.is_distributed = False
        self.gpu_global_rank = 0
        self.gpu_world_size = 1
        self._training_metrics: dict[str, float] = {}

    def _construct_him_algorithm(self, obs: TensorDict) -> HIMPPO:
        policy_cfg = dict(self.policy_cfg)
        policy_name = policy_cfg.pop("class_name", None)
        if policy_name != "HIMActorCritic":
            raise ValueError(f"unknown policy class: {policy_name}")
        algorithm_cfg = dict(self.alg_cfg)
        algorithm_name = algorithm_cfg.pop("class_name", None)
        if algorithm_name != "HIMPPO":
            raise ValueError(f"unknown algorithm class: {algorithm_name}")
        policy = HIMActorCritic(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **policy_cfg,
        ).to(self.device)
        algorithm = HIMPPO(policy, device=self.device, **algorithm_cfg)
        algorithm.init_storage(
            "rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions]
        )
        return algorithm

    @staticmethod
    def _require_finite(name: str, value) -> None:
        if isinstance(value, TensorDict):
            for key, tensor in value.items():
                HIMOnPolicyRunner._require_finite(f"{name}.{key}", tensor)
            return
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} is non-finite")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if self.log_dir is not None and not self._logging_prepared:
            self._prepare_logging_writer()
            self._logging_prepared = True
        if isinstance(num_learning_iterations, bool) or num_learning_iterations < 0:
            raise ValueError("number of learning iterations must be nonnegative")
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations().to(self.device)
        self.alg.policy.train()
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        last_saved_iteration: int | None = None
        for _ in range(num_learning_iterations):
            timeout_count = torch.zeros((), dtype=torch.long, device=self.env.device)
            collection_start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    self._require_finite("observations", obs)
                    actions = self.alg.act(obs)
                    self._require_finite("actions", actions)
                    next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    self._require_finite("next observations", next_obs)
                    self._require_finite("rewards", rewards)
                    time_outs = extras.get("time_outs")
                    if not isinstance(time_outs, torch.Tensor):
                        raise TypeError("time_outs must be a tensor")
                    if time_outs.dtype != torch.bool:
                        raise TypeError("time_outs must be boolean")
                    if tuple(time_outs.shape) != (self.env.num_envs,):
                        raise ValueError("time_outs shape does not match the environment batch")
                    timeout_count.add_(torch.count_nonzero(time_outs))
                    self.alg.process_env_step(next_obs, rewards, dones, extras)
                    obs = next_obs
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        completed = dones.nonzero(as_tuple=False).reshape(-1)
                        rewbuffer.extend(cur_reward_sum[completed].cpu().tolist())
                        lenbuffer.extend(cur_episode_length[completed].cpu().tolist())
                        cur_reward_sum[completed] = 0
                        cur_episode_length[completed] = 0
                self.alg.compute_returns(obs)
            collection_time = time.time() - collection_start
            learn_start = time.time()
            loss_dict = self.alg.update()
            learn_time = time.time() - learn_start
            for name, value in loss_dict.items():
                if not math.isfinite(value):
                    raise RuntimeError(f"training metric {name} is non-finite")
            self._training_metrics = dict(loss_dict)
            self.current_learning_iteration += 1
            completed_iteration = self.current_learning_iteration
            it = completed_iteration
            timeout_fraction = _rollout_timeout_fraction(
                int(timeout_count.item()),
                num_envs=self.env.num_envs,
                num_steps_per_env=self.num_steps_per_env,
            )
            if self.log_dir is not None:
                self.log(locals())
                if completed_iteration % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{completed_iteration}.pt"))
                    last_saved_iteration = completed_iteration
            else:
                self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            ep_infos.clear()
        if (
            self.log_dir is not None
            and self.current_learning_iteration != last_saved_iteration
        ):
            self.save(
                os.path.join(
                    self.log_dir, f"model_{self.current_learning_iteration}.pt"
                )
            )

    def log(self, locs: dict, *args, **kwargs) -> None:
        super().log(locs, *args, **kwargs)
        self.writer.add_scalar(TIMEOUT_SCALAR_TAG, locs["timeout_fraction"], locs["it"])

    def get_training_metrics(self) -> dict[str, float]:
        return dict(self._training_metrics)

    def train_mode(self) -> None:
        self.alg.policy.train()

    def eval_mode(self) -> None:
        self.alg.policy.eval()

    def get_inference_policy(self, device: str | None = None):
        if device is not None and str(next(self.alg.policy.parameters()).device) != str(device):
            raise ValueError("inference device must match the live policy device")
        return self.alg.policy.act_inference

    def _checkpoint_payload(self, infos: dict | None) -> dict:
        return {
            "schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "estimator_optimizer_state_dict": self.alg.estimator_optimizer.state_dict(),
            "learning_rate": self.alg.learning_rate,
            "estimator_learning_rate": self.alg.estimator_learning_rate,
            "estimator_lr_follows_policy": self.alg.estimator_lr_follows_policy,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }

    def save(self, path: str, infos: dict | None = None) -> None:
        destination = Path(path)
        if destination.exists() and destination.is_dir():
            raise IsADirectoryError(str(destination))
        if not destination.parent.is_dir():
            raise FileNotFoundError(str(destination.parent))
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = temporary.name
            torch.save(self._checkpoint_payload(infos), temporary_path)
            os.replace(temporary_path, destination)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass

    @staticmethod
    def _validate_scalar(name: str, value) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"checkpoint {name} must be finite")
        return float(value)

    def _validate_checkpoint(self, payload: dict, load_optimizer: bool) -> None:
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload must be a dictionary")
        if load_optimizer and "estimator_optimizer_state_dict" not in payload:
            raise KeyError("estimator_optimizer_state_dict")
        if load_optimizer and "optimizer_state_dict" not in payload:
            raise KeyError("optimizer_state_dict")
        required = {
            "schema_version",
            "model_state_dict",
            "learning_rate",
            "estimator_learning_rate",
            "estimator_lr_follows_policy",
            "iter",
        }
        if load_optimizer:
            required |= {"optimizer_state_dict", "estimator_optimizer_state_dict"}
        missing = required - payload.keys()
        if missing:
            raise KeyError(next(iter(sorted(missing))))
        if payload["schema_version"] != self.CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint schema version")
        if isinstance(payload["iter"], bool) or not isinstance(payload["iter"], int) or payload["iter"] < 0:
            raise ValueError("checkpoint iteration must be a nonnegative integer")
        if not isinstance(payload["estimator_lr_follows_policy"], bool):
            raise ValueError("checkpoint estimator LR follow flag must be boolean")
        self._validate_scalar("learning rate", payload["learning_rate"])
        self._validate_scalar("estimator learning rate", payload["estimator_learning_rate"])
        policy_probe = copy.deepcopy(self.alg.policy)
        policy_probe.load_state_dict(payload["model_state_dict"])
        if load_optimizer:
            policy_optimizer_probe = copy.deepcopy(self.alg.optimizer)
            estimator_optimizer_probe = copy.deepcopy(self.alg.estimator_optimizer)
            policy_optimizer_probe.load_state_dict(payload["optimizer_state_dict"])
            estimator_optimizer_probe.load_state_dict(payload["estimator_optimizer_state_dict"])

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict | None:
        payload = torch.load(path, weights_only=False, map_location=map_location)
        self._validate_checkpoint(payload, load_optimizer)
        model_before = copy.deepcopy(self.alg.policy.state_dict())
        policy_optimizer_before = copy.deepcopy(self.alg.optimizer.state_dict())
        estimator_optimizer_before = copy.deepcopy(self.alg.estimator_optimizer.state_dict())
        learning_rate_before = self.alg.learning_rate
        estimator_learning_rate_before = self.alg.estimator_learning_rate
        follows_before = self.alg.estimator_lr_follows_policy
        iteration_before = self.current_learning_iteration
        try:
            self.alg.policy.load_state_dict(payload["model_state_dict"])
            if load_optimizer:
                self.alg.optimizer.load_state_dict(payload["optimizer_state_dict"])
                self.alg.estimator_optimizer.load_state_dict(payload["estimator_optimizer_state_dict"])
                self.alg.learning_rate = float(payload["learning_rate"])
                self.alg.estimator_learning_rate = float(payload["estimator_learning_rate"])
                self.alg.estimator_lr_follows_policy = payload["estimator_lr_follows_policy"]
            self.current_learning_iteration = payload["iter"]
        except Exception:
            self.alg.policy.load_state_dict(model_before)
            self.alg.optimizer.load_state_dict(policy_optimizer_before)
            self.alg.estimator_optimizer.load_state_dict(estimator_optimizer_before)
            self.alg.learning_rate = learning_rate_before
            self.alg.estimator_learning_rate = estimator_learning_rate_before
            self.alg.estimator_lr_follows_policy = follows_before
            self.current_learning_iteration = iteration_before
            raise
        return payload.get("infos")
