from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from tensordict import TensorDict

from openhomie_isaaclab import elf3_constants as C
from openhomie_isaaclab.him_rl import (
    HIMOnPolicyRunner,
    HIMPolicyExporter,
    export_him_policy_onnx,
    export_him_policy_torchscript,
    verify_him_policy_onnx,
    verify_him_policy_torchscript,
)

from _elf3_m4_helpers import make_obs, runner_cfg


class FakeVecEnv:
    num_envs = 3
    num_actions = C.NUM_POLICY_ACTIONS
    device = "cpu"
    max_episode_length = 100
    cfg = {}

    def __init__(self):
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.step_count = 0
        self._obs = make_obs(self.num_envs, random=False)

    def get_observations(self) -> TensorDict:
        return self._obs.clone()

    def step(self, actions):
        assert actions.shape == (self.num_envs, self.num_actions)
        self.step_count += 1
        next_obs = make_obs(self.num_envs, random=False)
        next_obs["policy"].fill_(float(self.step_count))
        next_obs["critic"].fill_(float(self.step_count))
        dones = torch.tensor([False, True, True])
        terminal = next_obs["critic"].clone() + 100.0
        extras = {
            "time_outs": torch.tensor([False, False, True]),
            "terminal_critic_obs": terminal,
            "terminal_critic_obs_mask": dones.clone(),
            "log": {"episode_length": torch.ones(2)},
        }
        self._obs = next_obs
        return next_obs, torch.ones(self.num_envs), dones, extras


def optimizer_state_equal(actual, expected):
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for key in actual["state"]:
        assert actual["state"][key].keys() == expected["state"][key].keys()
        for field, value in actual["state"][key].items():
            reference = expected["state"][key][field]
            assert torch.equal(value, reference) if isinstance(value, torch.Tensor) else value == reference


def populate_optimizer_moments(runner):
    for optimizer in (runner.alg.optimizer, runner.alg.estimator_optimizer):
        loss = sum(p.square().sum() for group in optimizer.param_groups for p in group["params"])
        optimizer.zero_grad(); loss.backward(); optimizer.step(); optimizer.zero_grad()


def test_fake_vecenv_runner_completes_update_and_counts_repeated_learn_calls():
    torch.manual_seed(9)
    env = FakeVecEnv()
    config = runner_cfg(use_flip=True); config["num_steps_per_env"] = 1
    runner = HIMOnPolicyRunner(env, config, log_dir=None, device="cpu")
    runner.learn(1)
    assert runner.current_learning_iteration == 1
    assert env.step_count == 1
    assert all(np.isfinite(value) for value in runner.get_training_metrics().values())
    runner.learn(1)
    assert runner.current_learning_iteration == 2 and env.step_count == 2


def test_checkpoint_atomically_round_trips_model_normalizers_optimizers_and_lrs(tmp_path):
    torch.manual_seed(10)
    config = runner_cfg(normalization=True, use_flip=False)
    runner = HIMOnPolicyRunner(FakeVecEnv(), config, log_dir=None, device="cpu")
    runner.alg.policy.update_normalization(make_obs(3))
    populate_optimizer_moments(runner)
    runner.current_learning_iteration = 7
    expected_model = copy.deepcopy(runner.alg.policy.state_dict())
    expected_policy_optimizer = copy.deepcopy(runner.alg.optimizer.state_dict())
    expected_estimator_optimizer = copy.deepcopy(runner.alg.estimator_optimizer.state_dict())
    checkpoint = tmp_path / "model_7.pt"
    runner.save(str(checkpoint), infos={"stage": "M4"})
    assert checkpoint.exists()
    assert not list(tmp_path.glob(".*.tmp"))
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["schema_version"] >= 1
    assert payload["iter"] == 7
    with torch.no_grad():
        for parameter in runner.alg.policy.parameters():
            parameter.add_(1.0)
    infos = runner.load(str(checkpoint), load_optimizer=True)
    assert infos == {"stage": "M4"} and runner.current_learning_iteration == 7
    assert runner.alg.learning_rate == config["algorithm"]["learning_rate"]
    assert runner.alg.estimator_learning_rate == config["algorithm"]["estimator_learning_rate"]
    assert all(torch.equal(expected_model[k], v) for k, v in runner.alg.policy.state_dict().items())
    optimizer_state_equal(runner.alg.optimizer.state_dict(), expected_policy_optimizer)
    optimizer_state_equal(runner.alg.estimator_optimizer.state_dict(), expected_estimator_optimizer)


def test_training_resume_rejects_checkpoint_without_estimator_optimizer(tmp_path):
    runner = HIMOnPolicyRunner(FakeVecEnv(), runner_cfg(use_flip=False), None, "cpu")
    checkpoint = tmp_path / "bad.pt"
    torch.save({
        "schema_version": 1,
        "model_state_dict": runner.alg.policy.state_dict(),
        "optimizer_state_dict": runner.alg.optimizer.state_dict(),
        "iter": 1,
        "infos": None,
    }, checkpoint)
    with pytest.raises(KeyError, match="estimator_optimizer"):
        runner.load(str(checkpoint), load_optimizer=True)


def parity_histories():
    width = C.num_actor_obs()
    generator = torch.Generator().manual_seed(11)
    return (
        torch.zeros(1, width),
        torch.linspace(-1.0, 1.0, width).repeat(2, 1),
        torch.randn(5, width, generator=generator),
    )


def test_exporter_contains_complete_deterministic_him_path_and_normalizer():
    runner = HIMOnPolicyRunner(FakeVecEnv(), runner_cfg(normalization=True, use_flip=False), None, "cpu")
    runner.alg.policy.update_normalization(make_obs(3))
    exporter = HIMPolicyExporter(runner.alg.policy)
    history = parity_histories()[-1]
    obs = TensorDict(
        {"policy": history, "critic": torch.zeros(history.shape[0], C.num_critic_obs())},
        batch_size=[history.shape[0]],
    )
    with torch.inference_mode():
        actual = exporter(history)
        expected = runner.alg.policy.act_inference(obs)
    assert torch.allclose(actual, expected, atol=1.0e-7, rtol=0.0)
    names = {name for name, _ in exporter.named_modules()}
    assert "actor_obs_normalizer" in names
    assert "estimator_encoder" in names
    assert "actor" in names
    assert not any("critic" in name or "target" in name or "prototype" in name for name in names)


def test_torchscript_round_trip_meets_strict_parity_for_dynamic_batches(tmp_path):
    runner = HIMOnPolicyRunner(FakeVecEnv(), runner_cfg(use_flip=False), None, "cpu")
    path = tmp_path / "elf3_him.pt"
    exporter = HIMPolicyExporter(runner.alg.policy)
    export_him_policy_torchscript(runner.alg.policy, path)
    loaded = torch.jit.load(str(path), map_location="cpu")
    errors = []
    for history in parity_histories():
        with torch.inference_mode():
            errors.append((loaded(history) - exporter(history)).abs().max().item())
    assert max(errors) <= 1.0e-7
    assert verify_him_policy_torchscript(exporter, path, parity_histories(), max_abs_error=1.0e-7) <= 1.0e-7


def test_onnx_runtime_round_trip_meets_parity_without_skip(tmp_path):
    assert onnx.__version__ == "1.21.0"
    assert ort.__version__ == "1.28.0"
    runner = HIMOnPolicyRunner(FakeVecEnv(), runner_cfg(use_flip=False), None, "cpu")
    path = tmp_path / "elf3_him.onnx"
    exporter = export_him_policy_onnx(runner.alg.policy, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    errors = []
    for history in parity_histories():
        expected = exporter(history).detach().numpy()
        actual = session.run(None, {"observation_history": history.numpy()})[0]
        assert actual.shape == (history.shape[0], C.NUM_POLICY_ACTIONS)
        assert np.isfinite(actual).all()
        errors.append(float(np.max(np.abs(actual - expected))))
    assert max(errors) <= 1.0e-5
    assert verify_him_policy_onnx(exporter, path, parity_histories(), max_abs_error=1.0e-5) <= 1.0e-5


@pytest.mark.parametrize("limit", [True, -1.0, float("nan"), float("inf")])
def test_export_parity_rejects_invalid_threshold(limit):
    with pytest.raises(ValueError, match="finite and non-negative"):
        verify_him_policy_onnx(None, Path("missing.onnx"), [], max_abs_error=limit)
