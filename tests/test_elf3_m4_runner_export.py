from __future__ import annotations

import builtins
import copy
import importlib
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
    model = onnx.load(path)
    onnx.checker.check_model(model)
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


def test_runner_whitelists_classes_without_eval(monkeypatch):
    monkeypatch.setattr(
        builtins, "eval", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("eval called"))
    )
    runner = HIMOnPolicyRunner(FakeVecEnv(), runner_cfg(use_flip=False), None, "cpu")
    assert runner.alg.__class__.__name__ == "HIMPPO"
    bad = runner_cfg(use_flip=False)
    bad["policy"]["class_name"] = "UnknownPolicy"
    with pytest.raises(ValueError, match="UnknownPolicy|unknown"):
        HIMOnPolicyRunner(FakeVecEnv(), bad, None, "cpu")


def test_runner_updates_each_normalizer_once_per_returned_step():
    env = FakeVecEnv()
    config = runner_cfg(normalization=True, use_flip=False)
    config["num_steps_per_env"] = 2
    runner = HIMOnPolicyRunner(env, config, None, "cpu")
    actor_before = runner.alg.policy.actor_obs_normalizer.count.item()
    critic_before = runner.alg.policy.critic_obs_normalizer.count.item()
    runner.learn(2)
    expected_delta = 2 * config["num_steps_per_env"] * env.num_envs
    assert runner.alg.policy.actor_obs_normalizer.count.item() == pytest.approx(
        actor_before + expected_delta
    )
    assert runner.alg.policy.critic_obs_normalizer.count.item() == pytest.approx(
        critic_before + expected_delta
    )


def test_checkpoint_save_uses_same_directory_temp_and_os_replace(tmp_path, monkeypatch):
    runner = HIMOnPolicyRunner(FakeVecEnv(), runner_cfg(use_flip=False), None, "cpu")
    runner_module = importlib.import_module(
        "openhomie_isaaclab.him_rl.runner"
    )
    real_replace = runner_module.os.replace
    replacements = []

    def recording_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(runner_module.os, "replace", recording_replace)
    checkpoint = tmp_path / "atomic.pt"
    runner.save(str(checkpoint), infos={"atomic": True})
    assert len(replacements) == 1
    assert replacements[0][1] == checkpoint
    source, destination = replacements[0]
    assert source.parent == destination.parent == tmp_path
    assert source != destination
    assert checkpoint.exists()
    assert not source.exists()


def checkpoint_state(runner):
    return {
        "model": copy.deepcopy(runner.alg.policy.state_dict()),
        "policy_optimizer": copy.deepcopy(runner.alg.optimizer.state_dict()),
        "estimator_optimizer": copy.deepcopy(runner.alg.estimator_optimizer.state_dict()),
        "learning_rate": runner.alg.learning_rate,
        "estimator_learning_rate": runner.alg.estimator_learning_rate,
        "follows": runner.alg.estimator_lr_follows_policy,
        "iteration": runner.current_learning_iteration,
    }


def assert_checkpoint_state(runner, expected):
    assert all(
        torch.equal(expected["model"][key], value)
        for key, value in runner.alg.policy.state_dict().items()
    )
    optimizer_state_equal(runner.alg.optimizer.state_dict(), expected["policy_optimizer"])
    optimizer_state_equal(
        runner.alg.estimator_optimizer.state_dict(), expected["estimator_optimizer"]
    )
    assert runner.alg.learning_rate == expected["learning_rate"]
    assert runner.alg.estimator_learning_rate == expected["estimator_learning_rate"]
    assert runner.alg.estimator_lr_follows_policy is expected["follows"]
    assert runner.current_learning_iteration == expected["iteration"]


def test_checkpoint_payload_contains_complete_resume_contract(tmp_path):
    config = runner_cfg(use_flip=False)
    config["algorithm"]["estimator_learning_rate"] = None
    runner = HIMOnPolicyRunner(FakeVecEnv(), config, None, "cpu")
    runner.current_learning_iteration = 4
    checkpoint = tmp_path / "complete.pt"
    runner.save(str(checkpoint), infos={"phase": "D"})
    payload = torch.load(checkpoint, weights_only=False)
    assert {
        "schema_version",
        "model_state_dict",
        "optimizer_state_dict",
        "estimator_optimizer_state_dict",
        "learning_rate",
        "estimator_learning_rate",
        "estimator_lr_follows_policy",
        "iter",
        "infos",
    } <= payload.keys()
    assert payload["estimator_lr_follows_policy"] is True
    assert payload["infos"] == {"phase": "D"}


@pytest.mark.parametrize(
    "invalid",
    [
        "schema_version",
        "model_state_dict",
        "optimizer_state_dict",
        "estimator_optimizer_state_dict",
        "learning_rate",
        "estimator_learning_rate",
        "estimator_lr_follows_policy",
        "iter",
    ],
)
def test_training_load_validates_before_mutating_runner(tmp_path, invalid):
    runner = HIMOnPolicyRunner(FakeVecEnv(), runner_cfg(use_flip=False), None, "cpu")
    populate_optimizer_moments(runner)
    checkpoint = tmp_path / "valid.pt"
    runner.save(str(checkpoint), infos={"valid": True})
    payload = torch.load(checkpoint, weights_only=False)
    payload.pop(invalid)
    broken = tmp_path / f"missing_{invalid}.pt"
    torch.save(payload, broken)
    with torch.no_grad():
        next(runner.alg.policy.parameters()).add_(3.0)
    runner.current_learning_iteration = 19
    before = checkpoint_state(runner)
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        runner.load(str(broken), load_optimizer=True)
    assert_checkpoint_state(runner, before)


def test_malformed_model_load_is_transactional(tmp_path):
    runner = HIMOnPolicyRunner(FakeVecEnv(), runner_cfg(use_flip=False), None, "cpu")
    checkpoint = tmp_path / "valid.pt"
    runner.save(str(checkpoint))
    payload = torch.load(checkpoint, weights_only=False)
    payload["model_state_dict"] = dict(payload["model_state_dict"])
    payload["model_state_dict"].pop(next(iter(payload["model_state_dict"])))
    broken = tmp_path / "malformed.pt"
    torch.save(payload, broken)
    runner.current_learning_iteration = 23
    before = checkpoint_state(runner)
    with pytest.raises(RuntimeError):
        runner.load(str(broken), load_optimizer=True)
    assert_checkpoint_state(runner, before)


def test_inference_only_load_explicitly_skips_missing_optimizers(tmp_path):
    runner = HIMOnPolicyRunner(FakeVecEnv(), runner_cfg(use_flip=False), None, "cpu")
    checkpoint = tmp_path / "full.pt"
    runner.save(str(checkpoint), infos={"mode": "inference"})
    expected_model = copy.deepcopy(runner.alg.policy.state_dict())
    payload = torch.load(checkpoint, weights_only=False)
    payload.pop("optimizer_state_dict")
    payload.pop("estimator_optimizer_state_dict")
    inference = tmp_path / "inference.pt"
    torch.save(payload, inference)
    populate_optimizer_moments(runner)
    policy_optimizer = copy.deepcopy(runner.alg.optimizer.state_dict())
    estimator_optimizer = copy.deepcopy(runner.alg.estimator_optimizer.state_dict())
    with torch.no_grad():
        next(runner.alg.policy.parameters()).add_(2.0)
    infos = runner.load(str(inference), load_optimizer=False)
    assert infos == {"mode": "inference"}
    assert all(
        torch.equal(expected_model[key], value)
        for key, value in runner.alg.policy.state_dict().items()
    )
    optimizer_state_equal(runner.alg.optimizer.state_dict(), policy_optimizer)
    optimizer_state_equal(runner.alg.estimator_optimizer.state_dict(), estimator_optimizer)


def test_exporter_is_deep_copied_eval_only_and_does_not_mutate_policy():
    policy = HIMOnPolicyRunner(
        FakeVecEnv(), runner_cfg(use_flip=False), None, "cpu"
    ).alg.policy
    policy.train()
    before = copy.deepcopy(policy.state_dict())
    exporter = HIMPolicyExporter(policy)
    assert policy.training is True
    assert exporter.training is False
    assert all(module.training is False for module in exporter.modules())
    assert all(
        torch.equal(before[key], value) for key, value in policy.state_dict().items()
    )
    exporter_parameter_ids = {id(parameter) for parameter in exporter.parameters()}
    assert exporter_parameter_ids.isdisjoint({id(parameter) for parameter in policy.parameters()})
    names = {name for name, _ in exporter.named_parameters()}
    assert not any(
        token in name for name in names
        for token in ("critic", "target", "prototype", "std", "optimizer")
    )


@pytest.mark.parametrize("delta", [-1, 1])
def test_exporter_derives_and_validates_policy_input_width(delta):
    policy = HIMOnPolicyRunner(
        FakeVecEnv(), runner_cfg(use_flip=False), None, "cpu"
    ).alg.policy
    exporter = HIMPolicyExporter(policy)
    expected_width = policy.num_one_step_obs * policy.actor_history_length
    with pytest.raises(ValueError, match="width|dimension|history"):
        exporter(torch.zeros(2, expected_width + delta))


@pytest.mark.parametrize("kind", ["torchscript", "onnx"])
def test_export_rejects_directory_output_path(tmp_path, kind):
    policy = HIMOnPolicyRunner(
        FakeVecEnv(), runner_cfg(use_flip=False), None, "cpu"
    ).alg.policy
    function = (
        export_him_policy_torchscript if kind == "torchscript" else export_him_policy_onnx
    )
    with pytest.raises((ValueError, IsADirectoryError, OSError)):
        function(policy, tmp_path)


def test_verifiers_reject_missing_or_invalid_serialization(tmp_path):
    exporter = HIMPolicyExporter(
        HIMOnPolicyRunner(FakeVecEnv(), runner_cfg(use_flip=False), None, "cpu").alg.policy
    )
    histories = parity_histories()
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        verify_him_policy_torchscript(
            exporter, tmp_path / "missing.pt", histories, max_abs_error=1.0e-7
        )
    invalid = tmp_path / "invalid.pt"
    invalid.write_bytes(b"not a serialized model")
    with pytest.raises((ValueError, RuntimeError)):
        verify_him_policy_torchscript(
            exporter, invalid, histories, max_abs_error=1.0e-7
        )


def test_failed_atomic_save_preserves_existing_checkpoint(tmp_path, monkeypatch):
    runner = HIMOnPolicyRunner(FakeVecEnv(), runner_cfg(use_flip=False), None, "cpu")
    runner_module = importlib.import_module("openhomie_isaaclab.him_rl.runner")
    checkpoint = tmp_path / "existing.pt"
    checkpoint.write_bytes(b"previous checkpoint")

    def failing_replace(_source, _destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(runner_module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="replace"):
        runner.save(str(checkpoint))
    assert checkpoint.read_bytes() == b"previous checkpoint"
    assert not [path for path in tmp_path.iterdir() if path != checkpoint]


def test_checkpoint_rejects_unknown_schema_without_state_pollution(tmp_path):
    runner = HIMOnPolicyRunner(FakeVecEnv(), runner_cfg(use_flip=False), None, "cpu")
    checkpoint = tmp_path / "valid.pt"
    runner.save(str(checkpoint))
    payload = torch.load(checkpoint, weights_only=False)
    payload["schema_version"] = payload["schema_version"] + 1000
    broken = tmp_path / "future.pt"
    torch.save(payload, broken)
    runner.current_learning_iteration = 31
    before = checkpoint_state(runner)
    with pytest.raises((ValueError, RuntimeError), match="schema|version"):
        runner.load(str(broken), load_optimizer=True)
    assert_checkpoint_state(runner, before)


@pytest.mark.parametrize("optimizer_key", ["optimizer_state_dict", "estimator_optimizer_state_dict"])
def test_malformed_optimizer_load_is_transactional(tmp_path, optimizer_key):
    runner = HIMOnPolicyRunner(FakeVecEnv(), runner_cfg(use_flip=False), None, "cpu")
    populate_optimizer_moments(runner)
    checkpoint = tmp_path / "valid.pt"
    runner.save(str(checkpoint))
    payload = torch.load(checkpoint, weights_only=False)
    payload[optimizer_key] = {"state": {}, "param_groups": []}
    broken = tmp_path / f"malformed_{optimizer_key}.pt"
    torch.save(payload, broken)
    with torch.no_grad():
        next(runner.alg.policy.parameters()).add_(4.0)
    runner.current_learning_iteration = 37
    before = checkpoint_state(runner)
    with pytest.raises((ValueError, RuntimeError)):
        runner.load(str(broken), load_optimizer=True)
    assert_checkpoint_state(runner, before)


def test_export_functions_preserve_live_policy_mode_device_dtype_and_state(tmp_path):
    policy = HIMOnPolicyRunner(
        FakeVecEnv(), runner_cfg(use_flip=False), None, "cpu"
    ).alg.policy
    policy.train()
    before = copy.deepcopy(policy.state_dict())
    mode_before = policy.training
    devices_before = {parameter.device for parameter in policy.parameters()}
    dtypes_before = {parameter.dtype for parameter in policy.parameters()}
    export_him_policy_torchscript(policy, tmp_path / "policy.pt")
    export_him_policy_onnx(policy, tmp_path / "policy.onnx")
    assert policy.training is mode_before
    assert {parameter.device for parameter in policy.parameters()} == devices_before
    assert {parameter.dtype for parameter in policy.parameters()} == dtypes_before
    assert all(
        torch.equal(before[key], value) for key, value in policy.state_dict().items()
    )


def test_runner_retains_logging_and_checkpoint_hooks_when_log_dir_is_set(
    tmp_path, monkeypatch
):
    config = runner_cfg(use_flip=False)
    config["num_steps_per_env"] = 1
    config["save_interval"] = 1
    runner = HIMOnPolicyRunner(FakeVecEnv(), config, str(tmp_path), "cpu")
    prepared = []
    logged_iterations = []
    saved_paths = []
    monkeypatch.setattr(
        runner, "_prepare_logging_writer", lambda: prepared.append(True)
    )
    monkeypatch.setattr(
        runner,
        "log",
        lambda locs: logged_iterations.append(locs["completed_iteration"]),
    )
    monkeypatch.setattr(
        runner,
        "save",
        lambda path, infos=None: saved_paths.append(Path(path)),
    )
    runner.learn(1)
    assert prepared == [True]
    assert logged_iterations == [1]
    assert saved_paths
    assert all(path.parent == tmp_path for path in saved_paths)
    assert any(path.name == "model_1.pt" for path in saved_paths)
