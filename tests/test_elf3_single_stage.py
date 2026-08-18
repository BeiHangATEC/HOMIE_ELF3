from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab"
SCRIPT = ROOT / "isaaclab_ext/scripts/elf3_single_stage_train.py"
LIVE_LAUNCHER = ROOT / "isaaclab_ext/scripts/elf3_single_stage_train.sh"
WORKFLOW = PACKAGE / "workflows/elf3_single_stage.py"
ENV = PACKAGE / "tasks/locomotion/elf3/elf3_homie_env.py"
SINGLE_STAGE_ENV = PACKAGE / "tasks/locomotion/elf3/elf3_single_stage_env.py"
SINGLE_STAGE_CFG = PACKAGE / "tasks/locomotion/elf3/elf3_single_stage_env_cfg.py"


def test_single_stage_keeps_the_frozen_c1_environment_unchanged():
    assert hashlib.sha256(ENV.read_bytes()).hexdigest() == (
        "d7f54abb9b424e95d043df70ca350f32a61a43a7075ecc8859f2c87c7ed43342"
    )


def test_single_stage_has_a_dedicated_environment_and_config_surface():
    env_source = SINGLE_STAGE_ENV.read_text(encoding="utf-8")
    cfg_source = SINGLE_STAGE_CFG.read_text(encoding="utf-8")
    assert "class Elf3SingleStageEnv(Elf3HomieEnv)" in env_source
    assert "class Elf3SingleStageEnvCfg(Elf3HomieEnvCfg)" in cfg_source


def test_single_stage_surface_is_pure_before_app_launch():
    assert SCRIPT.is_file()
    assert WORKFLOW.is_file()
    source = str(PACKAGE.parent)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source, environment.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import openhomie_isaaclab.workflows.elf3_single_stage; "
            "assert not ({'isaaclab', 'isaaclab_rl'} & set(sys.modules))",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_live_launcher_disables_conda_output_capture_and_forwards_arguments():
    source = LIVE_LAUNCHER.read_text(encoding="utf-8")
    assert source.startswith("#!/usr/bin/env bash\n")
    assert "conda run --no-capture-output -n homie python" in source
    assert "elf3_single_stage_train.py" in source
    assert '"$@"' in source


def test_live_log_payloads_show_the_request_and_effective_training_config(tmp_path):
    from openhomie_isaaclab.workflows.elf3_single_stage import (
        effective_config_log_payload,
        request_log_payload,
    )

    request = SimpleNamespace(
        to_dict=lambda: {
            "run_dir": str(tmp_path / "run"),
            "device": "cuda:0",
            "seed": 42,
            "num_envs": 4096,
            "iterations": 100_000,
            "headless": True,
            "command_profile": "g1_single_stage",
        },
        command_profile="g1_single_stage",
    )
    env_cfg = SimpleNamespace(
        command_profile="g1_single_stage",
        single_stage_height_range=(0.40, 1.01),
        single_stage_stand_height=1.01,
        lin_vel_x_range=(-0.8, 1.2),
        lin_vel_y_range=(-0.5, 0.5),
        ang_vel_yaw_range=(-0.8, 0.8),
    )
    agent_cfg = SimpleNamespace(
        num_steps_per_env=50,
        max_iterations=100_000,
        save_interval=200,
        policy=SimpleNamespace(
            actor_hidden_dims=[512, 256, 256],
            critic_hidden_dims=[512, 256, 256],
            estimator_hidden_dims=[256, 256],
            estimator_target_hidden_dims=[256, 256],
            estimator_latent_dim=32,
            estimator_num_prototypes=64,
        ),
        algorithm=SimpleNamespace(
            learning_rate=1e-3,
            schedule="adaptive",
            entropy_coef=0.01,
            estimator_learning_rate=None,
        ),
    )

    request_payload = request_log_payload(request)
    config_payload = effective_config_log_payload(request, env_cfg, agent_cfg)

    assert request_payload["event"] == "ELF3_SINGLE_STAGE_REQUEST"
    assert request_payload["request"]["iterations"] == 100_000
    assert config_payload == {
        "event": "ELF3_SINGLE_STAGE_EFFECTIVE_CONFIG",
        "task_id": "OpenHomie-Elf3-Homie-SingleStage-v0",
        "profile": "g1_single_stage",
        "commands": {
            "mode_probabilities": {
                "walk": 0.5,
                "height_hold": 1.0 / 3.0,
                "stand": 1.0 / 6.0,
            },
            "height_range": [0.40, 1.01],
            "stand_height": 1.01,
            "lin_vel_x_range": [-0.8, 1.2],
            "lin_vel_y_range": [-0.5, 0.5],
            "ang_vel_yaw_range": [-0.8, 0.8],
        },
        "network": {
            "actor_hidden_dims": [512, 256, 256],
            "critic_hidden_dims": [512, 256, 256],
            "estimator_hidden_dims": [256, 256],
            "estimator_target_hidden_dims": [256, 256],
            "estimator_latent_dim": 32,
            "estimator_num_prototypes": 64,
        },
        "ppo": {
            "num_steps_per_env": 50,
            "max_iterations": 100_000,
            "save_interval": 200,
            "learning_rate": 1e-3,
            "schedule": "adaptive",
            "entropy_coef": 0.01,
            "estimator_learning_rate": None,
            "estimator_lr_follows_policy": True,
        },
    }


def test_parse_request_accepts_fresh_g1_single_stage_training(tmp_path):
    from openhomie_isaaclab.workflows.elf3_single_stage import parse_request

    request = parse_request(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--device",
            "cuda:0",
            "--seed",
            "42",
            "--num-envs",
            "4096",
            "--iterations",
            "2000",
            "--headless",
        ]
    )
    assert request.command_profile == "g1_single_stage"
    assert request.run_dir == (tmp_path / "run").resolve()


def test_json_safe_serializes_slice_configuration_values():
    from openhomie_isaaclab.workflows.elf3_single_stage import _json_safe

    assert _json_safe(slice(1, 5, 2)) == {"start": 1, "stop": 5, "step": 2}


@pytest.mark.parametrize(
    "argv",
    [
        ["--iterations", "0"],
        ["--device", "cuda"],
        ["--num-envs", "0"],
        ["--checkpoint", "model.pt"],
        ["--resume"],
    ],
)
def test_parse_request_rejects_non_fresh_or_malformed_arguments(argv):
    from openhomie_isaaclab.workflows.elf3_single_stage import parse_request

    with pytest.raises((SystemExit, ValueError)):
        parse_request(argv)
