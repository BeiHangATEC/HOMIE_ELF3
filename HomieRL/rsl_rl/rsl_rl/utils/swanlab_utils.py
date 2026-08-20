# Copyright 2021 ETH Zurich, NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from numbers import Integral

from torch.utils.tensorboard import SummaryWriter

try:
    import swanlab
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "SwanLab is required when runner.logger is set to 'swanlab'."
    ) from exc


def class_to_dict(obj):
    """Serialize nested configuration classes without importing legged_gym."""
    if not hasattr(obj, "__dict__"):
        return obj

    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        value = getattr(obj, key)
        if isinstance(value, list):
            result[key] = [class_to_dict(item) for item in value]
        else:
            result[key] = class_to_dict(value)
    return result


class SwanLabSummaryWriter(SummaryWriter):
    """Mirror TensorBoard scalars and training configuration to SwanLab."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir=log_dir, flush_secs=flush_secs)

        project = cfg.get("swanlab_project") or cfg.get("experiment_name") or "HomieRL"
        workspace = cfg.get("swanlab_workspace") or None
        mode = cfg.get("swanlab_mode", "cloud")
        swanlab_log_dir = os.path.join(log_dir, "swanlab")
        os.makedirs(swanlab_log_dir, exist_ok=True)

        self.run = swanlab.init(
            project=project,
            workspace=workspace,
            experiment_name=os.path.basename(os.path.normpath(log_dir)),
            logdir=swanlab_log_dir,
            mode=mode,
        )
        self._stopped = False

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.run.config.update({
            "env_cfg": class_to_dict(env_cfg),
            "runner_cfg": runner_cfg,
            "alg_cfg": alg_cfg,
            "policy_cfg": policy_cfg,
        })

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        tensorboard_step = global_step + 1 if global_step is not None else None
        super().add_scalar(
            tag,
            scalar_value,
            global_step=tensorboard_step,
            walltime=walltime,
            new_style=new_style,
        )

        value = scalar_value
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            value = value.item()

        swanlab_step = int(global_step) + 1 if isinstance(global_step, Integral) else None
        self.run.log({tag: value}, step=swanlab_step)

    def stop(self, error=None):
        if self._stopped:
            return
        self.flush()
        super().close()
        if error is None:
            self.run.finish()
        else:
            self.run.finish(state=swanlab.State.CRASHED, error=error)
        self._stopped = True
