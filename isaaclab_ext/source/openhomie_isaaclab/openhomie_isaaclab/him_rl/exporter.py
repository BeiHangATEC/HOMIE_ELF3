"""Inference-only HIM policy export and independent parity verification."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F


class HIMPolicyExporter(nn.Module):
    """Deep-copied estimator-source and actor inference graph."""

    def __init__(self, policy) -> None:
        super().__init__()
        self.history_dim = policy.num_one_step_obs * policy.actor_history_length
        self.one_step_obs_dim = policy.num_one_step_obs
        self.velocity_dim = policy.estimator.layout.velocity_dim
        self.latent_dim = policy.estimator.latent_dim
        self.actor_obs_normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        self.estimator_encoder = copy.deepcopy(policy.estimator.source_encoder)
        self.actor = copy.deepcopy(policy.actor)
        self.eval()

    def forward(self, observation_history: torch.Tensor) -> torch.Tensor:
        if observation_history.dim() != 2:
            raise ValueError("observation history must be rank-2")
        if observation_history.shape[-1] != self.history_dim:
            raise ValueError("observation history width does not match the policy history dimension")
        if not torch.isfinite(observation_history).all():
            raise ValueError("observation history must be finite")
        normalized_history = self.actor_obs_normalizer(observation_history)
        encoded = self.estimator_encoder(normalized_history)
        velocity = encoded[..., : self.velocity_dim]
        latent = F.normalize(encoded[..., self.velocity_dim :], dim=-1)
        latest_frame = normalized_history[..., -self.one_step_obs_dim :]
        actions = self.actor(torch.cat((latest_frame, velocity, latent), dim=-1))
        if not torch.isfinite(actions).all():
            raise RuntimeError("exported policy produced non-finite actions")
        return actions


def _validate_output_path(path: str | Path) -> Path:
    output = Path(path)
    if output.exists() and output.is_dir():
        raise IsADirectoryError(str(output))
    if not output.parent.is_dir():
        raise FileNotFoundError(str(output.parent))
    return output


def _model_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters())
    return parameter.device, parameter.dtype


def export_him_policy_torchscript(policy, path: str | Path) -> HIMPolicyExporter:
    output = _validate_output_path(path)
    exporter = HIMPolicyExporter(policy)
    scripted = torch.jit.script(exporter)
    scripted.save(str(output))
    return exporter


def export_him_policy_onnx(
    policy,
    path: str | Path,
    *,
    opset_version: int = 17,
) -> HIMPolicyExporter:
    output = _validate_output_path(path)
    exporter = HIMPolicyExporter(policy)
    device, dtype = _model_device_dtype(exporter)
    sample = torch.zeros(1, exporter.history_dim, device=device, dtype=dtype)
    torch.onnx.export(
        exporter,
        sample,
        str(output),
        input_names=["observation_history"],
        output_names=["actions"],
        dynamic_axes={"observation_history": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=opset_version,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(output))
    return exporter


def _validate_threshold(max_abs_error: float) -> float:
    if (
        isinstance(max_abs_error, bool)
        or not isinstance(max_abs_error, (int, float))
        or not math.isfinite(max_abs_error)
        or max_abs_error < 0
    ):
        raise ValueError("maximum absolute error must be finite and non-negative")
    return float(max_abs_error)


def _validated_histories(histories: Iterable[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    values = tuple(histories)
    if not values:
        raise ValueError("at least one parity history is required")
    for history in values:
        if history.dim() != 2 or not torch.isfinite(history).all():
            raise ValueError("parity histories must be finite rank-2 tensors")
    return values


def verify_him_policy_torchscript(
    exporter: HIMPolicyExporter,
    path: str | Path,
    histories: Iterable[torch.Tensor],
    *,
    max_abs_error: float,
) -> float:
    threshold = _validate_threshold(max_abs_error)
    samples = _validated_histories(histories)
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(str(model_path))
    loaded = torch.jit.load(str(model_path), map_location="cpu")
    loaded.eval()
    maximum = 0.0
    with torch.inference_mode():
        for history in samples:
            expected = exporter(history)
            actual = loaded(history.cpu()).to(expected.device)
            if actual.shape != expected.shape or not torch.isfinite(actual).all():
                raise RuntimeError("TorchScript parity output is invalid")
            maximum = max(maximum, (actual - expected).abs().max().item())
    if maximum > threshold:
        raise RuntimeError(f"TorchScript parity error {maximum} exceeds {threshold}")
    return maximum


def verify_him_policy_onnx(
    exporter: HIMPolicyExporter,
    path: str | Path,
    histories: Iterable[torch.Tensor],
    *,
    max_abs_error: float,
) -> float:
    threshold = _validate_threshold(max_abs_error)
    samples = _validated_histories(histories)
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(str(model_path))
    onnx.checker.check_model(onnx.load(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    maximum = 0.0
    with torch.inference_mode():
        for history in samples:
            expected = exporter(history).detach().cpu().numpy()
            actual = session.run(None, {"observation_history": history.detach().cpu().numpy()})[0]
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise RuntimeError("ONNX parity output is invalid")
            maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    if maximum > threshold:
        raise RuntimeError(f"ONNX parity error {maximum} exceeds {threshold}")
    return maximum
