"""
This file is used to transfer a .pt file to a .onnx file
"""
import argparse
import os
from pathlib import Path

import isaacgym
import numpy as np
import onnxruntime as ort
import torch

from legged_gym.envs import *
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import class_to_dict, task_registry
from legged_gym.utils.helpers import PolicyExporterHIM
from legged_gym.utils.sim2sim_contract import (
    MODEL_CLASSIFICATIONS,
    SUPPORTED_TASKS,
    build_contract,
    sha256_file,
    write_contract,
)
from rsl_rl.modules import HIMActorCritic


def load_policy_source(path, env_cfg, train_cfg):
    try:
        model = torch.jit.load(str(path), map_location="cpu")
        model.eval()
        return model, "torchscript"
    except RuntimeError as jit_error:
        checkpoint = torch.load(str(path), map_location="cpu")
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise RuntimeError(
                f"Policy source is neither TorchScript nor a training checkpoint: {path}"
            ) from jit_error

    policy_cfg = class_to_dict(train_cfg.policy)
    actor_critic = HIMActorCritic(
        int(env_cfg.env.num_observations),
        int(env_cfg.env.num_privileged_obs),
        int(env_cfg.env.num_one_step_observations),
        int(env_cfg.env.num_one_step_privileged_obs),
        int(env_cfg.env.num_actor_history),
        int(env_cfg.env.num_critic_history),
        int(env_cfg.env.num_actions),
        **policy_cfg,
    ).cpu()
    actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
    actor_critic.eval()
    model = torch.jit.script(PolicyExporterHIM(actor_critic).cpu().eval())
    return model, "training_checkpoint"

def export_jit_to_onnx(jit_model, path, dummy_input):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # export jit to onnx
    torch.onnx.export(
        jit_model,                  
        dummy_input,            
        path,                       
        export_params=True,         
        opset_version=11,           
        do_constant_folding=True,   
        input_names=['input'],      
        output_names=['output'],    
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported JIT model to ONNX at: {path}")


def verify_onnx(jit_model, path, input_dim, output_dim):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    if model_input.shape[1] != input_dim or model_output.shape[1] != output_dim:
        raise RuntimeError(
            f"ONNX contract mismatch: input={model_input.shape}, output={model_output.shape}"
        )

    generator = torch.Generator().manual_seed(1)
    samples = torch.randn(128, input_dim, generator=generator)
    with torch.no_grad():
        expected = jit_model(samples).cpu().numpy()
    actual = session.run(
        [model_output.name],
        {model_input.name: np.ascontiguousarray(samples.numpy())},
    )[0]
    if expected.shape != (128, output_dim) or not np.isfinite(expected).all():
        raise RuntimeError(f"TorchScript output is invalid: shape={expected.shape}")
    if actual.shape != (128, output_dim) or not np.isfinite(actual).all():
        raise RuntimeError(f"ONNX output is invalid: shape={actual.shape}")
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error > 5.0e-5:
        raise RuntimeError(f"TorchScript/ONNX max action error is too large: {max_error:.9g}")
    print(f"TorchScript/ONNX max action error: {max_error:.9g}")
    return max_error

def get_args():
    parser = argparse.ArgumentParser(
        description="Export a task checkpoint or TorchScript policy to ONNX"
    )
    parser.add_argument("--task", required=True, choices=sorted(task_registry.task_classes.keys()))
    parser.add_argument(
        "--pt-path",
        required=True,
        help="Path to a TorchScript policy or training checkpoint",
    )
    parser.add_argument("--export-path", required=True, help="Destination .onnx path")
    parser.add_argument(
        "--contract-path",
        help="ELF3 contract destination; defaults to <export-path-without-suffix>.contract.json",
    )
    parser.add_argument(
        "--model-path",
        default=str(
            Path(LEGGED_GYM_ROOT_DIR)
            / "resources/robots/elf3_description/mjcf/elf3_fixed_waist.xml"
        ),
        help="ELF3 MuJoCo model recorded in the generated contract",
    )
    parser.add_argument(
        "--model-classification",
        choices=sorted(MODEL_CLASSIFICATIONS),
        default="derived",
        help="Use vendor only after the supplied MJCF has been vendor validated",
    )
    return parser.parse_args()

def main():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    pt_path = Path(args.pt_path).expanduser().resolve()
    export_path = Path(args.export_path).expanduser().resolve()
    if not pt_path.is_file():
        raise FileNotFoundError(f"Policy source does not exist: {pt_path}")
    jit_model, source_format = load_policy_source(pt_path, env_cfg, train_cfg)
    dummy_input = torch.randn(1, env_cfg.env.num_observations, device="cpu")
    export_jit_to_onnx(jit_model, str(export_path), dummy_input)
    verify_onnx(
        jit_model,
        export_path,
        int(env_cfg.env.num_observations),
        int(env_cfg.env.num_actions),
    )

    if args.task in SUPPORTED_TASKS:
        model_path = Path(args.model_path).expanduser().resolve()
        urdf_path = (
            Path(LEGGED_GYM_ROOT_DIR)
            / "resources/robots/elf3_description/urdf/elf3.urdf"
        ).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"ELF3 MuJoCo model does not exist: {model_path}")
        contract = build_contract(
            args.task,
            env_cfg,
            export_path,
            model_path,
            urdf_path,
            model_classification=MODEL_CLASSIFICATIONS[args.model_classification],
        )
        contract["policy"]["source_torchscript_sha256"] = sha256_file(pt_path)
        contract["policy"]["source_format"] = source_format
        contract["policy"]["source_sha256"] = sha256_file(pt_path)
        if source_format == "training_checkpoint":
            contract["policy"].pop("source_torchscript_sha256", None)
        contract_path = (
            Path(args.contract_path).expanduser().resolve()
            if args.contract_path
            else export_path.with_suffix(".contract.json")
        )
        write_contract(contract, contract_path)
        print(f"ELF3 Sim2Sim contract: {contract_path}")

if __name__ == "__main__":
    main()
