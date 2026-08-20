"""
This file is used to transfer a .pt file to a .onnx file
"""
import argparse
import os

import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.utils import task_registry

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

def get_args():
    parser = argparse.ArgumentParser(description="Export a task's JIT policy to ONNX")
    parser.add_argument("--task", required=True, choices=sorted(task_registry.task_classes.keys()))
    parser.add_argument("--pt-path", required=True, help="Path to the exported JIT policy")
    parser.add_argument("--export-path", required=True, help="Destination .onnx path")
    return parser.parse_args()

def main():
    args = get_args()
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    jit_model = torch.jit.load(args.pt_path, map_location="cpu")
    jit_model.eval()
    dummy_input = torch.randn(1, env_cfg.env.num_observations, device="cpu")
    export_jit_to_onnx(jit_model, args.export_path, dummy_input)

if __name__ == "__main__":
    main()
