"""将 ELF3 HOMIE checkpoint 导出为 ONNX，并可安全替换 ROS2 部署模型。"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from export_policy_jit_from_checkpoint import ExportedHIMPolicy, build_actor_critic
from legged_gym.envs.g1.elf3_dof31_r1_config import Elf3Dof31R1Contract
from legged_gym.envs.g1.elf3_dof31_r2_config import Elf3Dof31R2Contract


OBS_DIM = 504
ACTION_DIM = 12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_checkpoint(checkpoint_path: Path, jit_path: Path, onnx_path: Path) -> torch.jit.ScriptModule:
    actor_critic = build_actor_critic()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
    actor_critic.eval()

    policy = ExportedHIMPolicy(actor_critic, one_step_obs=84).cpu().eval()
    scripted = torch.jit.script(policy)
    jit_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(jit_path))

    dummy_input = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        scripted,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    return scripted


def verify(scripted: torch.jit.ScriptModule, onnx_path: Path) -> None:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    if model_input.shape[1] != OBS_DIM or model_output.shape[1] != ACTION_DIM:
        raise RuntimeError(
            f"ONNX 契约错误：input={model_input.shape}, output={model_output.shape}"
        )

    generator = torch.Generator().manual_seed(1)
    samples = torch.randn(128, OBS_DIM, generator=generator)
    with torch.no_grad():
        expected = scripted(samples).cpu().numpy()
    actual = session.run([model_output.name], {model_input.name: samples.numpy()})[0]
    if actual.shape != (128, ACTION_DIM) or not np.isfinite(actual).all():
        raise RuntimeError(f"ONNX 输出无效：shape={actual.shape}")
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error > 5e-5:
        raise RuntimeError(f"TorchScript/ONNX 最大误差过大：{max_error:.9g}")
    print(f"TorchScript/ONNX 最大动作误差：{max_error:.9g}")


def replace_model(exported_onnx: Path, target: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) != sha256(exported_onnx):
        backup = target.with_name(f"{target.name}.backup_{timestamp}")
        shutil.copy2(target, backup)
        print(f"已备份：{backup}")
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copy2(exported_onnx, temporary)
    temporary.replace(target)
    print(f"已替换：{target}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    default_checkpoint = (
        repo_root
        / "legged_gym/logs/elf3_dof31/"
        "Aug10_22-23-04_stage8_8192env_2000iter_0945_to_0974/model_1999.pt"
    )
    deploy_root = Path("/home/wyg/ATEC-elf3/bxi_rl_controller_ros2_example")
    source_model = deploy_root / "src/bxi_example_py_elf3/data/homie/model_25999.onnx"
    install_model = deploy_root / "install/share/bxi_example_py_elf3/data/homie/model_25999.onnx"

    parser = argparse.ArgumentParser(description="导出 ELF3 Stage8 checkpoint 并替换 ROS2 HOMIE ONNX")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--contract-stage", choices=("R1", "R2"), default="R1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replace", action="store_true", help="验证通过后备份并替换当前部署模型，然后重新构建 ROS2 工作区")
    parser.add_argument("--source-model", type=Path, default=source_model)
    parser.add_argument("--install-model", type=Path, default=install_model)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint}")
    output_dir = args.output_dir or checkpoint.parent / "exported/policies"
    output_dir = output_dir.expanduser().resolve()
    jit_path = output_dir / "policy.pt"
    onnx_path = output_dir / "model_1999.onnx"

    print(f"Checkpoint：{checkpoint}")
    print(f"Checkpoint SHA-256：{sha256(checkpoint)}")
    scripted = export_checkpoint(checkpoint, jit_path, onnx_path)
    verify(scripted, onnx_path)
    print(f"TorchScript：{jit_path}")
    print(f"ONNX：{onnx_path}")
    print(f"ONNX SHA-256：{sha256(onnx_path)}")
    contract = Elf3Dof31R1Contract if args.contract_stage == "R1" else Elf3Dof31R2Contract
    contract_path = output_dir / f"{args.contract_stage.lower()}_contract.json"
    contract_data = {
        "contract_stage": args.contract_stage,
        "body_height_definition": contract.body_height_definition,
        "body_height_command_m": contract.body_height_command_m,
        "calibrated_standing_height_m": contract.calibrated_standing_height_m,
        "isaac_reset_root_world_z_m": contract.isaac_reset_root_world_z_m,
        "mujoco_reset_root_world_z_m": contract.mujoco_reset_root_world_z_m,
        "ankle_sole_distance_m": contract.ankle_sole_distance_m,
        "default_pose_rad": contract.default_pose_rad,
        "feet_distance_soft_window_m": contract.feet_distance_soft_window_m,
        "knee_distance_soft_window_m": contract.knee_distance_soft_window_m,
    }
    if args.contract_stage == "R2":
        contract_data.update({
            "standing_height_command_m": contract.standing_height_command_m,
            "max_relative_crouch_depth_m": contract.max_relative_crouch_depth_m,
            "minimum_height_command_m": contract.minimum_height_command_m,
            "initialization": contract.initialization,
        })
    contract_path.write_text(json.dumps(contract_data, indent=2) + "\n", encoding="utf-8")
    print(f"{args.contract_stage} contract：{contract_path}")

    if args.replace:
        replace_model(onnx_path, args.source_model)
        source_hash = sha256(args.source_model)
        if source_hash != sha256(onnx_path):
            raise RuntimeError("源码部署模型替换后的 SHA-256 不一致")
        build_script = args.source_model.parents[4] / "homie_sim2sim.sh"
        subprocess.run([str(build_script), "build"], check=True)
        install_hash = sha256(args.install_model)
        if install_hash != source_hash:
            raise RuntimeError("构建后的 install 模型与源码模型 SHA-256 不一致")
        print(f"部署模型 SHA-256：{source_hash}")
    else:
        print("未替换部署模型；确认后添加 --replace。")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"导出失败：{error}", file=sys.stderr)
        raise
