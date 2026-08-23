# ELF3 训练、推理与 Sim2Sim 命令

本文命令适用于包含 `elf3_height`、`elf3_walk`、`elf3_walk_toeout` 和
`sim2sim_elf3.py` 的两阶段训练版本。除特别说明外，所有命令均从仓库的
`HomieRL` 目录执行。

## 1. 环境准备

```bash
conda activate homie
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd /home/hpf/yuelk/HOMIE_ELF3/0821/HOMIE_ELF3/HomieRL
```

首次运行前安装项目依赖：

```bash
pip install -r requirements.txt
pip install -e rsl_rl
pip install -e legged_gym
```

训练日志和 checkpoint 默认保存在 `legged_gym/logs/任务名/运行目录/`。
下文优先使用当前工作区已有的 checkpoint 路径，可以按实际训练结果替换运行目录和
checkpoint 编号。

## 2. 训练

### 2.1 第一阶段：高度与稳定性

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task elf3_height \
  --num_envs 4096 \
  --run_name stage1_height \
  --headless \
  --rl_device cuda:0
```

第一阶段的速度命令固定为零，训练高度变化和机器人稳定性。默认日志目录为
`legged_gym/logs/elf3_height/`。

### 2.2 第二阶段：行走、站立与下蹲

先选择一个第一阶段 checkpoint，再通过 `--pretrained_path` 仅加载网络权重：

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task elf3_walk \
  --pretrained_path legged_gym/logs/elf3_height/Aug21_17-46-20_/model_12000.pt \
  --num_envs 4096 \
  --run_name stage2_walk \
  --headless \
  --rl_device cuda:0
```

第二阶段中 50% 的环境训练零速度下蹲，其余环境训练行走或站立。
`--pretrained_path` 不恢复优化器、迭代编号和课程状态。

### 2.3 可选：外八下蹲微调

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task elf3_walk_toeout \
  --pretrained_path legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/model_10200.pt \
  --run_name toeout15_from_walk10200 \
  --max_iterations 5000 \
  --headless \
  --rl_device cuda:0
```

该任务在第二阶段策略基础上微调下蹲时的足端外八角度。

### 2.4 同一阶段断点续训

继续最新一次第一阶段训练：

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task elf3_height \
  --resume \
  --load_run -1 \
  --checkpoint -1 \
  --headless \
  --rl_device cuda:0
```

继续最新一次第二阶段训练时，将 `--task elf3_height` 改为
`--task elf3_walk`。需要指定某次运行时，将 `-1` 替换为对应的运行目录和
checkpoint 编号。`--resume` 会恢复网络、优化器和迭代状态，不能与
`--pretrained_path` 同时使用。

## 3. Isaac Gym 推理

加载第二阶段 checkpoint 并启动 Isaac Gym viewer：

```bash
python legged_gym/legged_gym/scripts/play.py \
  --task elf3_walk \
  --num_envs 1 \
  --resume \
  --load_run Aug22_11-31-42_squat50_from_height12000 \
  --checkpoint 10200 \
  --rl_device cuda:0
```

运行后会同时打开速度与高度控制窗。第二阶段策略的训练分布要求：行走时高度保持
`1.0 m`，下蹲时三个速度保持为零；不要组合低姿态和非零速度。关闭控制窗可结束
推理。

`play.py` 当前设置 `EXPORT_POLICY=True`，运行时还会导出 TorchScript 策略：

```text
legged_gym/logs/elf3_walk/exported/policies/policy.pt
```

推理第一阶段或外八策略时，将 `--task` 分别改为 `elf3_height` 或
`elf3_walk_toeout`，并使用对应任务日志目录中的运行目录和 checkpoint 编号。

## 4. 导出 ONNX 与 Sim2Sim 契约

sim2sim 使用 ONNX 策略和同名 JSON 契约。可以直接从训练 checkpoint 导出：

```bash
python legged_gym/legged_gym/scripts/export_onnx.py \
  --task elf3_walk \
  --pt-path legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/model_10200.pt \
  --export-path legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/exported/policies/policy.onnx
```

成功后生成：

```text
legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/exported/policies/policy.onnx
legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/exported/policies/policy.contract.json
```

也可以把 `--pt-path` 指向 `play.py` 导出的 `policy.pt`。导出器会校验策略的
`444 -> 12` 接口，并检查 TorchScript 与 ONNX 输出误差。

## 5. MuJoCo Sim2Sim

### 5.1 交互运行

```bash
python legged_gym/legged_gym/scripts/sim2sim_elf3.py \
  --policy legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/exported/policies/policy.onnx \
  --contract legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/exported/policies/policy.contract.json \
  --duration 30
```

省略 `--headless` 时会打开 MuJoCo viewer 和控制窗。控制窗提供 `Walk` 与
`Squat` 模式；从下蹲切换到行走时，机器人会先升到站立高度再执行速度命令。

### 5.2 无界面行走门禁

```bash
python legged_gym/legged_gym/scripts/sim2sim_elf3.py \
  --policy legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/exported/policies/policy.onnx \
  --contract legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/exported/policies/policy.contract.json \
  --mode walk \
  --vx 0.3 \
  --vy 0.0 \
  --yaw 0.0 \
  --height 1.0 \
  --duration 30 \
  --headless \
  --gate \
  --metrics legged_gym/logs/elf3_walk/sim2sim_forward.json
```

### 5.3 无界面下蹲门禁

```bash
python legged_gym/legged_gym/scripts/sim2sim_elf3.py \
  --policy legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/exported/policies/policy.onnx \
  --contract legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/exported/policies/policy.contract.json \
  --mode squat \
  --vx 0.0 \
  --vy 0.0 \
  --yaw 0.0 \
  --height 0.30 \
  --duration 30 \
  --headless \
  --gate \
  --metrics legged_gym/logs/elf3_walk/sim2sim_squat.json
```

`elf3_walk` 的有效命令组合为：`Walk` 模式固定高度 `1.0 m`，`Squat`
模式固定三轴速度为零且高度范围为 `0.30-1.00 m`。越界命令、非法模式组合、
策略哈希不匹配或模型哈希不匹配都会直接报错。

仓库默认使用派生开发模型
`legged_gym/resources/robots/elf3_description/mjcf/elf3_fixed_waist.xml`。
如需使用厂家 MJCF，导出契约时增加：

```bash
--model-path /absolute/path/to/vendor.xml --model-classification vendor
```

sim2sim 命令也必须增加 `--model /absolute/path/to/vendor.xml`，确保模型与契约一致。
