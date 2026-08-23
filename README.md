<br>
<p align="center">
<h1 align="center"><strong>HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit (RL)</strong></h1>
  <p align="center">
    <a href='https://www.qingweiben.com' target='_blank'>Qingwei Ben*</a>, <a href='https://trap-1.github.io/' target='_blank'>Feiyu Jia*</a>, <a href='https://scholar.google.com/citations?user=kYrUfMoAAAAJ&hl=zh-CN' target='_blank'>Jia Zeng</a>, <a href='https://jtdong.com/' target='_blank'>Junting Dong</a>, <a href='https://dahua.site/' target='_blank'>Dahua Lin</a>, <a href='https://oceanpang.github.io/' target='_blank'>Jiangmiao Pang</a>
    <br>
    * Equal Controlbution
    <br>
    Shanghai Artificial Intelligence Laboratory & The Chinese University of Hong Kong
    <br>
  </p>
</p>

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2502.13013-orange)](https://arxiv.org/abs/2502.13013)
[![](https://img.shields.io/badge/Project-%F0%9F%9A%80-pink)](https://homietele.github.io/)
[![](https://img.shields.io/badge/Youtube-🎬-red)](https://www.youtube.com/watch?v=FxkGmjyMc5g&feature=youtu.be)
<!-- [![](https://img.shields.io/badge/bilibili-📹-blue)]() -->


## 🤖 [Demo](https://www.youtube.com/watch?v=FxkGmjyMc5g&feature=youtu.be)

[![demo](teaser.png "demo")]()

</div>


## 🔥 News

- \[2025-02\] We release the [paper](https://arxiv.org/abs/2502.13013) and demos of HOMIE.
- \[2025-02\] We open all of the resources of HOMIE

## 📋 Contents

- [🏠 About](#-about)
- [🔥 News](#-news)
- [📚 Usage](#-usage)
- [🤖 ELF3](#-elf3)
- [📝 TODO List](#-todo)
- [🔗 Citation](#-citation)
- [📄 License](#-license)
- [👏 Acknowledgements](#-acknowledgements)

## 🏠 About
<img src="./overview.png" alt="framework" width="100%" style="position: relative;">
<a name="-about"></a>
This repository is an official implementation of "HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit", which is a novel humanoid teleoperation cockpit composed of a humanoid loco-manipulation policy and an exoskeleton-based hardware system. 

HOMIE enables a single operator to precisely and efficiently control a humanoid robot's full-body movements for diverse loco-manipulation tasks. Integrated into simulation environments, our cockpit also enables seamless teleoperation in virtual settings. Specifically, we introduce three core techniques to our RL-based training framework: upper-body pose curriculum, height tracking reward, and symmetry utilization. These components collectively enhance the robot's physical agility, enabling robust walking, rapid squatting to any required heights, and stable balance maintenance during dynamic upper-body movements, thereby significantly expanding the robot's operational workspace beyond existing solutions. Unlike previous whole-body control methods that depend on motion priors derived from motion capture (MoCap) data, our framework eliminates this dependency, resulting in a more efficient pipeline. 

Our hardware system features isomorphic exoskeleton arms, a pair of motion-sensing gloves, and a pedal. The pedal design for locomotion command acquisition liberates the operator's upper body, enabling simultaneous acquisition of upper-body poses. Since the exoskeleton arms are isomorphic to the controlled robot and each glove has 15 degrees of freedom (DoF), which is more than most existing dexterous hands, we can directly set upper-body joint positions from the exoskeleton readings, dispensing with IK and achieving faster and more accurate teleoperation. Moreover, our gloves can be detached from the arms, allowing them to be reused in systems isomorphic to different robots. The total cost of the hardware system is only \$0.5k, which is significantly lower than that of MoCap devices.

This repository contains three key components of HOMIE:

* **HomieRL**: A novel reinforcement learning (RL)-based training framework that enables different kinds of humanoid robots to walk and squat robustly under any continuously changing upper-body poses.
* **HomieHardware**: It contains all necessary files to reimplement our hardware system, including design files, PCB principle files, and keil code for PCBs. (You are required to fill in a form to get such resources. NOTE: Please make sure that you use your actual name and actual institution when filling the form. People without a valid name or institution will not be considered as appropriate to have access to the resources in order to protect our knowledge priviledge. Individuals without any institutions may contact the authors directly.)
* **HomieDeploy**: It contains all deployment code for both PC that connected to our hardware system and the Unitree G1 with Dex3 hands.

We separate these parts into three different sub-directories, you can view them as three independent repositories. Each sub-directory has its own README, which describes their usage ways and functions. HOMIE is fully open-sourced, however, ***it is strictly forbidden to use HOMIE for any commercial purposes***.

## 📚 Usage
<a name="-usage"></a>

### Prerequisites

We recommend to use our code under the following environment:

- Ubuntu 20.04/22.04 Operating System
- IsaacGym Preview 4.0
  - NVIDIA GPU (RTX 2070 or higher)
  - NVIDIA GPU Driver (recommended version 535.183)
- Conda
  - Python 3.8
- Hardware
  - Unitree G1 with Dex3 Hands
  - Realsense D455 * 1
  - Realsense D435 * 2
### Installation
You should first clone this repository to your Ubuntu computer by running:
```
https://github.com/OpenRobotLab/Homie.git
```
Then you can follow the README.md in each sub-repostory to install all three parts or just one of them.

If you have any questions about the usage of this repository, please feel free to drop an e-mail at **elgceben@gmail.com**, we will respond to it as soon as possible. Or, you can join our discussion wechat group (However, it has over 200 people now, if you would like to join, please add wechat: elgceben with info like "I want to join HOMIE discussion wechat group")

## 🤖 ELF3
<a name="-elf3"></a>

The `elf3` Isaac Gym task reuses the G1 `LeggedRobot` environment and HIM PPO implementation while adapting the robot asset, joint layout, height commands, observation size, and logger.

### Changes from G1

| Item | G1 | ELF3 |
|---|---|---|
| Robot asset | `g1_description/g1.urdf` | `elf3_description/urdf/elf3.urdf` |
| Initial base height | `0.75 m` | `1.01 m` |
| Height-task range | `0.24-0.74 m` | `0.30-1.01 m` |
| Base-height config | `0.74 m` | `1.035 m` |
| Foot-clearance target | `0.14 m` | `0.181 m` |
| Total DOFs / policy actions | `27 / 12` | `26 / 12` |
| Actor / critic observations | `456 / 79` | `444 / 77` |
| Policy joint selection | First 12 asset DOFs | Explicit 12-leg-DOF name mapping |
| Default logger | Weights & Biases | SwanLab cloud project `HomieRL-ELF3` |

The reward weights, domain-randomization ranges, terrain and noise settings, and HIM PPO hyperparameters are unchanged. ELF3 sets `commands.height_target=1.01`; the current height reward follows this command, so `1.01 m`, rather than the separate `1.035 m` base-height field, is the normal standing and walking command.

### Train

Activate the installed HomieRL environment, authenticate SwanLab once, and start the registered `elf3` task from the `HomieRL` directory:

```
conda activate homierl
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
swanlab login
cd path_to_OpenHomie/HomieRL
python legged_gym/legged_gym/scripts/train.py --task elf3 --num_envs 4096 --headless --experiment_name elf3 --run_name elf3_policy --rl_device cuda:0
```

ELF3 uses SwanLab by default and also keeps local logs under `legged_gym/logs/elf3/`. The default run is `100000` iterations, collects `50` steps per environment per iteration, and saves a checkpoint every `200` iterations. Add `--max_iterations N` to override the run length.

### ELF3 两阶段训练

新增任务 `elf3_height` 和 `elf3_walk` 共用同一套 26 自由度模型、444 维 Actor 观测与 12 维腿部动作，原有 `elf3` 混合任务保持不变。

两个阶段默认使用本地 TensorBoard 日志，不要求安装或登录 SwanLab。

第一阶段只训练高度与稳定性。速度命令固定为零，高度命令从 `1.0 m` 开始，每 6 秒在 `0.3-1.0 m` 内重采样目标，并以 `0.20 m/s` 平滑变化。上肢扰动根据 20 秒统计窗内的存活率和高度误差逐步增加；当扰动比例达到 1.0 且连续 5 个窗口达标时，日志中的 `stage1_ready` 变为 1。

```bash
cd HomieRL
python legged_gym/legged_gym/scripts/train.py \
  --task elf3_height \
  --headless
```

第二阶段按 episode 固定任务分组：50% 的环境保持零速度，每 6 秒在 `0.3-1.0 m` 内重采样下蹲目标，并以 `0.20 m/s` 平滑变化；其余 50% 的环境固定高度为 `1.0 m`，其中一半采样全向运动命令、一半原地站立。因此全部环境的期望比例为 50% 下蹲、25% 行走和 25% 站立。人工选择第一阶段 checkpoint 后，通过 `--pretrained_path` 只迁移网络权重；优化器、迭代数和上肢课程状态都会重新开始。

```bash
cd HomieRL
python legged_gym/legged_gym/scripts/train.py \
  --task elf3_walk \
  --pretrained_path legged_gym/logs/elf3_height/<run>/model_<iteration>.pt \
  --headless
```

在已有第二阶段策略上继续微调外八下蹲时，使用独立任务 `elf3_walk_toeout`，不会修改原 `elf3_walk`。该任务保持 50% 下蹲、25% 行走、25% 站立的命令分布和 `444 -> 12` 策略接口。零速度下蹲高度从 `0.735 m` 降到 `0.50 m` 时，左右足端偏航目标由 `0°` 线性增加到 `+15°/-15°`；`0.50 m` 以下保持完整角度。双足中心横向距离仅在超出 `0.20-0.35 m` 时惩罚，不采用越宽奖励越高的形式。

```bash
cd HomieRL
python legged_gym/legged_gym/scripts/train.py \
  --task elf3_walk_toeout \
  --pretrained_path legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/model_10200.pt \
  --run_name toeout15_from_walk10200 \
  --max_iterations 5000 \
  --headless
```

该微调默认使用 `learning_rate=2e-4`、`entropy_coef=0.005`，每 200 次迭代保存一次 checkpoint。`--pretrained_path` 只初始化网络权重，优化器、迭代计数和课程状态从头开始。

同一阶段中断后继续训练时使用 `--resume`。`--load_run -1 --checkpoint -1` 表示选择该任务实验目录下最新的 run 和 checkpoint，也可以显式指定二者。

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task elf3_height \
  --resume \
  --load_run -1 \
  --checkpoint -1 \
  --headless
```

### Play in Isaac Gym

`play.py` 启动 Isaac Gym viewer 后会同时打开一个 Tkinter 控制窗。四个滑条会在每个推理周期更新全部仿真环境的速度与高度命令，数值会限制在 ELF3 的训练范围内。

| 参数 | 含义 | ELF3 训练范围 |
|---|---|---|
| `x_vel` | 机身坐标系前进（`+`）或后退（`-`）速度 | `-0.8` 至 `1.2 m/s` |
| `y_vel` | 机身坐标系向左（`+`）或向右（`-`）速度 | `-0.5` 至 `0.5 m/s` |
| `yaw_vel` | 向左（`+`）或向右（`-`）转向角速度 | `-0.8` 至 `0.8 rad/s` |
| `height` | 机身高度命令 | `0.30` 至 `1.01 m` |

文件末尾的 `play(...)` 参数作为控制窗初始值；当前默认速度为零，高度为 `0.8 m`。例如，以下调用会让控制窗从前进 `0.5 m/s`、高度 `1.01 m` 开始：

```
play(args, x_vel=0.5, y_vel=0.0, yaw_vel=0.0, height=1.01)
```

`Zero speeds` 会把三个速度立即归零并保留当前高度，`Restore defaults` 会恢复 `play(...)` 传入的全部初始值。关闭控制窗会结束播放。播放模式会禁用环境的周期命令重采样，因此滑条命令不会再被随机命令短暂覆盖。

四个滑条面向同时训练速度和高度的 `elf3` 混合策略。`elf3_height` checkpoint 的速度训练命令始终为零。第二阶段 `elf3_walk` checkpoint 包含互斥的行走和下蹲分布：行走时高度固定为 `1.0 m`，下蹲时速度固定为零，不要使用“低姿态 + 非零速度”的训练分布外组合。使用 `--headless` 时不会创建控制窗，推理过程持续使用 `play(...)` 的固定初始值。

播放 checkpoint 时同样使用 `--resume`、`--load_run` 和 `--checkpoint` 选择模型，不再需要复制为 `./example_model.pt`：

```
cd path_to_OpenHomie/HomieRL
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python legged_gym/legged_gym/scripts/play.py \
  --task elf3 \
  --num_envs 32 \
  --resume \
  --experiment_name elf3 \
  --load_run <run> \
  --checkpoint <iteration> \
  --rl_device cuda:0
```

The policy controls 12 leg joints in left-then-right `hip_y`, `hip_x`, `hip_z`, knee, `ankle_y`, and `ankle_x` order. Each policy output is scaled by `0.25` around the default joint angle before the shared controller computes the leg command. During Isaac Gym play, the environment continues to generate random upper-body position targets. `Esc` exits the viewer and `V` toggles viewer synchronization. With `EXPORT_POLICY=True`, play also exports a TorchScript policy to `legged_gym/logs/elf3/exported/policies/policy.pt`.

### Sim2Sim in MuJoCo

G1 仍使用 `MujocoDeploy/g1.yaml`、G1 MJCF 和 TorchScript 策略。其默认命令是 `[0.0, 0.0, 0.0]`，`height_cmd` 为 `0.34`；需要在 `g1.yaml` 中修改固定命令。

After placing the exported G1 TorchScript file at the `policy_path` configured in `g1.yaml`, run:

```
conda activate homierl
pip install mujoco==3.2.3
cd path_to_OpenHomie/MujocoDeploy
python mujoco_deploy_g1.py
```

ELF3 使用独立的 26 自由度 MuJoCo 模型、ONNX 策略和随策略生成的 JSON 契约，不复用 `MujocoDeploy/g1.yaml`。导出器可以直接读取训练 checkpoint 或 `play.py` 导出的 TorchScript，校验 `444 -> 12` 接口、比较 128 组 PyTorch/ONNX 输出，并在同目录生成 `policy.contract.json`。以下命令直接导出第二阶段 `model_10200.pt`：

```bash
cd path_to_OpenHomie/HomieRL
python legged_gym/legged_gym/scripts/export_onnx.py \
  --task elf3_walk \
  --pt-path legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/model_10200.pt \
  --export-path legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/exported/policies/policy.onnx
```

交互运行时省略 `--headless`，控制窗提供 `Walk` 和 `Squat` 两种模式。`Walk` 固定高度为 `1.0 m` 并开放三个速度滑条；`Squat` 将速度归零并开放 `0.30-1.00 m` 高度滑条。高度命令按训练配置的 `0.20 m/s` 平滑变化；从下蹲切回行走时，机器人先以零速度升到 `1.0 m`，再执行速度命令。

```bash
python legged_gym/legged_gym/scripts/sim2sim_elf3.py \
  --policy legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/exported/policies/policy.onnx \
  --contract legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/exported/policies/policy.contract.json \
  --duration 30
```

运行无界面行走门禁：

```bash
python legged_gym/legged_gym/scripts/sim2sim_elf3.py \
  --policy legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/exported/policies/policy.onnx \
  --contract legged_gym/logs/elf3_walk/Aug22_11-31-42_squat50_from_height12000/exported/policies/policy.contract.json \
  --mode walk \
  --vx 0.3 --vy 0.0 --yaw 0.0 --height 1.0 \
  --duration 30 --headless --gate \
  --metrics legged_gym/logs/elf3_walk/sim2sim_forward.json
```

无界面下蹲门禁使用 `--mode squat --vx 0 --vy 0 --yaw 0 --height 0.30`。`elf3_height` 只接受零速度和 `0.30-1.00 m` 高度，`elf3` 接受训练配置中的速度范围和 `0.30-1.01 m` 高度；越界命令、非法模式组合、策略或模型哈希不匹配都会直接失败。复位根高度由契约独立设置为 `1.05552264 m`，不能用策略高度命令替代。

导出外八策略时将任务改为 `elf3_walk_toeout`。生成的契约会额外记录外八高度区间、左右最大偏航角和双足距离范围。Sim2Sim 不会向 `hip_z` 注入偏置，只执行导出的策略；指标 JSON 会输出左右足端实际/目标偏航、稳定段偏航 RMSE 和双足中心横向距离。外八契约在 `Squat` 门禁中还要求每只脚偏航 RMSE 不超过 `5°`、足距全程保持在 `0.20-0.35 m`，并在 `0.50 m` 及以下达到左 `+15±3°`、右 `-15±3°`。

```bash
python legged_gym/legged_gym/scripts/export_onnx.py \
  --task elf3_walk_toeout \
  --pt-path legged_gym/logs/elf3_walk_toeout/<run>/model_<iteration>.pt \
  --export-path legged_gym/logs/elf3_walk_toeout/<run>/exported/policies/policy.onnx

python legged_gym/legged_gym/scripts/sim2sim_elf3.py \
  --policy legged_gym/logs/elf3_walk_toeout/<run>/exported/policies/policy.onnx \
  --contract legged_gym/logs/elf3_walk_toeout/<run>/exported/policies/policy.contract.json \
  --mode squat --vx 0 --vy 0 --yaw 0 --height 0.50 \
  --duration 30 --headless --gate \
  --metrics legged_gym/logs/elf3_walk_toeout/sim2sim_squat_050.json
```

仓库中的 `elf3_fixed_waist.xml` 是根据当前 URDF 和已校正足底几何生成的派生开发模型，不是厂家验证的最终 MJCF。它通过门禁只能说明开发阶段 Sim2Sim 可用；取得厂家 26 自由度 MJCF 后，使用 `--model-path <vendor.xml> --model-classification vendor` 重新导出契约并执行相同门禁。



## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@article{ben2025homie,
  title={HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit},
  author={Ben, Qingwei and Jia, Feiyu and Zeng, Jia and Dong, Junting and Lin, Dahua and Pang, Jiangmiao},
  journal={arXiv preprint arXiv:2502.13013},
  year={2025}
}
```

</details>

## 📝 TODO List

- \[x\] Release the paper with demos.
- \[x\] Release all necessary code.
- \[ \] Release training code for more robots.
- \[ \] Upgrade the low-level control policy for more complex terrains and more robots.

## 📄 License

All code of HOMIE is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License </a><a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>. It is strictly forbidden to use it for commercial purposes before asking our team.

## 👏 Acknowledgements


- [RSL_RL](https://github.com/leggedrobotics/rsl_rl): We use `rsl_rl` library to train the control policies for legged robots.
- [Legged_gym](https://github.com/leggedrobotics/rsl_rl): We use `legged_gym` library to train the control policies for legged robots.
- [HIMLoco](https://github.com/OpenRobotLab/HIMLoco): We use `HIMLoco` library as our codebase.
- [Walk-These-Ways](https://github.com/leggedrobotics/rsl_rl): Our robot deployment code is based on `walk-these-ways`.
- [Unitree SDK2](https://github.com/leggedrobotics/rsl_rl): We use `Unitree SDK2` library to control the robot.
- [HomunCulus](https://github.com/nepyope/Project-Homunculus): Our glove design refers to the principle of `HomunCulus` such as using `Hall sensors`.
