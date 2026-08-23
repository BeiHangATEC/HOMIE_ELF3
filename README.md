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
| Height-task range | `0.24-0.74 m` | `0.40-1.01 m` |
| Base-height config | `0.74 m` | `1.01 m` |
| Foot-clearance target | `0.14 m` | `0.181 m` |
| Total DOFs / policy actions | `27 / 12` | `28 / 12` |
| Actor / critic observations | `456 / 79` | `468 / 81` |
| Policy joint selection | First 12 asset DOFs | Explicit 12-leg-DOF name mapping |
| Default logger | Weights & Biases | SwanLab cloud project `HomieRL-ELF3` |

The reward weights, domain-randomization ranges, terrain and noise settings, and HIM PPO hyperparameters are unchanged. ELF3 sets both `commands.height_target` and the base-height reward target to `1.01 m`, which is the normal standing and walking command.

### Train

ELF3 samples height tracking and velocity tracking for `45%` of time steps each, with the remaining `10%` used for standing. Within height-tracking tasks, the `[0.40, 0.55] m` and `[0.55, 1.01] m` bands remain equally likely.

Activate the installed HomieRL environment, authenticate SwanLab once, and start the registered `elf3` task from the `HomieRL` directory:

```bash
conda activate homierl
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
swanlab login
cd /path/to/HOMIE_ELF3/HomieRL
python legged_gym/legged_gym/scripts/train.py \
  --task elf3 \
  --num_envs 4096 \
  --headless \
  --experiment_name elf3_formal \
  --run_name elf3_task_mix_45_45_10_v1 \
  --max_iterations 100000 \
  --rl_device cuda:0
```

ELF3 uses SwanLab by default and also keeps local logs under `legged_gym/logs/elf3_formal/`. The default run collects `50` steps per environment per iteration and saves a checkpoint every `200` iterations. Add `--max_iterations N` to override the documented run length.

The latest promoted checkpoint for the `0.40-1.01 m` height range is [`elf3_height_40_55_task_mix_45_45_10_v1_iter11400_20260823T162703Z.pt`](HomieRL/pretrained/elf3/elf3_height_40_55_task_mix_45_45_10_v1_iter11400_20260823T162703Z.pt) (iteration `11400`, saved `2026-08-23T16:27:03.577292483Z`, `9,504,953` bytes, SHA-256 `085ce1758c2e65d7f0914ab2fcc6de16ba71eeda433f8eba550418638dcd6579`). See its [training provenance](HomieRL/pretrained/elf3/elf3_height_40_55_task_mix_45_45_10_v1_iter11400_20260823T162703Z.json).

### Latest Isaac Gym Policy Demo

The [iteration-11400 policy video](HomieRL/recordings/elf3/elf3_height_40_55_task_mix_45_45_10_v1_iter11400_20260823T162703Z.mp4) demonstrates squat, stand, forward/backward motion, left/right lateral motion, and left/right in-place turns. It is a nominal Isaac Gym Preview 4 simulation on a plane—not real-robot validation—and completed the `50.5 s`, `1280x720`, `25 FPS` rollout with zero environment resets. See the [contact sheet](HomieRL/recordings/elf3/elf3_height_40_55_task_mix_45_45_10_v1_iter11400_20260823T162703Z_contact_sheet.png) and [artifact manifest](HomieRL/recordings/elf3/elf3_height_40_55_task_mix_45_45_10_v1_iter11400_20260823T162703Z.manifest.json).

### Play in Isaac Gym

`play.py` accepts four motion values in its final `play(...)` call and writes them before every simulation step. They are source-configured targets, not W/A/S/D or arrow-key controls.

| Value | Meaning | ELF3 training range |
|---|---|---|
| `x_vel` | Body-frame forward (`+`) or backward (`-`) velocity | `-1.0` to `1.0 m/s` |
| `y_vel` | Body-frame left (`+`) or right (`-`) velocity | `-0.5` to `0.5 m/s` |
| `yaw_vel` | Left (`+`) or right (`-`) yaw rate | `-0.5` to `0.5 rad/s` |
| `height` | Base-height command | `0.40` to `1.01 m` |

For example, use the following final call in `legged_gym/legged_gym/scripts/play.py` to command ELF3 to walk forward at `0.5 m/s` while standing at `1.01 m`:

```
play(args, x_vel=0.5, y_vel=0.0, yaw_vel=0.0, height=1.01)
```

Use zero velocity with `height=1.01` to stand, a negative `x_vel` to walk backward, a non-zero `y_vel` to move laterally, a non-zero `yaw_vel` to turn, or `height=0.40` to command the lowest trained squat. The current repository default is zero velocity with `height=0.24`, which is a low squat for G1 and is outside ELF3's trained height range; change it before playing ELF3. The shared environment still runs its four-second command-resampling callback, so a resampled command can appear in one returned observation before the next play-loop iteration restores these values.

The current checkpoint loader expects `./example_model.pt` relative to the `HomieRL` working directory. Select a checkpoint and run:

```
cd path_to_OpenHomie/HomieRL
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export ELF3_CHECKPOINT=pretrained/elf3/elf3_height_40_55_task_mix_45_45_10_v1_iter11400_20260823T162703Z.pt
cp "$ELF3_CHECKPOINT" ./example_model.pt
python legged_gym/legged_gym/scripts/play.py --task elf3 --num_envs 32 --resume --experiment_name elf3 --rl_device cuda:0
```

The policy controls 12 leg joints in left-then-right `hip_y`, `hip_x`, `hip_z`, knee, `ankle_y`, and `ankle_x` order. Each policy output is scaled by `0.25` around the default joint angle before the shared controller computes the leg command. During Isaac Gym play, the environment continues to generate random upper-body position targets. `Esc` exits the viewer and `V` toggles viewer synchronization. With `EXPORT_POLICY=True`, play also exports a TorchScript policy to `legged_gym/logs/elf3/exported/policies/policy.pt`.

### Sim2Sim in MuJoCo

The current MuJoCo Sim2Sim path is G1-only. It hard-codes `MujocoDeploy/g1.yaml`, requires the G1 MJCF model, and assumes the first 12 model joints are the policy-controlled legs. Its default command is `[0.0, 0.0, 0.0]` with `height_cmd: 0.34`, so G1 performs a low stationary squat rather than walking. There is no keyboard locomotion control; change `cmd_init` and `height_cmd` in `g1.yaml` before launch.

After placing the exported G1 TorchScript file at the `policy_path` configured in `g1.yaml`, run:

```
conda activate homierl
pip install mujoco==3.2.3
cd path_to_OpenHomie/MujocoDeploy
python mujoco_deploy_g1.py
```

ELF3 Sim2Sim is **not yet runnable in this repository**, so there is currently no valid ELF3 Sim2Sim command. Do not point `g1.yaml` at an ELF3 policy: ELF3 needs a MuJoCo model and YAML configuration with `468` observations and a compatible `0.40-1.01 m` height command, plus joint indexing or model ordering that preserves its explicit 12 policy DOFs and 16 upper-body DOFs. Name-based indexing is the preferred robust implementation. The current Sim2Sim loader consumes the TorchScript `policy.pt` exported by play, not an ONNX file.



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
