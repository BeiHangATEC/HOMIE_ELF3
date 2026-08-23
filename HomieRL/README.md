<br>
<div align="center">
  <h1><strong>HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit (RL)</strong></h1>
  <p>
    <a href="https://www.qingweiben.com" target="_blank">Qingwei Ben*</a>,
    <a href="https://trap-1.github.io/" target="_blank">Feiyu Jia*</a>,
    <a href="https://scholar.google.com/citations?user=kYrUfMoAAAAJ&hl=zh-CN" target="_blank">Jia Zeng</a>,
    <a href="https://jtdong.com/" target="_blank">Junting Dong</a>,
    <a href="https://dahua.site/" target="_blank">Dahua Lin</a>,
    <a href="https://oceanpang.github.io/" target="_blank">Jiangmiao Pang</a>
    <br>
    * Equal Contribution
    <br>
    Shanghai Artificial Intelligence Laboratory & The Chinese University of Hong Kong
  </p>

  [![arXiv](https://img.shields.io/badge/arXiv-2502.13013-orange)](https://arxiv.org/abs/2502.13013)

  <img src="./rl.png" alt="HOMIE reinforcement-learning framework" width="100%" />
</div>

## 📋 Contents

- [🏠 Description](#description)
- [📚 Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Train](#train)
  - [Logging and Checkpoints](#logging-and-checkpoints)
  - [ELF3 Task Distribution](#elf3-task-distribution-and-upper-body-control)
  - [Play](#play)
  - [Export Policy to ONNX](#export-policy-to-onnx)
  - [Troubleshooting](#troubleshooting)
- [🔗 Citation](#citation)
- [📄 License](#license)
- [👏 Acknowledgements](#acknowledgements)

## 🏠 Description

This repository is an official implementation of the reinforcement-learning framework proposed in [HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit](https://arxiv.org/abs/2502.13013). It is built on [Isaac Gym](https://developer.nvidia.com/isaac-gym) and [HIMLoco](https://github.com/OpenRobotLab/HIMLoco).

The registered tasks are `g1` and `elf3`. They train humanoid policies to walk and squat under continuously changing upper-body poses.

The framework has three key components:

- **Upper-body Pose Curriculum**: gradually expands upper-body targets while the robot learns to balance.
- **Height Reward Tracking**: uses height and knee-related rewards to track squat targets precisely.
- **Symmetry Utilization**: augments training data with robot symmetry and optimizes the symmetry loss $L_{sym}$.

## 📚 Getting Started

All commands below are intended to run from the `HomieRL` repository root.

### Prerequisites

We recommend the following environment:

- Ubuntu 20.04 or 22.04
- Isaac Gym Preview 4
  - NVIDIA GPU (RTX 2070 or newer recommended)
  - NVIDIA driver compatible with the installed CUDA/PyTorch build
- Conda with Python 3.8

### Installation

1. Create the Conda environment and install Isaac Gym:

   ```bash
   conda create -n homierl python=3.8
   conda activate homierl
   cd <path-to-isaac-gym>/python
   pip install -e .
   ```

2. Install this repository:

   ```bash
   git clone https://github.com/OpenRobotLab/HomieRL.git
   cd HomieRL
   pip install -r requirements.txt
   pip install -e rsl_rl
   pip install -e legged_gym
   ```

3. Before training or play, make the Conda binaries and shared libraries available:

   ```bash
   conda activate homierl
   export PATH="$CONDA_PREFIX/bin:$PATH"
   export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
   ```

### Train

The training entry point is `legged_gym/legged_gym/scripts/train.py`.

#### G1

G1 uses W&B by default. Set `wandb_project` and `wandb_user` in `legged_gym/legged_gym/envs/g1/g1_29dof_config.py` before starting a cloud run.

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task g1 \
  --num_envs 4096 \
  --headless \
  --experiment_name g1_formal \
  --run_name g1_policy_v1 \
  --max_iterations 100000 \
  --rl_device cuda:0
```

#### ELF3 with SwanLab

ELF3 defaults to SwanLab cloud logging with project `HomieRL-ELF3`. Authenticate once before the first cloud run:

```bash
swanlab login
```

Start formal training:

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task elf3 \
  --num_envs 4096 \
  --headless \
  --experiment_name elf3_formal \
  --run_name elf3_task_mix_45_45_10_v1 \
  --max_iterations 100000 \
  --rl_device cuda:0
```

The current argument parser binds the simulator to `--rl_device`; it does not provide independent RL and simulation device selection. If 4096 environments exceed available GPU memory, reduce `--num_envs` to 2048 or 1024.

Supported training overrides include:

| Option | Meaning |
|---|---|
| `--task` | Registered task: `g1` or `elf3` |
| `--num_envs` | Number of parallel simulation environments |
| `--headless` | Disable the Isaac Gym viewer |
| `--experiment_name` | Local log-directory group |
| `--run_name` | Run name appended to local and cloud metadata |
| `--max_iterations` | Maximum learning iterations |
| `--seed` | Random seed |
| `--rl_device` | Device used by both the policy runner and simulator |

`--resume`, `--load_run`, and `--checkpoint` are parsed, but the current loader still uses the hard-coded `./example_model.pt` path. See [Play](#play) before using resume-related options.

#### Smoke test

A one-iteration smoke test still creates local logs/checkpoints and uses the task's configured cloud logger. To keep the test local, temporarily set `Elf3RoughCfgPPO.runner.logger = "tensorboard"` in `elf3_config.py`, then run:

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task elf3 \
  --num_envs 64 \
  --headless \
  --experiment_name elf3_smoke \
  --run_name smoke \
  --max_iterations 1 \
  --rl_device cuda:0
```

### Logging and Checkpoints

Local run files are written to:

```text
legged_gym/logs/<experiment_name>/<MonDD_HH-MM-SS>_<run_name>/
```

G1 defaults to W&B; ELF3 defaults to SwanLab cloud mode; both can be changed to `tensorboard` in their runner configuration. Cloud logger selection is a configuration value, not a CLI option.

Checkpoints are named `model_<iteration>.pt`. The G1 and ELF3 configurations save periodically every 200 iterations and once again when training exits. Typical logged groups include `Episode/*`, `Loss/*`, `Policy/*`, `Perf/*`, `Train/mean_reward`, and `Train/mean_episode_length`.

To inspect local event files:

```bash
tensorboard --logdir legged_gym/logs
```

### ELF3 Task Distribution and Upper-body Control

The ELF3 policy controls 12 leg joints and observes all 28 movable joints. The remaining 16 upper-body joints receive position targets generated by the environment.

The default command distribution is:

- height tracking: `45%` of time steps;
- velocity tracking: `45%` of time steps;
- standing: `10%` of time steps;
- within height-tracking tasks, `[0.30, 0.50] m` and `[0.50, 1.01] m` are sampled with equal probability;
- x velocity is limited to `[-1.0, 1.0] m/s`; y velocity and yaw rate remain limited to `[-0.5, 0.5] m/s` and `[-0.5, 0.5] rad/s`, respectively.

At full upper-body curriculum:

- `waist_y_joint` uses the configured absolute target range `[-0.50, 0.50] rad`;
- `waist_z_joint` is locked to `0 rad` during reset, action assembly, delayed control, and final position targeting;
- upper-body joints not listed in `upper_body_joint_position_ranges` or `upper_body_locked_joint_positions` retain their URDF target limits.

ELF3 logs the time-step fractions spent in each task plus `Episode/height_mae`, `Episode/low_height_mae`, and `Episode/high_height_mae`.

### Play

The latest promoted checkpoint for the `45% / 45% / 10%` task-mix run is `pretrained/elf3/elf3_task_mix_45_45_10_v1_iter13200_20260823T024427Z.pt`. It was saved at `2026-08-23T02:44:27.888651Z`, promoted on `2026-08-23`, contains iteration `13200`, is `9,504,953` bytes, and has SHA-256 `45d026c3e68754c19e191f996ce3cd4a07966b257b5dc924f3491faa65ca37a5`. Its portable provenance record is the adjacent `.json` file. The older height-lock checkpoint remains available at `pretrained/elf3/model_13400.pt`.

The current play path has several runtime constraints:

- `task_registry.py` loads `./example_model.pt` when resume is enabled;
- the path is relative to the current working directory, so for ELF3 copy the promoted task-mix checkpoint to `./example_model.pt` in the `HomieRL` root; use a task-compatible checkpoint for G1;
- `--load_run` and `--checkpoint` do not currently change that path;
- x/y/yaw/height targets are values in the final `play(...)` call, not CLI options; the current script uses `x=0`, `y=0`, `yaw=0`, and `height=0.24`;
- `EXPORT_POLICY = True`, so a successful play run also writes a JIT policy to `legged_gym/logs/<experiment_name>/exported/policies/policy.pt`.

Run the included ELF3 checkpoint with a viewer:

```bash
cp pretrained/elf3/elf3_task_mix_45_45_10_v1_iter13200_20260823T024427Z.pt ./example_model.pt
python legged_gym/legged_gym/scripts/play.py --task elf3 --num_envs 1 --resume --rl_device cuda:0
```

For G1, place a G1-compatible checkpoint at `./example_model.pt` and replace `--task elf3` with `--task g1`.

Use `--headless` for rollout/export without a viewer. Headless play does not record video or print an evaluation summary.

### Export Policy to ONNX

`export_onnx.py` converts an exported JIT policy—not a raw training `model_<iteration>.pt` checkpoint—to ONNX. Run play first to create `policy.pt`, then execute:

```bash
python legged_gym/legged_gym/scripts/export_onnx.py \
  --task elf3 \
  --pt-path legged_gym/logs/elf3_formal/exported/policies/policy.pt \
  --export-path legged_gym/logs/elf3_formal/exported/policies/policy.onnx
```

Use `--task g1` and the matching G1 JIT policy when exporting a G1 model.

### Troubleshooting

If Isaac Gym cannot find `libpython3.8.so.1.0`, ensure the active Conda environment's library directory is on `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

If `gymtorch` reports that Ninja is unavailable, ensure the active environment's binary directory is on `PATH`:

```bash
export PATH="$CONDA_PREFIX/bin:$PATH"
```

Before starting a large run, check available GPU memory:

```bash
nvidia-smi
```

## 🔗 Citation

If you find this work helpful, please cite:

```bibtex
@article{ben2025homie,
  title={HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit},
  author={Ben, Qingwei and Jia, Feiyu and Zeng, Jia and Dong, Junting and Lin, Dahua and Pang, Jiangmiao},
  journal={arXiv preprint arXiv:2502.13013},
  year={2025}
}
```

## 📄 License

The repository's original work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE). Bundled third-party components retain their respective license notices. Commercial use requires permission from the project authors.

## 👏 Acknowledgements

- [RSL_RL](https://github.com/leggedrobotics/rsl_rl): PPO implementation used by this project.
- [Legged Gym](https://github.com/leggedrobotics/legged_gym): Isaac Gym environments and locomotion framework.
- [HIMLoco](https://github.com/OpenRobotLab/HIMLoco): base code and learning framework.
