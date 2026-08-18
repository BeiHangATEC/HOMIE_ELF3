# ELF3 / Isaac Lab 适配总体计划

状态：M0-M2 已完成并验收，M3 待执行
规划者：Fable 5 ｜ 执行者：Sonnet 5
前置阅读：`CLAUDE.md`（不变量与红线）

## 目标

在 upstream OpenHomie 上新增 ELF3 机器人的 HOMIE 训练支持，迁移到
Isaac Sim 5.1 + Isaac Lab 2.3.2，跑通 train / play / sim2sim(MuJoCo)。

## 成功判据（唯一硬标准）

`mean_episode_length` 随训练**显著上升**（从 ~40 步涨到数百步），
且 play 里机器人能站住、能按速度命令行走、能按高度命令下蹲。

上一版失败正是因为只验证了「流程跑通」而没验证「指标上升」。

---

## M0 — 资产与不变量锁定（无需 Isaac Sim）· 已完成

**目的**：把 ELF3 的物理事实固化成测试，先于任何环境代码。

改动：
- `isaaclab_ext/pyproject.toml` — 包定义，**显式 pin** `isaaclab==2.3.2`、`rsl-rl-lib==3.1.2`
- `isaaclab_ext/source/openhomie_isaaclab/openhomie_isaaclab/assets/elf3/`
  — 从 `HomieRL/legged_gym/resources/robots/elf3_description/urdf/` 复制
  `elf3.urdf` + `meshes/`（保持 `./meshes/*.STL` 相对路径）
- `.../elf3_constants.py` — 唯一事实来源：28 关节名表、腿/上身索引、
  默认关节角、PD 增益、armature、link 名、镜像 spec。
  **所有维度用函数从关节表推导**，不写 78/81/468/113 字面量。

Fable 5 先写的测试（`tests/test_elf3_constants.py`）：
1. URDF 里 revolute 关节数 == 28，且 `waist_x_joint` 为 `fixed`
2. 关节名表与 URDF 逐项一致（顺序敏感）
3. 腿 12 个、上身 16 个，无交集且并集为全集
4. 维度推导：`one_step_actor_obs() == 78`、`critic == 81`、`actor_obs() == 468`
5. 镜像 spec 是置换 + 对合 + 符号对称（对 DOF 和 action 两组都验）
6. 正向运动学：默认姿态下 torso 距脚底 1.013 m（±1 mm）
7. `init_state.pos.z` 与第 6 条推出的值一致 —— **这条直接防住上一版的致命 bug**
8. 所有 mesh 文件存在
9. 关节 limit / effort / velocity 与 URDF 一致，且全为有限正数

验收：
```bash
cd /home/user/wang-sm/OpenHomie && python -m pytest tests/test_elf3_constants.py -v
```

实测：43 passed。过程中修正了两处我自己写错的地方 ——
projected gravity 是**真向量** `(1,-1,1)`（不是伪向量），
以及高度上限从 1.01 提到 1.02 才容纳 1.013 的默认站高。

---

## M1 — URDF → USD 离线转换 · 已完成

**目的**：消除每次启动重转 USD 的 2.1 GB 浪费。

新增 `isaaclab_ext/scripts/convert_elf3_usd.py`：把 URDF 转成
`assets/elf3/elf3.usd`，落盘、可复用。运行时用 `UsdFileCfg`。
脚本幂等，并在 URDF 或任一 mesh 更新时通过 sha256 sidecar 检测过期。

验收：
```bash
python isaaclab_ext/scripts/convert_elf3_usd.py          # 首次转换
python isaaclab_ext/scripts/convert_elf3_usd.py          # 报告 up to date
python isaaclab_ext/scripts/convert_elf3_usd.py --check  # CI 用，过期则非零退出
```

实测：第二次运行输出 `up to date, nothing to do`，且不启动 Isaac Sim。
产物 24 MB（`configuration/elf3_base.usd` 等），已加入 `.gitignore`
（URDF 是事实来源，USD 是派生物）。`/tmp/IsaacLab` 不再增长。

转换时会对 4 个无 geometry 的传感器 link 报
`Unresolved reference prim path .../visuals` 警告 —— 无害，可忽略。

---

## M2 — 资产自检：机器人能站住 · 已完成

**目的**：确认机器人在 Isaac Lab 里站在正确高度上。这是上一版真正失败的地方。

已新增：
- `.../tasks/locomotion/elf3/elf3_articulation.py` — `build_elf3_articulation_cfg()`
  两个 actuator group（腿 `IdealPDActuatorCfg` stiffness=damping=0，
  上身 `ImplicitActuatorCfg` 带真实增益），显式 armature 与 effort limit，
  `init_state.pos=(0, 0, DEFAULT_BASE_HEIGHT)`，外加 `massless_link_event()`
- `isaaclab_ext/scripts/check_elf3_asset.py` — 6 项检查

验收：
```bash
python isaaclab_ext/scripts/check_elf3_asset.py --headless --settle_steps 100
```

实测全过（关键几行）：

```
joint names match the canonical table (28 DOFs)
actuator groups: 12 legs (no Isaac drive) + 16 upper body (position drive)
total mass 43.222 kg (URDF says 43.22 kg)
legs: effort limits in [20.0, 150.0] N m
torso above soles (measured): 0.9956 m
torso above soles (FK)      : 1.0130 m
all ELF3 asset checks passed
```

这一步抓到了 3 个真实缺陷（都记进 CLAUDE.md「Isaac Lab 的坑」）：
运行时关节顺序被广度优先重排（`l_shoulder_y_joint` 是 runtime[0]）、
URDF effort limit 不进 USD（报 1e9）、无 inertial 的 link 被默认给 1 kg。

**注意**：`timeout` 可能返回 124（Isaac Sim 关闭很慢），
判断成败要看日志里的 `check_elf3_asset exit code:` 那一行。

剩余待做（M3 一并完成）：`Elf3HomieEnvCfg` / `Elf3HomieEnv` /
task 注册 / 零 action 静置 drop test。

---

## M3 — 完整 obs / reward / 课程 / 域随机化 · 进行中

已完成（纯张量部分，108 个静态测试通过）：

- `elf3_homie_rewards.py` — 33 项 reward，全部纯张量、零 Isaac 依赖。
  `scale_reward_terms` 严格校验名称集合，缺项/多项直接报错，
  所以新增一项 reward 必须是两处**有意**修改，不会静默失效。
- `elf3_homie_curriculum.py` — 4 模式采样（walk 0.60 / high_stand 0.15 /
  crouch_low 0.15 / crouch_full 0.10）、命令构建、上身幅度指数课程、
  `advance_action_curriculum`。所有随机性作为 `draws` 张量注入，可完全确定性测试。
- `elf3_stages.py` — **阶段表已重排为从站高往下**：
  `S0..S5` 高度 1.01→0.80（窄速度带），`V1..V3` 保持 1.01 并放宽速度。
  旧的 `H0=0.34` 顺序是反的，已彻底废弃（`get_stage("H0")` 现在会报错）。

测试抓到的真实 bug：`_exponential_amplitude` 里 `torch.exp(-rate)` 对
Python float 调用会 `TypeError`。在写实现时没发现，是测试逮到的。
改用 `math.exp`。

剩余（需要 Isaac Lab）：
- `elf3_homie_env_cfg.py` / `elf3_homie_env.py` — DirectRLEnv，obs 组装、
  action scatter、mixed control、终止判据、虚拟脚底四角点、域随机化事件
- `_randomize_control` **必须用赋值语义**（`self._x[env_ids] = ...`），
  并写测试断言随机化后张量真的变了
- task 注册 + 零 action 静置 drop test

验收：静态 reward/课程测试全绿（已达成）+ 随机 action 1000 步无 NaN（待做）。

---

## M4 — HIM PPO on rsl-rl 3.1.2

新增 `him_rl/`：`estimator.py`（log-space Sinkhorn）、`actor_critic.py`、
`ppo.py`（双优化器 + 对称损失）、`storage.py`（branch-aware GAE +
`estimator_masks`）、`runner.py`、`symmetry.py`（spec 驱动）、`exporter.py`。

必须做 upstream 没做的：**estimator 监督不跨 episode 边界**
（env 通过 `extras["terminal_critic_obs"]` + mask 暴露）。

验收：单元测试覆盖 estimator 损失、镜像不变性、storage 形状、
export 数值一致性（TorchScript 1e-7 / ONNX 1e-5）。

---

## M5 — 训练收敛（真正的验收）

`isaaclab_ext/scripts/train_elf3.py` + `play_elf3.py`。
单一命令即可训练，不做 upstream 那种硬编码 resume。
**第一阶段不移植 HOMIE 的 5000 行 `training/` 编排层。**

验收：
```bash
python isaaclab_ext/scripts/train_elf3.py --num-envs 4096 --max-iterations 2000 --headless
```
判据：`mean_episode_length` 从 ~40 上升到 **> 300**。
上升即通过；恒定即环境仍有问题，回 M2 查。

然后 play 目视确认能站、能走、能蹲。

---

## M6 — sim2sim (MuJoCo)

- `pip install mujoco`（homie 环境目前没装；已用 `unitree_rl_mjlab` 环境的
  mujoco 3.5.0 做过模型验证）
- 官方 BXI MJCF 已下载到 `MujocoDeploy/elf3_description/`：
  `elf3.xml`（原样保留）+ 32 个 mesh + `sensors/` + `terrains/`
- `MujocoDeploy/make_elf3_flat.py` 生成 `elf3_flat.xml`：剥掉 7 个
  `<include>`（深度相机、激光雷达、大型障碍赛道），换成平地 + 光源。
  训练在平地上，sim2sim 必须也在平地，否则比的是地形不是策略。
- 新增 `MujocoDeploy/mujoco_deploy_elf3.py` + `elf3.yaml`，**不改 g1 的两个文件**

已验证：`elf3_flat.xml` 加载正常，`nq=38 nv=37 nu=31`，
**总质量 43.222 kg —— 与 URDF 完全一致**，独立印证了 Isaac Lab 侧
幽灵质量修复的正确性（Isaac 侧未修复时是 47.222 kg）。

关键：MJCF 有 31 个执行器，训练只有 28 个。必须
**把 `waist_x_joint` / `head_z_joint` / `head_y_joint` 锁在 0**，
其余 28 个按名字映射（不能按索引，MJCF 顺序与 HOMIE 顺序不同）。
映射已算好：HOMIE idx 0→qpos[7]、1→qpos[9]、2..27→qpos[10..35]。

不要照抄 upstream 的 scale bug：`ang_vel_scale` 必须是 0.5（不是 0.25），
`cmd_scale` 必须是 `[2.0, 2.0, 0.5]`。写测试断言 yaml 与训练 cfg 逐项相等。

### 已知：默认姿态在 MuJoCo 里开环站不住

用真实 PD 增益和 effort limit 在 `elf3_flat.xml` 里静置，
机器人站约 250 步（1.25 s）后前倾倒下（终态 R22 ≈ −0.07，完全倒扣）。
把踝关节 kd 从 2.0 提到 8.0 也只是把终态从倒扣改成瘫倒，仍然站不住。

原因：默认姿态下 **CoM 距脚底中心只有 3.5 mm**，是临界平衡；
脚掌是 7 根 capsule（半径 1 cm）而不是实心底板，支撑多边形很窄。
这是欠驱动双足的正常现象 —— **开环 PD 站不住不代表模型错了**，
站稳本身就是 RL 策略要解决的问题。Isaac Lab 侧能站住 200 步是因为
PhysX 的接触求解更"黏"，不是因为姿态更稳。

**不要为了让静置测试通过去改默认姿态或增益。** M6 的正确判据是
「训练好的策略在 MuJoCo 里能站住并跟随命令」，不是「零策略能站住」。
如果 M5 训出的策略在 Isaac Lab 里能走而在 MuJoCo 里站不住，
那时才需要排查两个仿真器的接触参数差异（friction 0.6 vs 1.0、solref、
condim），以及 obs 是否逐通道一致。

验收：加载训练好的策略后，MuJoCo viewer 里能站住并跟随命令；
与 Isaac Lab play 行为定性一致。

---

## 风险

| 风险 | 应对 |
|---|---|
| spawn 高度再错 | M2 的 drop test 是硬门禁，不过不许往下 |
| 课程方向仍不利 | 从站高往下，且 M5 只看 episode length 是否上升 |
| rsl-rl 3.1.2 API 变动 | pyproject 里 pin 死版本 |
| MJCF 与 URDF 物理不一致 | M6 先做静置对比：两边同姿态的接触力与高度 |
| GPU 被其他任务占用 | 启动前查 `nvidia-smi`，不 kill 他人进程 |

## 执行纪律

每个里程碑：Fable 5 写测试 → Sonnet 5 实现并自测贴输出 →
Fable 5 独立跑验收命令确认。测试红了改实现，不改测试。
