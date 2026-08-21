# ELF3 腰部三自由度固定实施计划

> **面向执行代理：** 必须使用 `executing-plans` 按任务逐项实施，并遵循测试先行流程。

**目标：** 固定 ELF3 三个腰部关节，并保持 Isaac Gym 训练配置、观测维度和左右对称增强与新的 26 自由度模型一致。

**架构：** URDF 是可动关节集合的唯一来源；ELF3 配置显式声明策略关节和上半身关节，并由环境现有校验确认完整覆盖；`HIMPPO` 根据单步观测维度选择 ELF3 专用镜像映射。测试分别从 XML、Python AST 和真实 PyTorch 张量三个层面验证这一契约。

**技术栈：** URDF/XML、Python、`unittest`、PyTorch、Isaac Gym 配置。

---

### 任务一：建立失败的自由度契约测试

**文件：**

- 新建：`tests/test_elf3_fixed_waist.py`
- 检查：`HomieRL/legged_gym/resources/robots/elf3_description/urdf/elf3.urdf`
- 检查：`HomieRL/legged_gym/legged_gym/envs/g1/elf3_config.py`

- [ ] **步骤一：编写 URDF 与配置测试**

测试使用 `xml.etree.ElementTree` 解析 URDF，使用 `ast` 提取 `Elf3RoughCfg.asset`、`init_state`、`control` 和 `env` 的赋值。断言三个腰部关节均为 `fixed`，模型含 26 个 `revolute` 和 9 个 `fixed`，策略与上半身名单分别为 12 和 14 个且完整覆盖可动关节，并断言观测维度为 74、77、444。

```python
import ast
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = REPO_ROOT / "HomieRL/legged_gym/resources/robots/elf3_description/urdf/elf3.urdf"
CONFIG_PATH = REPO_ROOT / "HomieRL/legged_gym/legged_gym/envs/g1/elf3_config.py"
sys.path.insert(0, str(REPO_ROOT / "HomieRL/rsl_rl"))

from rsl_rl.algorithms.him_ppo import HIMPPO


def load_elf3_config_assignments(path):
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    elf3 = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Elf3RoughCfg")
    nested_classes = {node.name: node for node in elf3.body if isinstance(node, ast.ClassDef)}
    result = {}

    for class_name in ("init_state", "control", "asset", "env"):
        values = {}
        for node in nested_classes[class_name].body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            expression = ast.Expression(body=node.value)
            try:
                values[node.targets[0].id] = eval(
                    compile(ast.fix_missing_locations(expression), str(path), "eval"),
                    {"__builtins__": {}},
                    values,
                )
            except (NameError, TypeError):
                continue
        result[class_name] = values

    return result


class Elf3FixedWaistTest(unittest.TestCase):
    def test_fixed_waist_matches_training_layout(self):
        robot = ET.parse(URDF_PATH).getroot()
        joints = robot.findall("joint")
        joint_types = {joint.get("name"): joint.get("type") for joint in joints}

        for name in ("waist_y_joint", "waist_x_joint", "waist_z_joint"):
            self.assertEqual(joint_types[name], "fixed")

        movable_names = {joint.get("name") for joint in joints if joint.get("type") != "fixed"}
        self.assertEqual(sum(joint.get("type") == "revolute" for joint in joints), 26)
        self.assertEqual(sum(joint.get("type") == "fixed" for joint in joints), 9)

        config = load_elf3_config_assignments(CONFIG_PATH)
        self.assertEqual(config["env"]["num_actions"], 12)
        self.assertEqual(config["env"]["num_dofs"], 26)
        self.assertEqual(config["env"]["num_one_step_observations"], 74)
        self.assertEqual(config["env"]["num_one_step_privileged_obs"], 77)
        self.assertEqual(config["env"]["num_observations"], 444)
        self.assertEqual(len(config["asset"]["upper_body_dof_names"]), 14)
        self.assertNotIn("waist_y_joint", config["init_state"]["default_joint_angles"])
        self.assertNotIn("waist_z_joint", config["init_state"]["default_joint_angles"])
        self.assertNotIn("waist", config["control"]["stiffness"])
        self.assertNotIn("waist", config["control"]["damping"])
        self.assertEqual(
            set(config["asset"]["policy_dof_names"] + config["asset"]["upper_body_dof_names"]),
            movable_names,
        )
```

- [ ] **步骤二：运行测试并确认正确失败**

运行：

```bash
conda run -n homie python -m unittest tests.test_elf3_fixed_waist.Elf3FixedWaistTest.test_fixed_waist_matches_training_layout -v
```

预期：失败信息指出 `waist_y_joint` 仍为 `revolute`，证明测试能捕获本次行为变化。

### 任务二：固定 URDF 并同步 ELF3 配置

**文件：**

- 修改：`HomieRL/legged_gym/resources/robots/elf3_description/urdf/elf3.urdf:48`
- 修改：`HomieRL/legged_gym/resources/robots/elf3_description/urdf/elf3.urdf:104`
- 修改：`HomieRL/legged_gym/legged_gym/envs/g1/elf3_config.py:37`
- 测试：`tests/test_elf3_fixed_waist.py`

- [ ] **步骤一：实施最小 URDF 修改**

将两个仍可动的腰部关节改为固定类型，并删除固定关节不再使用的 `axis` 与 `limit`：

```xml
<joint name="waist_y_joint" type="fixed">
  <origin xyz="0 0 -0.2265" rpy="0 0 0"/>
  <parent link="torso_link"/>
  <child link="waist_y_link"/>
</joint>

<joint name="waist_z_joint" type="fixed">
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <parent link="waist_x_link"/>
  <child link="waist_z_link"/>
</joint>
```

- [ ] **步骤二：同步配置**

删除 `default_joint_angles` 中两个腰部项，删除 `stiffness` 和 `damping` 中不再使用的 `waist` 项，从 `upper_body_dof_names` 删除两个腰部关节，并修改环境维度：

```python
num_actions = 12
num_dofs = 26
num_one_step_observations = 2 * num_dofs + 10 + num_actions  # 52 + 10 + 12 = 74
num_one_step_privileged_obs = num_one_step_observations + 3
```

- [ ] **步骤三：运行结构测试并确认通过**

运行任务一中的测试，预期 `OK`。

### 任务三：建立失败的 ELF3 镜像测试

**文件：**

- 修改：`tests/test_elf3_fixed_waist.py`
- 检查：`HomieRL/rsl_rl/rsl_rl/algorithms/him_ppo.py`

- [ ] **步骤一：编写真实张量镜像测试**

使用 `HIMPPO.__new__(HIMPPO)` 绕过优化器初始化，提供包含 74/77 单步维度的最小 `actor_critic`，验证 Actor、Critic 和动作经过两次镜像后精确恢复，并检查 26 个关节的预期左右交换关系。

```python
def test_elf3_symmetry_is_involutive_for_26_dofs(self):
    ppo = HIMPPO.__new__(HIMPPO)
    ppo.actor_critic = SimpleNamespace(
        num_one_step_obs=74,
        actor_history_length=2,
        num_one_step_critic_obs=77,
        critic_history_length=1,
    )
    actor_obs = torch.arange(2 * 2 * 74, dtype=torch.float32).reshape(2, 148)
    critic_obs = torch.arange(2 * 77, dtype=torch.float32).reshape(2, 77)
    actions = torch.arange(24, dtype=torch.float32).reshape(2, 12)

    flipped_actor = ppo.flip_g1_actor_obs(actor_obs)
    dof_indices = torch.tensor([
        6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5,
        19, 20, 21, 22, 23, 24, 25, 12, 13, 14, 15, 16, 17, 18,
    ])
    dof_signs = torch.tensor(
        [1., -1., -1., 1., 1., -1.] * 2
        + [1., -1., -1., 1., -1., 1., -1.] * 2
    )
    first_frame = actor_obs.reshape(2, 2, 74)[0, 0]
    flipped_first_frame = flipped_actor.reshape(2, 2, 74)[0, 0]
    self.assertTrue(torch.equal(flipped_first_frame[10:36], first_frame[10:36][dof_indices] * dof_signs))

    self.assertTrue(torch.equal(ppo.flip_g1_actor_obs(flipped_actor), actor_obs))
    self.assertTrue(torch.equal(ppo.flip_g1_critic_obs(ppo.flip_g1_critic_obs(critic_obs)), critic_obs))
    self.assertTrue(torch.equal(ppo.flip_g1_actions(ppo.flip_g1_actions(actions)), actions))
```

- [ ] **步骤二：运行镜像测试并确认正确失败**

运行：

```bash
PYTHONPATH=HomieRL/rsl_rl conda run -n homie python -m unittest tests.test_elf3_fixed_waist.Elf3FixedWaistTest.test_elf3_symmetry_is_involutive_for_26_dofs -v
```

预期：失败或下标越界，因为 74/77 维尚未分派到 ELF3 路径，且 ELF3 帮助方法仍按 28 个关节切片。

### 任务四：调整 ELF3 对称观测映射

**文件：**

- 修改：`HomieRL/rsl_rl/rsl_rl/algorithms/him_ppo.py:239`
- 修改：`HomieRL/rsl_rl/rsl_rl/algorithms/him_ppo.py:345`
- 修改：`HomieRL/rsl_rl/rsl_rl/algorithms/him_ppo.py:469`
- 测试：`tests/test_elf3_fixed_waist.py`

- [ ] **步骤一：更新分派维度**

Actor 和 Critic 分别以 74 和 77 识别固定腰部后的 ELF3：

```python
if self.actor_critic.num_one_step_obs == 74:
    return self._flip_elf3_observations(
        obs,
        self.actor_critic.actor_history_length,
        self.actor_critic.num_one_step_obs,
        include_base_linear_velocity=False,
    )

if self.actor_critic.num_one_step_critic_obs == 77:
    return self._flip_elf3_observations(
        critic_obs,
        self.actor_critic.critic_history_length,
        self.actor_critic.num_one_step_critic_obs,
        include_base_linear_velocity=True,
    )
```

- [ ] **步骤二：更新 26 自由度映射**

删除腰部项，更新双臂索引和关节宽度：

```python
arm_indices = [19, 20, 21, 22, 23, 24, 25, 12, 13, 14, 15, 16, 17, 18]
arm_signs = [1., -1., -1., 1., -1., 1., -1.] * 2
dof_indices = lower_indices + arm_indices
dof_signs = proprioceptive_obs.new_tensor(lower_signs + arm_signs)
num_dofs = 26
```

- [ ] **步骤三：运行全部测试并确认通过**

运行：

```bash
PYTHONPATH=HomieRL/rsl_rl conda run -n homie python -m unittest tests.test_elf3_fixed_waist -v
```

预期：全部通过，无错误和警告。

### 任务五：更新用户文档并完成验证

**文件：**

- 修改：`README.md:102`
- 修改：`README.md:167`

- [ ] **步骤一：更新 ELF3 公开维度**

将 README 中 ELF3 总自由度改为 26，Actor/Critic 观测改为 444/77，上半身自由度改为 14，并将尚未实现的 Sim2Sim 预期输入从 468 改为 444。

- [ ] **步骤二：运行静态与语法验证**

```bash
PYTHONPATH=HomieRL/rsl_rl conda run -n homie python -m unittest tests.test_elf3_fixed_waist -v
python -m py_compile HomieRL/legged_gym/legged_gym/envs/g1/elf3_config.py HomieRL/rsl_rl/rsl_rl/algorithms/him_ppo.py tests/test_elf3_fixed_waist.py
git diff --check
```

预期：测试全部通过，三个 Python 文件编译成功，`git diff --check` 无输出。

- [ ] **步骤三：审查最终差异**

确认只包含已批准的 URDF、ELF3 配置、对称映射、README 和测试修改，不包含用户已有的 `HomieRL/requirements.txt` 修改。
