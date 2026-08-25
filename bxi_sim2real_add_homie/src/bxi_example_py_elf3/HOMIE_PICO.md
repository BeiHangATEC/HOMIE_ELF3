# HOMIE 下蹲 + PICO 上身遥操作

该组合保留 `com.bxi.homie/homie` 作为腰腿和高度控制源，只在最终电机命令层按关节名覆盖双臂。移动手臂使用官方 ELF3 上身遥操作增益，而不是 HOMIE 用来固定手臂的高刚度增益；接管默认用 1 秒渐变，并使用官方双臂重力模型产生限幅前馈。PICO 或状态机信息断流时，覆盖层先回到 HOMIE 的手臂 PD 目标，再显式释放。zero-torque 状态始终拒绝覆盖。

PICO 接入和 Pinocchio IK 直接运行官方 [`com.bxi.upper_body_teleop`](https://github.com/konodoki/com.bxi.upper_body_teleop) 代码，其 XRoboToolkit 资产来自官方 [`com.bxi.sonic`](https://github.com/konodoki/com.bxi.sonic)。不要把这两个仓库放入 `/opt/bxi/mods`，否则官方半身遥操状态会被同时加载并与本组合重复启动 PICO 端口。

## 运行时目录

两个仓库必须是相邻目录：

```text
/opt/bxi/homie_pico_runtime/
├── com.bxi.sonic/
└── com.bxi.upper_body_teleop/
```

也可以放在普通用户目录，并通过 `upper_body_mod_root:=...` 或环境变量 `BXI_UPPER_BODY_MOD_ROOT` 指向 `com.bxi.upper_body_teleop`。当前机器使用：

```text
/home/bxi/bxi_wsm/homie_pico_runtime/com.bxi.upper_body_teleop
/home/bxi/bxi_wsm/homie_pico_runtime/com.bxi.sonic
```

官方 PICO manager 和 IK launcher 会共同使用具备 Pinocchio、`rclpy`、`sensor_msgs`、`pyzmq` 和本 ROS 包的 Python。可通过 `arm_ik_python:=/absolute/path/to/python` 或 `BXI_ARM_IK_PYTHON` 显式指定。不要使用名称相同但不含机器人学 API 的 PyPI `pinocchio` 包。本机已经验证可用的解释器是 `/home/bxi/bxi_lj/bxi_pnlink_wholebody_teleop/.venv_teleop/bin/python`。

## 构建

```bash
cd /home/bxi/bxi_wsm/bxi_sim2real_add_homie
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
colcon build --merge-install --symlink-install --packages-select bxi_example_py_elf3
source install/setup.bash
```

## Sim2Sim

```bash
ros2 launch bxi_example_py_elf3 example_homie_pico_sim.launch.py \
  upper_body_mod_root:=/home/bxi/bxi_wsm/homie_pico_runtime/com.bxi.upper_body_teleop \
  arm_ik_python:=/home/bxi/bxi_lj/bxi_pnlink_wholebody_teleop/.venv_teleop/bin/python
```

如果先验证命令合成而不启动 PICO，可使用：

```bash
ros2 launch bxi_example_py_elf3 example_homie_pico_sim.launch.py \
  upper_body_mod_root:=/home/bxi/bxi_wsm/homie_pico_runtime/com.bxi.upper_body_teleop \
  start_pico_runtime:=false
```

本机图形会话应使用 `DISPLAY=:0 XAUTHORITY=/home/bxi/.Xauthority`；SSH 转发得到的 `DISPLAY=localhost:10.0` 不能启动 MuJoCo。先用手柄组合键产生 `btn_10=13` 进入 `com.bxi.homie/homie`，然后在 PICO 按 `A+B+X+Y` 校准；握左/右 grip 后对应手臂渐进接管。头部仅在机器人反馈实际包含 `head_y_joint` 和 `head_z_joint` 时启用。

验收话题：

```bash
ros2 topic hz /pico_control_joint_commands
ros2 topic hz /simulation/actuators_cmds_override
ros2 topic echo /simulation/state_machine_info std_msgs/msg/String --field data --once
```

## Sim2Real

真机首次测试必须使用吊架或等效保护，清空双臂运动范围并保持急停可达。没有现场明确允许时，不要启动下列硬件 launch。`robot_config.yaml` 不是必需项：文件不存在时硬件 launch 使用工程内置默认值；当前机器应先保持头部和夹爪关闭。

硬件 launch 已经运行时，先在第二个终端执行只读预检。该命令不发布控制指令，也不改变电机状态：

```bash
cd /home/bxi/bxi_wsm/bxi_sim2real_add_homie
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source install/setup.bash
sudo -H env \
  BXI_UPPER_BODY_MOD_ROOT=/home/bxi/bxi_wsm/homie_pico_runtime/com.bxi.upper_body_teleop \
  BXI_ARM_IK_PYTHON=/home/bxi/bxi_lj/bxi_pnlink_wholebody_teleop/.venv_teleop/bin/python \
  ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  /home/bxi/bxi_wsm/bxi_sim2real_add_homie/install/lib/bxi_example_py_elf3/homie_pico_hw_preflight \
  --duration 5 --motor-recv-mask 0x603FFFFF
```

把 `--motor-recv-mask` 替换为本次硬件启动日志中最后一条 `motor_timeout` 后的十六进制数。虽然日志标签叫 `motor_timeout`，底层实际打印的是 `get_motor_recv_flag()` 的已收到反馈位图。`0x603FFFFF` 表示腰、双腿和左臂均已收到反馈，仅右臂 `can4` 的 7 个电机缺失；`0x60000000` 是禁用的两个头部位。腿部反馈存在时可以继续定位“无法站立”的控制链，但双臂 PICO 动作测试要求预检 JSON 的 `static.ready`、`ros.ready` 和 `motor_receive.ready` 全部为 `true`。

```bash
sudo -H env \
  BXI_UPPER_BODY_MOD_ROOT=/home/bxi/bxi_wsm/homie_pico_runtime/com.bxi.upper_body_teleop \
  BXI_ARM_IK_PYTHON=/home/bxi/bxi_lj/bxi_pnlink_wholebody_teleop/.venv_teleop/bin/python \
  ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  bash -c '
    set -e
    cd /home/bxi/bxi_wsm/bxi_sim2real_add_homie
    source /opt/ros/humble/setup.bash
    source /opt/bxi/bxi_ros2_pkg/setup.bash
    source install/setup.bash
    exec ros2 launch bxi_example_py_elf3 example_homie_pico_hw.launch.py \
      enable_head:=false start_video_runtime:=false hardware_gripper:=false
  '
```

真机覆盖话题为 `/hardware/actuators_cmds_override`。控制器在消息超过 0.2 秒未刷新时自动释放，在 zero-torque 状态下无条件忽略覆盖。重力补偿要求 `/hardware/actuator_states` 和 `/hardware/imu_data` 都持续新鲜，否则不会接管手臂。

首次只验收站立、进入 HOMIE、单臂小幅接管、松 grip 回退、断开 PICO 回退、退出 HOMIE 和急停。即使 `0x603FFFFF` 只定位到右臂 `can4`，若预检看不到持续的 `/hardware/actuator_states`、`/hardware/imu_data`、`/hardware/state_machine_info` 或 `/hardware/actuators_cmds`，也不要继续 PICO 真机动作测试。

夹爪默认关闭。确认左右夹爪确实位于 CANFD bus 5/6，并准备好限位校准后，才可在吊架测试中显式增加 `hardware_gripper:=true`；进入 HOMIE 后它会自动驱动夹爪寻找机械限位。头部硬件确认后再改为 `enable_head:=true`。
