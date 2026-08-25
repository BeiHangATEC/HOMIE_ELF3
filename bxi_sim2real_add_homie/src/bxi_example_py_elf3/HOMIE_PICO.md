# HOMIE 下蹲 + PICO 上身遥操作

该组合保留 `com.bxi.homie/homie` 作为腰腿和高度控制源，只在最终电机命令层按关节名覆盖双臂。PICO 或状态机信息断流时，覆盖层先回到 HOMIE 的手臂 PD 目标，再显式释放；zero-torque 状态始终拒绝覆盖。

PICO 接入和 Pinocchio IK 直接运行官方 [`com.bxi.upper_body_teleop`](https://github.com/konodoki/com.bxi.upper_body_teleop) 代码，其 XRoboToolkit 资产来自官方 [`com.bxi.sonic`](https://github.com/konodoki/com.bxi.sonic)。不要把这两个仓库放入 `/opt/bxi/mods`，否则官方半身遥操状态会被同时加载并与本组合重复启动 PICO 端口。

## 运行时目录

两个仓库必须是相邻目录：

```text
/opt/bxi/homie_pico_runtime/
├── com.bxi.sonic/
└── com.bxi.upper_body_teleop/
```

也可以放在普通用户目录，并通过 `upper_body_mod_root:=...` 或环境变量 `BXI_UPPER_BODY_MOD_ROOT` 指向 `com.bxi.upper_body_teleop`。

官方 IK launcher 会寻找同时具备 Pinocchio、`rclpy`、`sensor_msgs`、`pyzmq` 和本 ROS 包的 Python。可通过 `arm_ik_python:=/absolute/path/to/python` 或 `BXI_ARM_IK_PYTHON` 显式指定。不要使用名称相同但不含机器人学 API 的 PyPI `pinocchio` 包。

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
  upper_body_mod_root:=/opt/bxi/homie_pico_runtime/com.bxi.upper_body_teleop \
  arm_ik_python:=/absolute/path/to/pinocchio_ros_python
```

如果先验证命令合成而不启动 PICO，可使用：

```bash
ros2 launch bxi_example_py_elf3 example_homie_pico_sim.launch.py \
  start_pico_runtime:=false
```

进入 `com.bxi.homie/homie` 后，PICO 按 `A+B+X+Y` 校准；握左/右 grip 后对应手臂渐进接管。头部仅在当前机器人反馈实际包含 `head_y_joint` 和 `head_z_joint` 时启用。

验收话题：

```bash
ros2 topic hz /pico_control_joint_commands
ros2 topic hz /simulation/actuators_cmds_override
ros2 topic echo /simulation/state_machine_info std_msgs/msg/String --field data --once
```

## Sim2Real

真机首次测试必须使用吊架或等效保护，清空双臂运动范围并保持急停可达。没有现场明确允许时，不要启动下列硬件 launch。

```bash
sudo -H env \
  BXI_UPPER_BODY_MOD_ROOT=/opt/bxi/homie_pico_runtime/com.bxi.upper_body_teleop \
  BXI_ARM_IK_PYTHON=/absolute/path/to/pinocchio_ros_python \
  ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  bash -c '
    set -e
    cd /home/bxi/bxi_wsm/bxi_sim2real_add_homie
    source /opt/ros/humble/setup.bash
    source /opt/bxi/bxi_ros2_pkg/setup.bash
    source install/setup.bash
    exec ros2 launch bxi_example_py_elf3 example_homie_pico_hw.launch.py \
      robot_config_file:=/opt/bxi/robot_config.yaml enable_head:=auto
  '
```

真机覆盖话题为 `/hardware/actuators_cmds_override`。控制器在消息超过 0.2 秒未刷新时自动释放，在 zero-torque 状态下无条件忽略覆盖。手臂方向、软限位、最大速度、断流回退和急停都必须在现场逐项验收后，才能扩大动作范围。
