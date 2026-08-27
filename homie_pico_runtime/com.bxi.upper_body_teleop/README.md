# PICO 半身遥操

该 Mod 是旧 `TeleopState` 的新框架迁移版，不修改 SONIC 或其他现有状态：

- 下半身由 `com.bxi.basic_actions/without_arm_policy` 持续保持站立/行走；
- PICO 三点姿态沿用 SONIC 的 XRoboToolkit 读取、校准和 POSE 数据，新 Mod 仅将交互收敛为校准后自动进入 POSE；
- 双臂使用 Pinocchio 对 `waist_z_link -> wrist_z_link` 的具名目标做阻尼最小二乘 IK，并使用实时关节状态作为每帧 seed；
- IK 通过 `joint_limit_margin_rad` 在 URDF 硬限位内设置可配置余量；设为 `0` 时软限位与硬限位重合。零空间回中主动增加限位余量，并以每帧实测肩-肘-腕 Swivel 为连续性参考；解算后通过位置、姿态和限位检查，合法参考再由发布端逐关节限速；
- 未握 grip、POSE 未就绪或数据超过 `live_reference_timeout_s` 时，对应手臂保持 `without_arm_policy` 的具名 PD 站立姿态，并叠加重力前馈；
- 夹爪采用与 SONIC 相同的 CAN FD 使能、反馈校验、限位校准和 trigger 映射；
- 头部相机 RTSP 参数与 SONIC 一致，默认地址为 `rtsp://<机器人IP>:2212/video`。

## 操作

1. 在 `normal` 状态按 `LB + LT + X`（`btn_10=17`）进入半身遥操。
2. PICO 同时按下 `A+B+X+Y` 请求校准；保持标准站姿，校准成功后自动开始输出实时 POSE，无需再按 `A+X`。
3. 握紧左/右 grip 才允许对应手臂跟随；松开后按侧平滑退回 PD 站立姿态。
4. trigger 控制对应夹爪。夹爪在每次进入状态后先完成与 SONIC 一致的低速限位校准。

`A+X` 在该 Mod 中不再切换模式。需要重新校准时再次按下 `A+B+X+Y`；重新校准期间 POSE 暂停，双臂自动回到 PD 站立姿态。

PICO 未连接时状态不会自动退出，机器人下半身继续受步态策略控制，双臂保持 PD 站立姿态。退出仍使用 normal、PD brake、recover 或 zero-torque 的标准事件。

位置或姿态残差超过质量阈值时，bridge 会继续采用 Pinocchio 得到的有限、软限位内 best-effort 候选，并经过逐关节限速持续发布；残差只用于首次告警和每 5 秒质量汇总，不再让手臂在可达边界处一卡一卡。只有数值失败、缺少完整机器人反馈或候选本身越过软限位时，才保持最后一帧安全参考。PICO POSE 真正停止或退出校准模式并超过 `live_reference_timeout_s` 时，状态才平滑退回 `withoutarm` 的双臂 PD 姿态。

关节差值和 Swivel 差值仅表示目标离当前姿态有多远，不再被当作 IK 合法性条件。合法 IK 参考由 `maximum_joint_step_rad=0.12` 限制相邻发布帧的逐关节变化，状态层再通过 `arm_gain_ramp_s=0.4` 在握下 grip 时从 PD 姿态平滑接管，避免初始大姿态差形成“拒绝后机器人不动、下一帧继续拒绝”的死锁。

Swivel 连续性只作为手腕 SE(3) 主任务的零空间次目标。当手臂接近伸直、肘部到肩腕轴的距离小于 `swivel_min_radius_m=0.02` 时，Swivel 几何退化，该侧会自动跳过次目标，仍由原有残差、软限位与发布限速保护。

## 运行依赖

该 Mod 显式依赖 `com.bxi.sonic >=1.3`，只复用其已部署的 PICO、MediaMTX 和相机推流运行资产，不加载或调用 SONIC 状态代码。Pinocchio IK bridge 使用独立启动器选择同时具备机器人学 Pinocchio API、ZeroMQ、ROS 消息和本包的 Python；PICO 管理器也会在加载 SONIC 原生校准前激活当前解释器的 `cmeel.prefix` Pinocchio。两条进程都会拒绝误装的同名 PyPI `pinocchio` 测试插件，IK bridge 还支持 `BXI_ARM_IK_PYTHON` 显式指定解释器。CAN FD 消息接口仍由清单检查。

PICO 管理器在启动 RoboticsService 和绑定 5556 前先加载校准模型；进入设备初始化后若任一步失败，会显式关闭 reader、ZMQ publisher/context 和 XRT session，避免重启时把自身残留误报成其他 SONIC/PICO 管理器占用端口。

首次真机验证必须使用吊架、清空双臂运动范围、保持急停可达，并从保守增益开始。离线装载、IK 和状态命令测试不能替代真机方向、限位、力矩和断流验证。
