"""Shared launch actions for HOMIE + official PICO upper-body runtime."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


DEFAULT_RUNTIME_ROOT = "/opt/bxi/homie_pico_runtime/com.bxi.upper_body_teleop"


def _as_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"launch argument {name} must be true or false, got {value!r}")


def declare_homie_pico_arguments(*, start_video_default: bool) -> list:
    return [
        DeclareLaunchArgument(
            "upper_body_mod_root",
            default_value=EnvironmentVariable(
                "BXI_UPPER_BODY_MOD_ROOT",
                default_value=DEFAULT_RUNTIME_ROOT,
            ),
            description=(
                "Official com.bxi.upper_body_teleop checkout. Its sibling "
                "com.bxi.sonic checkout supplies the XRoboToolkit runtime."
            ),
        ),
        DeclareLaunchArgument("start_pico_runtime", default_value="true"),
        DeclareLaunchArgument(
            "start_video_runtime",
            default_value="true" if start_video_default else "false",
        ),
        DeclareLaunchArgument("pico_host", default_value="127.0.0.1"),
        DeclareLaunchArgument("pico_port", default_value="5556"),
        DeclareLaunchArgument("pico_topic", default_value="pose"),
        DeclareLaunchArgument(
            "arm_ik_python",
            default_value=EnvironmentVariable(
                "BXI_ARM_IK_PYTHON",
                default_value="",
            ),
            description=(
                "Python executable containing Pinocchio, ROS 2, ZeroMQ and "
                "bxi_example_py_elf3. Empty lets the official launcher probe."
            ),
        ),
        DeclareLaunchArgument("reference_topic", default_value="pico_control_joint_commands"),
        DeclareLaunchArgument("robot_state_topic", default_value="pico_control_joint_states"),
        DeclareLaunchArgument("head_control_enabled", default_value="true"),
        DeclareLaunchArgument("arm_gain_ramp_s", default_value="0.4"),
        DeclareLaunchArgument("grip_threshold", default_value="0.5"),
        DeclareLaunchArgument("reference_timeout_s", default_value="0.5"),
    ]


def _runtime_processes(context):
    if not _as_bool(
        LaunchConfiguration("start_pico_runtime").perform(context),
        name="start_pico_runtime",
    ):
        return []

    upper_root = Path(
        os.path.abspath(
            os.path.expanduser(LaunchConfiguration("upper_body_mod_root").perform(context))
        )
    )
    sonic_root = upper_root.parent / "com.bxi.sonic"
    shared_launcher = upper_root / "shared_runtime_launcher.py"
    arm_launcher = upper_root / "arm_bridge_launcher.py"
    required = (
        upper_root / "mod.yaml",
        shared_launcher,
        arm_launcher,
        sonic_root / "mod.yaml",
    )
    missing = tuple(str(path) for path in required if not path.is_file())
    if missing:
        raise RuntimeError(
            "HOMIE PICO runtime is incomplete; missing: " + ", ".join(missing)
        )

    python = sys.executable
    actions = [
        ExecuteProcess(
            cmd=[
                python,
                str(shared_launcher),
                "pico-manager",
                "--num_frames_to_send",
                "10",
                "--target_fps",
                "50",
            ],
            name="homie_pico_manager",
            output="screen",
            additional_env={"PYTHONUNBUFFERED": "1"},
        ),
        ExecuteProcess(
            cmd=[
                python,
                str(arm_launcher),
                "--pico-host",
                LaunchConfiguration("pico_host"),
                "--pico-port",
                LaunchConfiguration("pico_port"),
                "--pico-topic",
                LaunchConfiguration("pico_topic"),
                "--robot-state-topic",
                LaunchConfiguration("robot_state_topic"),
                "--reference-topic",
                LaunchConfiguration("reference_topic"),
                "--rate-hz",
                "50.0",
                "--stale-timeout-s",
                LaunchConfiguration("reference_timeout_s"),
                "--required-consecutive-frames",
                "3",
                "--ik-iterations",
                "24",
                "--ik-damping",
                "0.001",
                "--ik-step-size",
                "0.7",
                "--ik-tolerance",
                "0.0001",
                "--maximum-position-error-m",
                "0.003",
                "--maximum-orientation-error-rad",
                "0.03",
                "--maximum-joint-step-rad",
                "0.12",
                "--joint-limit-margin-rad",
                "0.0",
                "--joint-centering-gain",
                "0.005",
                "--swivel-continuity-gain",
                "0.02",
                "--swivel-min-radius-m",
                "0.02",
            ],
            name="homie_pico_arm_ik",
            output="screen",
            additional_env={
                "PYTHONUNBUFFERED": "1",
                "BXI_ARM_IK_PYTHON": LaunchConfiguration("arm_ik_python"),
            },
        ),
    ]

    if _as_bool(
        LaunchConfiguration("start_video_runtime").perform(context),
        name="start_video_runtime",
    ):
        camera_binary = sonic_root / "bin" / (
            "linux-aarch64" if os.uname().machine in {"aarch64", "arm64"} else "linux-x86_64"
        ) / "head_camera_rtsp_node"
        if not camera_binary.is_file():
            raise RuntimeError(f"HOMIE PICO camera runtime is missing: {camera_binary}")
        actions.extend(
            (
                ExecuteProcess(
                    cmd=[python, str(shared_launcher), "mediamtx"],
                    name="homie_pico_mediamtx",
                    output="screen",
                    additional_env={"PYTHONUNBUFFERED": "1"},
                ),
                ExecuteProcess(
                    cmd=[
                        python,
                        str(shared_launcher),
                        "camera",
                        "--ros-args",
                        "-p",
                        "simulation_topic:=/simulation/head_depth_camera/color/image_raw",
                        "-p",
                        "hardware_topic:=/hardware/head_depth_camera/color/image_raw",
                        "-p",
                        "source_mode:=auto",
                        "-p",
                        "source_timeout_s:=0.5",
                        "-p",
                        "rtsp_url:=rtsp://127.0.0.1:2212/video",
                    ],
                    name="homie_pico_head_camera_rtsp",
                    output="screen",
                    additional_env={"PYTHONUNBUFFERED": "1"},
                ),
            )
        )
    return actions


def homie_pico_actions(*, topic_prefix: str) -> list:
    state_topic = f"{topic_prefix.strip('/')}/state_machine_info"
    return [
        Node(
            package="bxi_example_py_elf3",
            executable="homie_pico_arm_override",
            name="homie_pico_arm_override",
            output="screen",
            parameters=[
                {"topic_prefix": topic_prefix},
                {"state_machine_info_topic": state_topic},
                {"target_state": "com.bxi.homie/homie"},
                {"reference_topic": LaunchConfiguration("reference_topic")},
                {"robot_state_topic": LaunchConfiguration("robot_state_topic")},
                {
                    "head_control_enabled": ParameterValue(
                        LaunchConfiguration("head_control_enabled"), value_type=bool
                    )
                },
                {
                    "arm_gain_ramp_s": ParameterValue(
                        LaunchConfiguration("arm_gain_ramp_s"), value_type=float
                    )
                },
                {
                    "grip_threshold": ParameterValue(
                        LaunchConfiguration("grip_threshold"), value_type=float
                    )
                },
                {
                    "reference_timeout_s": ParameterValue(
                        LaunchConfiguration("reference_timeout_s"), value_type=float
                    )
                },
            ],
            emulate_tty=True,
        ),
        OpaqueFunction(function=_runtime_processes),
    ]


__all__ = [
    "DEFAULT_RUNTIME_ROOT",
    "declare_homie_pico_arguments",
    "homie_pico_actions",
]
