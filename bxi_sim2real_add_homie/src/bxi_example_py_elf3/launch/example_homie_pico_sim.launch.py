"""HOMIE crouch locomotion plus official PICO upper-body teleoperation in MuJoCo."""

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from bxi_example_py_elf3.homie_pico.launch_support import (
    declare_homie_pico_arguments,
    homie_pico_actions,
)


def generate_launch_description():
    base_launch = (
        get_package_share_path("bxi_example_py_elf3")
        / "launch"
        / "example_launch_demo.py"
    )
    return LaunchDescription(
        declare_homie_pico_arguments(start_video_default=False)
        + [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(base_launch))
            )
        ]
        + homie_pico_actions(topic_prefix="simulation/")
    )
