import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    method_name = LaunchConfiguration("method_name")
    output_dir = LaunchConfiguration("output_dir")
    ground_truth_topic = LaunchConfiguration("ground_truth_topic")
    fusion_pose_topic = LaunchConfiguration("fusion_pose_topic")
    debug_topic = LaunchConfiguration("debug_topic")
    alignment_mode = LaunchConfiguration("alignment_mode")
    eval_start_delay_sec = LaunchConfiguration("eval_start_delay_sec")
    eval_duration_sec = LaunchConfiguration("eval_duration_sec")

    eval_node = Node(
        package="adaptive_fusion_eval",
        executable="eval_node",
        name="adaptive_fusion_eval_node",
        output="screen",
        parameters=[
            {
                "method_name": method_name,
                "output_dir": output_dir,
                "ground_truth_topic": ground_truth_topic,
                "fusion_pose_topic": fusion_pose_topic,
                "debug_topic": debug_topic,
                "alignment_mode": alignment_mode,
                "eval_start_delay_sec": eval_start_delay_sec,
                "eval_duration_sec": eval_duration_sec,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("method_name", default_value="dual_adaptive"),
            DeclareLaunchArgument(
                "output_dir",
                default_value=os.path.join("/tmp", "adaptive_fusion_eval"),
            ),
            DeclareLaunchArgument(
                "ground_truth_topic",
                default_value="/ground_truth/odom",
            ),
            DeclareLaunchArgument(
                "fusion_pose_topic",
                default_value="/fusion/pose",
            ),
            DeclareLaunchArgument(
                "debug_topic",
                default_value="/fusion/debug",
            ),
            DeclareLaunchArgument(
                "alignment_mode",
                default_value="initial_pose",
            ),
            DeclareLaunchArgument(
                "eval_start_delay_sec",
                default_value="0.0",
            ),
            DeclareLaunchArgument(
                "eval_duration_sec",
                default_value="0.0",
            ),
            eval_node,
        ]
    )
