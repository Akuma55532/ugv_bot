import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("adaptive_fusion_slam_adapter").find(
        "adaptive_fusion_slam_adapter"
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    resolution = LaunchConfiguration("resolution")
    publish_period_sec = LaunchConfiguration("publish_period_sec")
    use_occupancy_grid = LaunchConfiguration("use_occupancy_grid")
    configuration_directory = LaunchConfiguration("configuration_directory")
    configuration_basename = LaunchConfiguration("configuration_basename")
    map_frame = LaunchConfiguration("map_frame")
    robot_frame = LaunchConfiguration("robot_frame")
    slam_input_mode = LaunchConfiguration("slam_input_mode")
    tracked_pose_topic = LaunchConfiguration("tracked_pose_topic")
    slam_pose_topic = LaunchConfiguration("slam_pose_topic")
    slam_pose_rate = LaunchConfiguration("slam_pose_rate")

    cartographer_node = Node(
        package="cartographer_ros",
        executable="cartographer_node",
        name="cartographer_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=[
            "-configuration_directory",
            configuration_directory,
            "-configuration_basename",
            configuration_basename,
        ],
    )

    cartographer_occupancy_grid_node = Node(
        package="cartographer_ros",
        executable="cartographer_occupancy_grid_node",
        name="cartographer_occupancy_grid_node",
        output="screen",
        condition=IfCondition(use_occupancy_grid),
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=[
            "-resolution",
            resolution,
            "-publish_period_sec",
            publish_period_sec,
        ],
    )

    slam_pose_adapter_node = Node(
        package="adaptive_fusion_slam_adapter",
        executable="slam_pose_adapter_node",
        name="slam_pose_adapter_node",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"input_mode": slam_input_mode},
            {"map_frame": map_frame},
            {"robot_frame": robot_frame},
            {"tracked_pose_topic": tracked_pose_topic},
            {"slam_pose_topic": slam_pose_topic},
            {"publish_rate": slam_pose_rate},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("resolution", default_value="0.05"),
            DeclareLaunchArgument("publish_period_sec", default_value="1.0"),
            DeclareLaunchArgument("use_occupancy_grid", default_value="false"),
            DeclareLaunchArgument("map_frame", default_value="map"),
            DeclareLaunchArgument("robot_frame", default_value="base_footprint"),
            DeclareLaunchArgument("slam_input_mode", default_value="tracked_pose"),
            DeclareLaunchArgument("tracked_pose_topic", default_value="/tracked_pose"),
            DeclareLaunchArgument("slam_pose_topic", default_value="/slam/pose"),
            DeclareLaunchArgument("slam_pose_rate", default_value="10.0"),
            DeclareLaunchArgument(
                "configuration_directory",
                default_value=os.path.join(pkg_share, "config"),
            ),
            DeclareLaunchArgument(
                "configuration_basename",
                default_value="turtlebot3_carto_config.lua",
            ),
            cartographer_node,
            cartographer_occupancy_grid_node,
            slam_pose_adapter_node,
        ]
    )
