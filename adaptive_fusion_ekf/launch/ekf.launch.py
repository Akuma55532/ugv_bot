import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("adaptive_fusion_ekf").find("adaptive_fusion_ekf")

    use_sim_time = LaunchConfiguration("use_sim_time")
    params_file = LaunchConfiguration("params_file")
    mode = LaunchConfiguration("mode")
    use_measurement_covariance = LaunchConfiguration("use_measurement_covariance")
    use_uwb_measurement_covariance = LaunchConfiguration(
        "use_uwb_measurement_covariance"
    )
    use_slam_measurement_covariance = LaunchConfiguration(
        "use_slam_measurement_covariance"
    )
    use_imu_measurement_covariance = LaunchConfiguration(
        "use_imu_measurement_covariance"
    )

    ekf_fusion_node = Node(
        package="adaptive_fusion_ekf",
        executable="ekf_fusion_node",
        name="ekf_fusion_node",
        output="screen",
        parameters=[
            params_file,
            {
                "use_sim_time": use_sim_time,
                "mode": mode,
                "use_measurement_covariance": use_measurement_covariance,
                "use_uwb_measurement_covariance": use_uwb_measurement_covariance,
                "use_slam_measurement_covariance": use_slam_measurement_covariance,
                "use_imu_measurement_covariance": use_imu_measurement_covariance,
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "params_file",
                default_value=os.path.join(pkg_share, "config", "ekf_params.yaml"),
            ),
            DeclareLaunchArgument("mode", default_value="dual_adaptive"),
            DeclareLaunchArgument("use_measurement_covariance", default_value="true"),
            DeclareLaunchArgument(
                "use_uwb_measurement_covariance", default_value="true"
            ),
            DeclareLaunchArgument(
                "use_slam_measurement_covariance", default_value="true"
            ),
            DeclareLaunchArgument(
                "use_imu_measurement_covariance", default_value="true"
            ),
            ekf_fusion_node,
        ]
    )
