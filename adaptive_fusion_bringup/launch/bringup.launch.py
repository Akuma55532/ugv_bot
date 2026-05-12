import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    gazebo_share = get_package_share_directory("adaptive_fusion_gazebo")
    slam_share = get_package_share_directory("adaptive_fusion_slam_adapter")
    ekf_share = get_package_share_directory("adaptive_fusion_ekf")
    nav_share = get_package_share_directory("adaptive_fusion_navigation")

    use_sim_time = LaunchConfiguration("use_sim_time")
    world = LaunchConfiguration("world")
    map_yaml = LaunchConfiguration("map")

    use_nav2 = LaunchConfiguration("use_nav2")
    use_static_map = LaunchConfiguration("use_static_map")
    use_rviz = LaunchConfiguration("use_rviz")
    fusion_method = LaunchConfiguration("fusion_method")

    start_uwb = LaunchConfiguration("start_uwb")
    start_ekf = LaunchConfiguration("start_ekf")

    autostart = LaunchConfiguration("autostart")
    use_composition = LaunchConfiguration("use_composition")
    use_respawn = LaunchConfiguration("use_respawn")
    log_level = LaunchConfiguration("log_level")

    spawn_x = LaunchConfiguration("x_pose")
    spawn_y = LaunchConfiguration("y_pose")
    spawn_z = LaunchConfiguration("z_pose")
    spawn_yaw = LaunchConfiguration("yaw")

    uwb_pose_file = LaunchConfiguration("uwb_pose_file")
    uwb_noise_stddev = LaunchConfiguration("uwb_noise_stddev")
    uwb_random_seed = LaunchConfiguration("uwb_random_seed")
    uwb_publish_pose_topic = LaunchConfiguration("uwb_publish_pose_topic")
    ground_truth_topic = LaunchConfiguration("ground_truth_topic")

    ekf_params_file = LaunchConfiguration("ekf_params_file")
    nav2_params_file = LaunchConfiguration("nav2_params_file")
    carto_config_dir = LaunchConfiguration("cartographer_config_dir")
    carto_config_basename = LaunchConfiguration("cartographer_config_basename")
    slam_pose_topic = LaunchConfiguration("slam_pose_topic")
    tracked_pose_topic = LaunchConfiguration("tracked_pose_topic")
    slam_pose_rate = LaunchConfiguration("slam_pose_rate")
    occupancy_grid_resolution = LaunchConfiguration("occupancy_grid_resolution")
    occupancy_grid_publish_period = LaunchConfiguration(
        "occupancy_grid_publish_period"
    )
    rviz_config = LaunchConfiguration("rviz_config")

    ekf_mode = PythonExpression(["'", fusion_method, "'"])
    use_uwb_measurement_covariance = PythonExpression(
        [
            "'true' if '",
            fusion_method,
            "' in ['uwb_adaptive', 'dual_adaptive'] else 'false'"
        ]
    )
    use_slam_measurement_covariance = PythonExpression(
        [
            "'true' if '",
            fusion_method,
            "' in ['slam_adaptive', 'dual_adaptive'] else 'false'"
        ]
    )

    nav2_enabled_in_slam_mode = PythonExpression(
        ["'", use_nav2, "' == 'true' and '", use_static_map, "' != 'true'"]
    )
    nav2_enabled_in_static_mode = PythonExpression(
        ["'", use_nav2, "' == 'true' and '", use_static_map, "' == 'true'"]
    )
    use_occupancy_grid = PythonExpression(
        [
            "'true' if '",
            use_nav2,
            "' == 'true' and '",
            use_static_map,
            "' != 'true' else 'false'"
        ]
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_share, "launch", "spawn_car.launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "world": world,
            "x_pose": spawn_x,
            "y_pose": spawn_y,
            "z_pose": spawn_z,
            "yaw": spawn_yaw,
            "start_uwb": start_uwb,
            "uwb_pose_file": uwb_pose_file,
            "ground_truth_topic": ground_truth_topic,
            "uwb_noise_stddev": uwb_noise_stddev,
            "uwb_random_seed": uwb_random_seed,
            "uwb_publish_pose_topic": uwb_publish_pose_topic,
        }.items(),
    )

    cartographer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_share, "launch", "carto_slam.launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "configuration_directory": carto_config_dir,
            "configuration_basename": carto_config_basename,
            "use_occupancy_grid": use_occupancy_grid,
            "resolution": occupancy_grid_resolution,
            "publish_period_sec": occupancy_grid_publish_period,
            "slam_input_mode": "tracked_pose",
            "tracked_pose_topic": tracked_pose_topic,
            "slam_pose_topic": slam_pose_topic,
            "slam_pose_rate": slam_pose_rate,
        }.items(),
    )

    ekf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ekf_share, "launch", "ekf.launch.py")
        ),
        condition=IfCondition(start_ekf),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": ekf_params_file,
            "mode": ekf_mode,
            "use_measurement_covariance": "true",
            "use_uwb_measurement_covariance": use_uwb_measurement_covariance,
            "use_slam_measurement_covariance": use_slam_measurement_covariance,
            "use_imu_measurement_covariance": "true",
        }.items(),
    )

    nav2_slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav_share, "launch", "nav2_navigation.launch.py")
        ),
        condition=IfCondition(nav2_enabled_in_slam_mode),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": nav2_params_file,
            "autostart": autostart,
            "use_composition": use_composition,
            "use_respawn": use_respawn,
            "log_level": log_level,
            "use_rviz": use_rviz,
            "rviz_config": rviz_config,
        }.items(),
    )

    nav2_map_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav_share, "launch", "nav2_bringup_map.launch.py")
        ),
        condition=IfCondition(nav2_enabled_in_static_mode),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": nav2_params_file,
            "map": map_yaml,
            "autostart": autostart,
            "use_composition": use_composition,
            "use_respawn": use_respawn,
            "log_level": log_level,
            "use_rviz": use_rviz,
            "rviz_config": rviz_config,
        }.items(),
    )

    delayed_slam = TimerAction(
        period=2.0,
        actions=[cartographer_launch],
    )

    delayed_ekf = TimerAction(
        period=4.0,
        actions=[ekf_launch],
    )

    delayed_nav2_slam = TimerAction(
        period=6.0,
        actions=[nav2_slam_launch],
    )

    delayed_nav2_map = TimerAction(
        period=6.0,
        actions=[nav2_map_launch],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "world",
                default_value=os.path.join(
                    gazebo_share, "worlds", "turtlebot3_world.world"
                ),
            ),
            DeclareLaunchArgument(
                "map",
                default_value=os.path.join(
                    nav_share, "maps", "turtlebot3_world.yaml"
                ),
            ),
            DeclareLaunchArgument("use_nav2", default_value="true"),
            DeclareLaunchArgument("use_static_map", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument(
                "fusion_method",
                default_value="dual_adaptive",
            ),
            DeclareLaunchArgument("start_uwb", default_value="true"),
            DeclareLaunchArgument("start_ekf", default_value="true"),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument("use_composition", default_value="False"),
            DeclareLaunchArgument("use_respawn", default_value="False"),
            DeclareLaunchArgument("log_level", default_value="info"),
            DeclareLaunchArgument("x_pose", default_value="-2.00"),
            DeclareLaunchArgument("y_pose", default_value="-0.50"),
            DeclareLaunchArgument("z_pose", default_value="0.01"),
            DeclareLaunchArgument("yaw", default_value="0.00"),
            DeclareLaunchArgument(
                "uwb_pose_file",
                default_value=os.path.join(gazebo_share, "config", "uwb_pose.yaml"),
            ),
            DeclareLaunchArgument(
                "ground_truth_topic",
                default_value="/ground_truth/odom",
            ),
            DeclareLaunchArgument("uwb_noise_stddev", default_value="0.1"),
            DeclareLaunchArgument("uwb_random_seed", default_value="-1"),
            DeclareLaunchArgument(
                "uwb_publish_pose_topic",
                default_value="/uwb/pose",
            ),
            DeclareLaunchArgument(
                "ekf_params_file",
                default_value=os.path.join(ekf_share, "config", "ekf_params.yaml"),
            ),
            DeclareLaunchArgument(
                "nav2_params_file",
                default_value=os.path.join(
                    nav_share, "config", "nav2_fusion_params.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "cartographer_config_dir",
                default_value=os.path.join(slam_share, "config"),
            ),
            DeclareLaunchArgument(
                "cartographer_config_basename",
                default_value="turtlebot3_carto_config.lua",
            ),
            DeclareLaunchArgument("slam_pose_topic", default_value="/slam/pose"),
            DeclareLaunchArgument("tracked_pose_topic", default_value="/tracked_pose"),
            DeclareLaunchArgument("slam_pose_rate", default_value="10.0"),
            DeclareLaunchArgument(
                "occupancy_grid_resolution",
                default_value="0.05",
            ),
            DeclareLaunchArgument(
                "occupancy_grid_publish_period",
                default_value="1.0",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=os.path.join(
                    get_package_share_directory("nav2_bringup"),
                    "rviz",
                    "nav2_default_view.rviz",
                ),
            ),
            gazebo_launch,
            delayed_slam,
            delayed_ekf,
            delayed_nav2_map,
        ]
    )
