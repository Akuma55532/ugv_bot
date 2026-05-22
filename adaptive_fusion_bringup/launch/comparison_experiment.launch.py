import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    bringup_share = get_package_share_directory("adaptive_fusion_bringup")
    gazebo_share = get_package_share_directory("adaptive_fusion_gazebo")
    ekf_share = get_package_share_directory("adaptive_fusion_ekf")
    eval_share = get_package_share_directory("adaptive_fusion_eval")

    fusion_method = LaunchConfiguration("fusion_method")
    use_rviz = LaunchConfiguration("use_rviz")
    use_nav2 = LaunchConfiguration("use_nav2")
    use_static_map = LaunchConfiguration("use_static_map")
    uwb_random_seed = LaunchConfiguration("uwb_random_seed")
    use_clean_uwb = LaunchConfiguration("use_clean_uwb")
    uwb_noise_stddev = LaunchConfiguration("uwb_noise_stddev")
    enable_slam_disturbance = LaunchConfiguration("enable_slam_disturbance")
    slam_disturbance_position_stddev = LaunchConfiguration(
        "slam_disturbance_position_stddev"
    )
    slam_disturbance_yaw_stddev = LaunchConfiguration(
        "slam_disturbance_yaw_stddev"
    )
    slam_disturbance_seed = LaunchConfiguration("slam_disturbance_seed")
    start_eval = LaunchConfiguration("start_eval")
    eval_output_dir = LaunchConfiguration("eval_output_dir")
    eval_alignment_mode = LaunchConfiguration("eval_alignment_mode")
    eval_start_delay_sec = LaunchConfiguration("eval_start_delay_sec")
    eval_duration_sec = LaunchConfiguration("eval_duration_sec")
    fixed_ekf_params_file = LaunchConfiguration("fixed_ekf_params_file")
    uwb_adaptive_ekf_params_file = LaunchConfiguration(
        "uwb_adaptive_ekf_params_file"
    )
    slam_adaptive_ekf_params_file = LaunchConfiguration(
        "slam_adaptive_ekf_params_file"
    )
    dual_adaptive_ekf_params_file = LaunchConfiguration(
        "dual_adaptive_ekf_params_file"
    )
    only_uwb_ekf_params_file = LaunchConfiguration("only_uwb_ekf_params_file")
    only_slam_ekf_params_file = LaunchConfiguration("only_slam_ekf_params_file")

    ekf_params_file = PythonExpression(
        [
            "'",
            fixed_ekf_params_file,
            "' if '",
            fusion_method,
            "' == 'fixed' else '",
            uwb_adaptive_ekf_params_file,
            "' if '",
            fusion_method,
            "' == 'uwb_adaptive' else '",
            slam_adaptive_ekf_params_file,
            "' if '",
            fusion_method,
            "' == 'slam_adaptive' else '",
            only_uwb_ekf_params_file,
            "' if '",
            fusion_method,
            "' == 'only_uwb' else '",
            only_slam_ekf_params_file,
            "' if '",
            fusion_method,
            "' == 'only_slam' else '",
            dual_adaptive_ekf_params_file,
            "'",
        ]
    )
    uwb_pose_file = PythonExpression(
        [
            "'",
            os.path.join(gazebo_share, "config", "uwb_pose_clean.yaml"),
            "' if '",
            use_clean_uwb,
            "' == 'true' else '",
            os.path.join(gazebo_share, "config", "uwb_pose_formal_comparison.yaml"),
            "'",
        ]
    )
    uwb_enable_range_noise = PythonExpression(
        ["'false' if '", use_clean_uwb, "' == 'true' else 'true'"]
    )
    uwb_enable_zone_disturbance = PythonExpression(
        ["'false' if '", use_clean_uwb, "' == 'true' else 'true'"]
    )
    uwb_ideal_pose_mode = PythonExpression(
        ["'true' if '", use_clean_uwb, "' == 'true' else 'false'"]
    )

    bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, "launch", "bringup.launch.py")
        ),
        launch_arguments={
            "fusion_method": fusion_method,
            "use_rviz": use_rviz,
            "use_nav2": use_nav2,
            "use_static_map": use_static_map,
            "uwb_random_seed": uwb_random_seed,
            "uwb_pose_file": uwb_pose_file,
            "uwb_noise_stddev": uwb_noise_stddev,
            "uwb_enable_range_noise": uwb_enable_range_noise,
            "uwb_enable_zone_disturbance": uwb_enable_zone_disturbance,
            "uwb_ideal_pose_mode": uwb_ideal_pose_mode,
            "uwb_adaptive_covariance": "true",
            "uwb_position_covariance_xy": "0.02",
            "uwb_consistency_min_score": "0.15",
            "uwb_consistency_max_scale": "8.0",
            "ekf_params_file": ekf_params_file,
            "slam_adaptive_covariance": "true",
            "slam_position_covariance_xy": "0.015",
            "slam_yaw_covariance": "0.015",
            "slam_inject_noise": enable_slam_disturbance,
            "slam_position_noise_stddev": slam_disturbance_position_stddev,
            "slam_yaw_noise_stddev": slam_disturbance_yaw_stddev,
            "slam_noise_seed": slam_disturbance_seed,
        }.items(),
    )

    eval_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(eval_share, "launch", "eval.launch.py")
        ),
        condition=IfCondition(start_eval),
        launch_arguments={
            "method_name": fusion_method,
            "output_dir": eval_output_dir,
            "alignment_mode": eval_alignment_mode,
            "eval_start_delay_sec": eval_start_delay_sec,
            "eval_duration_sec": eval_duration_sec,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("fusion_method", default_value="dual_adaptive"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("use_nav2", default_value="true"),
            DeclareLaunchArgument("use_static_map", default_value="true"),
            DeclareLaunchArgument("uwb_random_seed", default_value="7"),
            DeclareLaunchArgument("use_clean_uwb", default_value="true"),
            DeclareLaunchArgument("uwb_noise_stddev", default_value="0.06"),
            DeclareLaunchArgument("enable_slam_disturbance", default_value="true"),
            DeclareLaunchArgument(
                "slam_disturbance_position_stddev", default_value="0.03"
            ),
            DeclareLaunchArgument(
                "slam_disturbance_yaw_stddev", default_value="0.06"
            ),
            DeclareLaunchArgument("slam_disturbance_seed", default_value="11"),
            DeclareLaunchArgument("start_eval", default_value="true"),
            DeclareLaunchArgument(
                "eval_output_dir",
                default_value=os.path.join("/tmp", "adaptive_fusion_eval"),
            ),
            DeclareLaunchArgument("eval_alignment_mode", default_value="initial_pose"),
            DeclareLaunchArgument("eval_start_delay_sec", default_value="3.0"),
            DeclareLaunchArgument("eval_duration_sec", default_value="0.0"),
            DeclareLaunchArgument(
                "fixed_ekf_params_file",
                default_value=os.path.join(
                    ekf_share, "config", "ekf_params_fixed_comparison.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "uwb_adaptive_ekf_params_file",
                default_value=os.path.join(
                    ekf_share, "config", "ekf_params_uwb_adaptive_comparison.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "slam_adaptive_ekf_params_file",
                default_value=os.path.join(
                    ekf_share, "config", "ekf_params_slam_adaptive_comparison.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "dual_adaptive_ekf_params_file",
                default_value=os.path.join(
                    ekf_share, "config", "ekf_params_dual_adaptive_comparison.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "only_uwb_ekf_params_file",
                default_value=os.path.join(
                    ekf_share, "config", "ekf_params_only_uwb_comparison.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "only_slam_ekf_params_file",
                default_value=os.path.join(
                    ekf_share, "config", "ekf_params_only_slam_comparison.yaml"
                ),
            ),
            bringup_launch,
            eval_launch,
        ]
    )
