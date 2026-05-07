import os
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def load_uwb_poses(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    return config.get("uwb_anchors", {})


def create_uwb_spawner(entity_name, uwb_sdf, namespace, pose):
    return Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-entity', entity_name,
            '-file', uwb_sdf,
            '-robot_namespace', namespace,
            '-x', str(pose.get('x', 0.0)),
            '-y', str(pose.get('y', 0.0)),
            '-z', str(pose.get('z', 0.01)),
            '-R', str(pose.get('roll', 0.0)),
            '-P', str(pose.get('pitch', 0.0)),
            '-Y', str(pose.get('yaw', 0.0)),
        ])


def generate_launch_description():
    pkg_share = get_package_share_directory("adaptive_fusion_gazebo")
    turtlebot3_gazebo_share = get_package_share_directory("turtlebot3_gazebo")
    uwb_pose_config = os.path.join(pkg_share, 'config', 'uwb_pose.yaml')
    uwb_poses = load_uwb_poses(uwb_pose_config)

    description_pkg_share = get_package_share_directory("adaptive_fusion_description")
    robot_description_path = os.path.join(description_pkg_share, 'urdf', 'turtlebot3_waffle.urdf')
    with open(robot_description_path, 'r') as file:
        robot_description = file.read()

    namespace = LaunchConfiguration('namespace')
    world = LaunchConfiguration('world')
    use_sim_time = LaunchConfiguration('use_sim_time')
    start_uwb = LaunchConfiguration('start_uwb')
    uwb_pose_file = LaunchConfiguration('uwb_pose_file')
    ground_truth_topic = LaunchConfiguration('ground_truth_topic')
    uwb_noise_stddev = LaunchConfiguration('uwb_noise_stddev')
    uwb_random_seed = LaunchConfiguration('uwb_random_seed')
    uwb_publish_pose_topic = LaunchConfiguration('uwb_publish_pose_topic')
    pose = {'x': LaunchConfiguration('x_pose', default='-2.00'),
            'y': LaunchConfiguration('y_pose', default='-0.50'),
            'z': LaunchConfiguration('z_pose', default='0.01'),
            'R': LaunchConfiguration('roll', default='0.00'),
            'P': LaunchConfiguration('pitch', default='0.00'),
            'Y': LaunchConfiguration('yaw', default='0.00')}
    robot_name = LaunchConfiguration('robot_name')
    robot_sdf = LaunchConfiguration('robot_sdf')
    uwb_sdf = LaunchConfiguration('uwb_sdf')

    EnvTurtleBot = SetEnvironmentVariable(
            name="TURTLEBOT3_MODEL",
            value="waffle")

    gazebo_model_paths = [
        os.path.join(turtlebot3_gazebo_share, "models"),
    ]
    existing_model_path = os.environ.get("GAZEBO_MODEL_PATH")
    if existing_model_path:
        gazebo_model_paths.append(existing_model_path)

    EnvModelPath = SetEnvironmentVariable(
        name="GAZEBO_MODEL_PATH",
        value=os.pathsep.join(gazebo_model_paths + ['/opt/ros/$ROS_DISTRO/share/turtlebot3_gazebo/models']))

    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top-level namespace')

    declare_world_cmd = DeclareLaunchArgument(
        'world',
        # TODO(orduno) Switch back once ROS argument passing has been fixed upstream
        #              https://github.com/ROBOTIS-GIT/turtlebot3_simulations/issues/91
        # default_value=os.path.join(get_package_share_directory('turtlebot3_gazebo'),
        # worlds/turtlebot3_worlds/waffle.model')
        default_value=os.path.join(pkg_share, 'worlds', 'turtlebot3_world.world'),
        description='Full path to world model file to load')

    declare_robot_name_cmd = DeclareLaunchArgument(
        'robot_name',
        default_value='turtlebot3_waffle_robot',
        description='name of the robot')

    declare_robot_sdf_cmd = DeclareLaunchArgument(
        'robot_sdf',
        default_value=os.path.join(pkg_share, 'models', 'turtlebot3_waffle', 'model.sdf'),
        description='Full path to robot sdf file to spawn the robot in gazebo')

    declare_uwb_sdf_cmd = DeclareLaunchArgument(
        'uwb_sdf',
        default_value=os.path.join(pkg_share, 'models', 'UWB_Base', 'model.sdf'),
        description='Full path to UWB sdf file to spawn in gazebo')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')

    declare_start_uwb_cmd = DeclareLaunchArgument(
        'start_uwb',
        default_value='true',
        description='Start the UWB simulation node together with Gazebo')

    declare_uwb_pose_file_cmd = DeclareLaunchArgument(
        'uwb_pose_file',
        default_value=uwb_pose_config,
        description='UWB anchor and disturbance configuration YAML file')

    declare_ground_truth_topic_cmd = DeclareLaunchArgument(
        'ground_truth_topic',
        default_value='/ground_truth/odom',
        description='Ground truth odometry topic consumed by the UWB simulator')

    declare_uwb_noise_stddev_cmd = DeclareLaunchArgument(
        'uwb_noise_stddev',
        default_value='0.1',
        description='Gaussian noise stddev in meters for simulated UWB ranges')

    declare_uwb_random_seed_cmd = DeclareLaunchArgument(
        'uwb_random_seed',
        default_value='-1',
        description='Random seed for the UWB simulator, -1 means random')

    declare_uwb_publish_pose_topic_cmd = DeclareLaunchArgument(
        'uwb_publish_pose_topic',
        default_value='/uwb/pose',
        description='Output topic for the simulated UWB position estimate')

    start_gazebo_server_cmd = ExecuteProcess(
        cmd=['gzserver', '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so', world],
        output='screen')

    start_gazebo_client_cmd = ExecuteProcess(
        cmd=['gzclient'],
        output='screen')

    start_gazebo_spawner_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-entity', robot_name,
            '-file', robot_sdf,
            '-robot_namespace', namespace,
            '-x', pose['x'], '-y', pose['y'], '-z', pose['z'],
            '-R', pose['R'], '-P', pose['P'], '-Y', pose['Y']])

    uwb_spawners = [
        create_uwb_spawner(entity_name, uwb_sdf, namespace, anchor_pose)
        for entity_name, anchor_pose in uwb_poses.items()
    ]

    start_robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=namespace,
        output='screen',
        parameters=[{'use_sim_time': use_sim_time,
                     'robot_description': robot_description}],
        )

    start_uwb_range_node_cmd = Node(
        package='adaptive_fusion_uwb',
        executable='uwb_range_node',
        name='uwb_range_node',
        output='screen',
        condition=IfCondition(start_uwb),
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'ground_truth_topic': ground_truth_topic,
                'uwb_pose_file': uwb_pose_file,
                'noise_stddev': uwb_noise_stddev,
                'random_seed': uwb_random_seed,
                'publish_pose': True,
                'publish_pose_topic': uwb_publish_pose_topic,
                'pose_frame': 'map',
            }
        ],
    )

    ld = LaunchDescription()

    # Add any set environment variables
    ld.add_action(EnvTurtleBot)
    ld.add_action(EnvModelPath)

    # Add any declare launch arguments
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_world_cmd)
    ld.add_action(declare_robot_name_cmd)
    ld.add_action(declare_robot_sdf_cmd)
    ld.add_action(declare_uwb_sdf_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_start_uwb_cmd)
    ld.add_action(declare_uwb_pose_file_cmd)
    ld.add_action(declare_ground_truth_topic_cmd)
    ld.add_action(declare_uwb_noise_stddev_cmd)
    ld.add_action(declare_uwb_random_seed_cmd)
    ld.add_action(declare_uwb_publish_pose_topic_cmd)
    # Add any conditioned actions
    ld.add_action(start_gazebo_server_cmd)
    ld.add_action(start_gazebo_client_cmd)
    ld.add_action(start_gazebo_spawner_cmd)
    ld.add_action(start_robot_state_publisher_cmd)
    ld.add_action(start_uwb_range_node_cmd)
    for uwb_spawner in uwb_spawners:
        ld.add_action(uwb_spawner)

    return ld
