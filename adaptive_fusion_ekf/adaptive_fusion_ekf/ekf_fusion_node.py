import math
from collections import deque
from typing import Deque
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = 0.5 * yaw
    return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return sum((value - avg) * (value - avg) for value in values) / float(len(values))


class EkfFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("ekf_fusion_node")

        self.declare_parameter("mode", "fixed")
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("uwb_topic", "/uwb/pose")
        self.declare_parameter("slam_topic", "/slam/pose")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("fusion_pose_topic", "/fusion/pose")
        self.declare_parameter("fusion_global_odom_topic", "/fusion/global_odom")
        self.declare_parameter("fusion_odom_topic", "/fusion/odom")
        self.declare_parameter("fusion_path_topic", "/fusion/path")
        self.declare_parameter("debug_topic", "/fusion/debug")
        self.declare_parameter("publish_global_pose", True)
        self.declare_parameter("publish_global_odom", True)
        self.declare_parameter("publish_local_odom", True)
        self.declare_parameter("publish_map_odom_tf", True)
        self.declare_parameter("path_publish_stride", 5)
        self.declare_parameter("path_max_length", 500)
        self.declare_parameter("use_measurement_covariance", True)
        self.declare_parameter("use_uwb_measurement_covariance", True)
        self.declare_parameter("use_slam_measurement_covariance", True)
        self.declare_parameter("use_imu_measurement_covariance", True)
        self.declare_parameter("min_measurement_variance", 1e-4)

        self.declare_parameter("initial_covariance_x", 1.0)
        self.declare_parameter("initial_covariance_y", 1.0)
        self.declare_parameter("initial_covariance_theta", 0.5)

        self.declare_parameter("process_noise_x", 0.05)
        self.declare_parameter("process_noise_y", 0.05)
        self.declare_parameter("process_noise_theta", 0.02)

        self.declare_parameter("uwb_noise_x", 0.20)
        self.declare_parameter("uwb_noise_y", 0.20)
        self.declare_parameter("slam_noise_x", 0.05)
        self.declare_parameter("slam_noise_y", 0.05)
        self.declare_parameter("slam_noise_theta", 0.05)
        self.declare_parameter("imu_noise_theta", 0.03)

        self.declare_parameter("adaptive_window_size", 10)

        self.declare_parameter("uwb_min_score", 0.05)
        self.declare_parameter("uwb_max_scale", 20.0)
        self.declare_parameter("uwb_innovation_beta", 0.60)
        self.declare_parameter("uwb_jump_beta", 0.45)
        self.declare_parameter("uwb_variance_beta", 0.30)
        self.declare_parameter("uwb_innovation_weight", 0.50)
        self.declare_parameter("uwb_jump_weight", 0.25)
        self.declare_parameter("uwb_variance_weight", 0.25)

        self.declare_parameter("slam_min_score", 0.05)
        self.declare_parameter("slam_max_scale", 20.0)
        self.declare_parameter("slam_position_innovation_beta", 0.30)
        self.declare_parameter("slam_yaw_innovation_beta", 0.25)
        self.declare_parameter("slam_step_position_beta", 0.18)
        self.declare_parameter("slam_step_yaw_beta", 0.18)
        self.declare_parameter("slam_odom_consistency_beta", 0.20)
        self.declare_parameter("slam_position_weight", 0.35)
        self.declare_parameter("slam_yaw_weight", 0.20)
        self.declare_parameter("slam_step_position_weight", 0.15)
        self.declare_parameter("slam_step_yaw_weight", 0.10)
        self.declare_parameter("slam_odom_consistency_weight", 0.20)

        self.mode = self.get_parameter("mode").get_parameter_value().string_value
        self.world_frame = (
            self.get_parameter("world_frame").get_parameter_value().string_value
        )
        self.odom_frame = (
            self.get_parameter("odom_frame").get_parameter_value().string_value
        )
        self.base_frame = (
            self.get_parameter("base_frame").get_parameter_value().string_value
        )
        self.fusion_pose_topic = (
            self.get_parameter("fusion_pose_topic").get_parameter_value().string_value
        )
        self.fusion_global_odom_topic = (
            self.get_parameter("fusion_global_odom_topic")
            .get_parameter_value()
            .string_value
        )
        self.fusion_odom_topic = (
            self.get_parameter("fusion_odom_topic").get_parameter_value().string_value
        )
        self.fusion_path_topic = (
            self.get_parameter("fusion_path_topic").get_parameter_value().string_value
        )
        self.debug_topic = (
            self.get_parameter("debug_topic").get_parameter_value().string_value
        )
        self.publish_global_pose = (
            self.get_parameter("publish_global_pose").get_parameter_value().bool_value
        )
        self.publish_global_odom = (
            self.get_parameter("publish_global_odom").get_parameter_value().bool_value
        )
        self.publish_local_odom = (
            self.get_parameter("publish_local_odom").get_parameter_value().bool_value
        )
        self.publish_map_odom_tf = (
            self.get_parameter("publish_map_odom_tf").get_parameter_value().bool_value
        )
        self.path_publish_stride = (
            self.get_parameter("path_publish_stride").get_parameter_value().integer_value
        )
        self.path_max_length = (
            self.get_parameter("path_max_length").get_parameter_value().integer_value
        )
        self.use_measurement_covariance = (
            self.get_parameter("use_measurement_covariance")
            .get_parameter_value()
            .bool_value
        )
        self.use_uwb_measurement_covariance = (
            self.get_parameter("use_uwb_measurement_covariance")
            .get_parameter_value()
            .bool_value
        )
        self.use_slam_measurement_covariance = (
            self.get_parameter("use_slam_measurement_covariance")
            .get_parameter_value()
            .bool_value
        )
        self.use_imu_measurement_covariance = (
            self.get_parameter("use_imu_measurement_covariance")
            .get_parameter_value()
            .bool_value
        )
        self.min_measurement_variance = (
            self.get_parameter("min_measurement_variance")
            .get_parameter_value()
            .double_value
        )
        adaptive_window_size = max(
            2,
            self.get_parameter("adaptive_window_size")
            .get_parameter_value()
            .integer_value,
        )

        self.uwb_topic = self.get_parameter("uwb_topic").get_parameter_value().string_value
        self.slam_topic = (
            self.get_parameter("slam_topic").get_parameter_value().string_value
        )
        self.imu_topic = self.get_parameter("imu_topic").get_parameter_value().string_value
        self.odom_topic = (
            self.get_parameter("odom_topic").get_parameter_value().string_value
        )

        self.initial_covariance = np.diag(
            [
                self.get_parameter("initial_covariance_x")
                .get_parameter_value()
                .double_value,
                self.get_parameter("initial_covariance_y")
                .get_parameter_value()
                .double_value,
                self.get_parameter("initial_covariance_theta")
                .get_parameter_value()
                .double_value,
            ]
        )
        self.Q_base = np.diag(
            [
                self.get_parameter("process_noise_x").get_parameter_value().double_value,
                self.get_parameter("process_noise_y").get_parameter_value().double_value,
                self.get_parameter("process_noise_theta").get_parameter_value().double_value,
            ]
        )
        self.R_uwb_fixed = np.diag(
            [
                self.get_parameter("uwb_noise_x").get_parameter_value().double_value,
                self.get_parameter("uwb_noise_y").get_parameter_value().double_value,
            ]
        )
        self.R_slam_fixed = np.diag(
            [
                self.get_parameter("slam_noise_x").get_parameter_value().double_value,
                self.get_parameter("slam_noise_y").get_parameter_value().double_value,
                self.get_parameter("slam_noise_theta").get_parameter_value().double_value,
            ]
        )
        self.R_imu_fixed = np.array(
            [[self.get_parameter("imu_noise_theta").get_parameter_value().double_value]]
        )
        self.uwb_min_score = (
            self.get_parameter("uwb_min_score").get_parameter_value().double_value
        )
        self.uwb_max_scale = (
            self.get_parameter("uwb_max_scale").get_parameter_value().double_value
        )
        self.uwb_innovation_beta = (
            self.get_parameter("uwb_innovation_beta")
            .get_parameter_value()
            .double_value
        )
        self.uwb_jump_beta = (
            self.get_parameter("uwb_jump_beta").get_parameter_value().double_value
        )
        self.uwb_variance_beta = (
            self.get_parameter("uwb_variance_beta")
            .get_parameter_value()
            .double_value
        )
        self.uwb_innovation_weight = (
            self.get_parameter("uwb_innovation_weight")
            .get_parameter_value()
            .double_value
        )
        self.uwb_jump_weight = (
            self.get_parameter("uwb_jump_weight").get_parameter_value().double_value
        )
        self.uwb_variance_weight = (
            self.get_parameter("uwb_variance_weight")
            .get_parameter_value()
            .double_value
        )

        self.slam_min_score = (
            self.get_parameter("slam_min_score").get_parameter_value().double_value
        )
        self.slam_max_scale = (
            self.get_parameter("slam_max_scale").get_parameter_value().double_value
        )
        self.slam_position_innovation_beta = (
            self.get_parameter("slam_position_innovation_beta")
            .get_parameter_value()
            .double_value
        )
        self.slam_yaw_innovation_beta = (
            self.get_parameter("slam_yaw_innovation_beta")
            .get_parameter_value()
            .double_value
        )
        self.slam_step_position_beta = (
            self.get_parameter("slam_step_position_beta")
            .get_parameter_value()
            .double_value
        )
        self.slam_step_yaw_beta = (
            self.get_parameter("slam_step_yaw_beta")
            .get_parameter_value()
            .double_value
        )
        self.slam_odom_consistency_beta = (
            self.get_parameter("slam_odom_consistency_beta")
            .get_parameter_value()
            .double_value
        )
        self.slam_position_weight = (
            self.get_parameter("slam_position_weight")
            .get_parameter_value()
            .double_value
        )
        self.slam_yaw_weight = (
            self.get_parameter("slam_yaw_weight").get_parameter_value().double_value
        )
        self.slam_step_position_weight = (
            self.get_parameter("slam_step_position_weight")
            .get_parameter_value()
            .double_value
        )
        self.slam_step_yaw_weight = (
            self.get_parameter("slam_step_yaw_weight")
            .get_parameter_value()
            .double_value
        )
        self.slam_odom_consistency_weight = (
            self.get_parameter("slam_odom_consistency_weight")
            .get_parameter_value()
            .double_value
        )

        self.state = np.zeros(3, dtype=float)
        self.covariance = self.initial_covariance.copy()
        self.initialized = False
        self.last_odom_time: Optional[float] = None
        self.last_velocity_x = 0.0
        self.last_velocity_yaw = 0.0
        self.path_counter = 0

        self.predict_count = 0
        self.uwb_update_count = 0
        self.slam_update_count = 0
        self.imu_update_count = 0
        self.last_update_source = "none"
        self.initialization_source = "none"

        self.latest_raw_odom_pose: Optional[np.ndarray] = None
        self.latest_raw_odom_pose_covariance = [0.0] * 36
        self.latest_raw_odom_twist_covariance = [0.0] * 36
        self.previous_raw_odom_pose: Optional[np.ndarray] = None

        self.uwb_history: Deque[np.ndarray] = deque(maxlen=adaptive_window_size)
        self.previous_uwb_measurement: Optional[np.ndarray] = None
        self.latest_uwb_score = 1.0
        self.latest_uwb_scale = 1.0

        self.slam_history: Deque[np.ndarray] = deque(maxlen=adaptive_window_size)
        self.previous_slam_measurement: Optional[np.ndarray] = None
        self.latest_slam_score = 1.0
        self.latest_slam_scale = 1.0

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.world_frame

        self.uwb_subscription = self.create_subscription(
            PoseWithCovarianceStamped, self.uwb_topic, self.uwb_callback, 10
        )
        self.slam_subscription = self.create_subscription(
            PoseWithCovarianceStamped, self.slam_topic, self.slam_callback, 10
        )
        self.imu_subscription = self.create_subscription(
            Imu, self.imu_topic, self.imu_callback, 10
        )
        self.odom_subscription = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 50
        )

        self.fusion_pose_publisher = None
        if self.publish_global_pose:
            self.fusion_pose_publisher = self.create_publisher(
                PoseWithCovarianceStamped, self.fusion_pose_topic, 10
            )

        self.fusion_global_odom_publisher = None
        if self.publish_global_odom:
            self.fusion_global_odom_publisher = self.create_publisher(
                Odometry, self.fusion_global_odom_topic, 10
            )

        self.fusion_odom_publisher = None
        if self.publish_local_odom:
            self.fusion_odom_publisher = self.create_publisher(
                Odometry, self.fusion_odom_topic, 10
            )

        self.fusion_path_publisher = self.create_publisher(
            Path, self.fusion_path_topic, 10
        )
        self.debug_publisher = self.create_publisher(
            Float64MultiArray, self.debug_topic, 10
        )
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            "EKF fusion node started. "
            f"mode={self.mode}, world_frame={self.world_frame}, "
            f"odom_frame={self.odom_frame}, base_frame={self.base_frame}, "
            f"publish_map_odom_tf={self.publish_map_odom_tf}, "
            f"use_uwb_measurement_covariance={self.use_uwb_measurement_covariance}, "
            f"use_slam_measurement_covariance={self.use_slam_measurement_covariance}, "
            f"use_imu_measurement_covariance={self.use_imu_measurement_covariance}"
        )

    def odom_callback(self, msg: Odometry) -> None:
        current_time = self.stamp_to_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec)
        linear_velocity = msg.twist.twist.linear.x
        angular_velocity = msg.twist.twist.angular.z
        self.last_velocity_x = linear_velocity
        self.last_velocity_yaw = angular_velocity

        raw_yaw = quaternion_to_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        self.latest_raw_odom_pose = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, raw_yaw], dtype=float
        )
        self.latest_raw_odom_pose_covariance = list(msg.pose.covariance)
        self.latest_raw_odom_twist_covariance = list(msg.twist.covariance)

        if not self.initialized:
            self.state = self.latest_raw_odom_pose.copy()
            self.covariance = self.initial_covariance.copy()
            self.initialized = True
            self.initialization_source = "odom"
            self.last_update_source = "odom_init"
            self.last_odom_time = current_time
            self.previous_raw_odom_pose = self.latest_raw_odom_pose.copy()
            self.publish_outputs(msg.header.stamp)
            return

        if self.last_odom_time is None:
            self.last_odom_time = current_time
            self.previous_raw_odom_pose = self.latest_raw_odom_pose.copy()
            return

        dt = max(0.0, current_time - self.last_odom_time)
        self.last_odom_time = current_time
        if dt <= 1e-6:
            return

        self.predict(linear_velocity, angular_velocity, dt)
        self.predict_count += 1
        self.last_update_source = "predict"
        self.publish_outputs(msg.header.stamp)
        self.previous_raw_odom_pose = self.latest_raw_odom_pose.copy()

    def uwb_callback(self, msg: PoseWithCovarianceStamped) -> None:
        if not self.is_uwb_update_enabled():
            return

        measurement = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=float
        )
        if not self.initialized:
            self.state[0:2] = measurement
            self.covariance = self.initial_covariance.copy()
            self.initialized = True
            self.initialization_source = "uwb"
            self.last_update_source = "uwb_init"
        else:
            H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
            self.update(
                measurement,
                H,
                self.resolve_uwb_covariance(msg),
                normalize_indices=[],
            )
            self.uwb_update_count += 1
            self.last_update_source = "uwb"
        self.publish_outputs(msg.header.stamp)

    def slam_callback(self, msg: PoseWithCovarianceStamped) -> None:
        if not self.is_slam_update_enabled():
            return

        yaw = quaternion_to_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        measurement = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw], dtype=float
        )
        if not self.initialized:
            self.state = measurement
            self.covariance = self.initial_covariance.copy()
            self.initialized = True
            self.initialization_source = "slam"
            self.last_update_source = "slam_init"
        else:
            H = np.eye(3, dtype=float)
            self.update(
                measurement,
                H,
                self.resolve_slam_covariance(msg),
                normalize_indices=[2],
            )
            self.slam_update_count += 1
            self.last_update_source = "slam"
        self.publish_outputs(msg.header.stamp)

    def imu_callback(self, msg: Imu) -> None:
        yaw = quaternion_to_yaw(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        measurement = np.array([yaw], dtype=float)
        if not self.initialized:
            self.state[2] = yaw
            self.last_update_source = "imu_seed"
            return

        H = np.array([[0.0, 0.0, 1.0]], dtype=float)
        self.update(
            measurement,
            H,
            self.resolve_imu_covariance(msg),
            normalize_indices=[0],
        )
        self.imu_update_count += 1
        self.last_update_source = "imu"
        self.publish_outputs(msg.header.stamp)

    def predict(self, linear_velocity: float, angular_velocity: float, dt: float) -> None:
        theta = self.state[2]
        predicted_x = self.state[0] + linear_velocity * dt * math.cos(theta)
        predicted_y = self.state[1] + linear_velocity * dt * math.sin(theta)
        predicted_theta = normalize_angle(self.state[2] + angular_velocity * dt)

        F = np.array(
            [
                [1.0, 0.0, -linear_velocity * dt * math.sin(theta)],
                [0.0, 1.0, linear_velocity * dt * math.cos(theta)],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        Q = self.Q_base * max(dt, 1e-3)
        self.state = np.array([predicted_x, predicted_y, predicted_theta], dtype=float)
        self.covariance = F @ self.covariance @ F.T + Q
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def update(
        self,
        measurement: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        normalize_indices: list[int],
    ) -> None:
        innovation = measurement - H @ self.state
        for index in normalize_indices:
            innovation[index] = normalize_angle(float(innovation[index]))

        S = H @ self.covariance @ H.T + R
        try:
            K = self.covariance @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S = S + np.eye(S.shape[0], dtype=float) * 1e-6
            K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ innovation
        self.state[2] = normalize_angle(float(self.state[2]))

        identity = np.eye(3, dtype=float)
        innovation_projector = identity - K @ H
        self.covariance = (
            innovation_projector @ self.covariance @ innovation_projector.T + K @ R @ K.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def resolve_uwb_covariance(self, msg: PoseWithCovarianceStamped) -> np.ndarray:
        fallback = self.R_uwb_fixed.copy()
        if not self.is_uwb_adaptive_enabled():
            self.latest_uwb_score = 1.0
            self.latest_uwb_scale = 1.0
            return fallback
        measurement = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=float
        )
        score, scale = self.compute_uwb_adaptive_scale(measurement)
        self.latest_uwb_score = score
        self.latest_uwb_scale = scale
        return fallback * scale

    def resolve_slam_covariance(self, msg: PoseWithCovarianceStamped) -> np.ndarray:
        fallback = self.R_slam_fixed.copy()
        if not self.is_slam_adaptive_enabled():
            self.latest_slam_score = 1.0
            self.latest_slam_scale = 1.0
            return fallback
        yaw = quaternion_to_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        measurement = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw], dtype=float
        )
        score, scale = self.compute_slam_adaptive_scale(measurement)
        self.latest_slam_score = score
        self.latest_slam_scale = scale
        return fallback * scale

    def resolve_imu_covariance(self, msg: Imu) -> np.ndarray:
        fallback = self.R_imu_fixed.copy()
        if not (
            self.use_measurement_covariance and self.use_imu_measurement_covariance
        ):
            return fallback

        return np.array(
            [[self.safe_variance(msg.orientation_covariance[8], fallback[0, 0])]],
            dtype=float,
        )

    def is_uwb_adaptive_enabled(self) -> bool:
        return self.mode in ("uwb_adaptive", "dual_adaptive")

    def is_slam_adaptive_enabled(self) -> bool:
        return self.mode in ("slam_adaptive", "dual_adaptive")

    def is_uwb_update_enabled(self) -> bool:
        return self.mode != "only_slam"

    def is_slam_update_enabled(self) -> bool:
        return self.mode != "only_uwb"

    def compute_uwb_adaptive_scale(
        self, measurement: np.ndarray
    ) -> tuple[float, float]:
        innovation_norm = np.linalg.norm(measurement - self.state[0:2])
        innovation_score = math.exp(
            -(innovation_norm * innovation_norm)
            / max(self.uwb_innovation_beta * self.uwb_innovation_beta, 1e-6)
        )

        if self.previous_uwb_measurement is None:
            jump_norm = 0.0
        else:
            jump_norm = np.linalg.norm(measurement - self.previous_uwb_measurement)
        jump_score = math.exp(
            -(jump_norm * jump_norm)
            / max(self.uwb_jump_beta * self.uwb_jump_beta, 1e-6)
        )

        history = list(self.uwb_history)
        history.append(measurement.copy())
        x_values = [float(item[0]) for item in history]
        y_values = [float(item[1]) for item in history]
        measurement_variance = math.sqrt(variance(x_values) + variance(y_values))
        variance_score = math.exp(
            -(measurement_variance * measurement_variance)
            / max(self.uwb_variance_beta * self.uwb_variance_beta, 1e-6)
        )

        weight_sum = (
            self.uwb_innovation_weight
            + self.uwb_jump_weight
            + self.uwb_variance_weight
        )
        if weight_sum <= 1e-9:
            weight_sum = 1.0
        score = (
            self.uwb_innovation_weight * innovation_score
            + self.uwb_jump_weight * jump_score
            + self.uwb_variance_weight * variance_score
        ) / weight_sum
        score = max(self.uwb_min_score, min(1.0, score))
        scale = min(self.uwb_max_scale, max(1.0, 1.0 / max(score, 1e-6)))

        self.previous_uwb_measurement = measurement.copy()
        self.uwb_history.append(measurement.copy())
        return score, scale

    def compute_slam_adaptive_scale(
        self, measurement: np.ndarray
    ) -> tuple[float, float]:
        position_innovation = np.linalg.norm(measurement[0:2] - self.state[0:2])
        position_score = math.exp(
            -(position_innovation * position_innovation)
            / max(
                self.slam_position_innovation_beta
                * self.slam_position_innovation_beta,
                1e-6,
            )
        )

        yaw_innovation = abs(normalize_angle(float(measurement[2] - self.state[2])))
        yaw_score = math.exp(
            -(yaw_innovation * yaw_innovation)
            / max(self.slam_yaw_innovation_beta * self.slam_yaw_innovation_beta, 1e-6)
        )

        step_position = 0.0
        step_yaw = 0.0
        if self.previous_slam_measurement is not None:
            step_position = np.linalg.norm(measurement[0:2] - self.previous_slam_measurement[0:2])
            step_yaw = abs(
                normalize_angle(float(measurement[2] - self.previous_slam_measurement[2]))
            )
        step_position_score = math.exp(
            -(step_position * step_position)
            / max(
                self.slam_step_position_beta * self.slam_step_position_beta,
                1e-6,
            )
        )
        step_yaw_score = math.exp(
            -(step_yaw * step_yaw)
            / max(self.slam_step_yaw_beta * self.slam_step_yaw_beta, 1e-6)
        )

        odom_consistency_score = 1.0
        if self.previous_raw_odom_pose is not None and self.latest_raw_odom_pose is not None:
            odom_step_position = np.linalg.norm(
                self.latest_raw_odom_pose[0:2] - self.previous_raw_odom_pose[0:2]
            )
            odom_step_yaw = abs(
                normalize_angle(
                    float(self.latest_raw_odom_pose[2] - self.previous_raw_odom_pose[2])
                )
            )
            odom_consistency = math.sqrt(
                (step_position - odom_step_position) * (step_position - odom_step_position)
                + (step_yaw - odom_step_yaw) * (step_yaw - odom_step_yaw)
            )
            odom_consistency_score = math.exp(
                -(odom_consistency * odom_consistency)
                / max(
                    self.slam_odom_consistency_beta
                    * self.slam_odom_consistency_beta,
                    1e-6,
                )
            )

        weight_sum = (
            self.slam_position_weight
            + self.slam_yaw_weight
            + self.slam_step_position_weight
            + self.slam_step_yaw_weight
            + self.slam_odom_consistency_weight
        )
        if weight_sum <= 1e-9:
            weight_sum = 1.0
        score = (
            self.slam_position_weight * position_score
            + self.slam_yaw_weight * yaw_score
            + self.slam_step_position_weight * step_position_score
            + self.slam_step_yaw_weight * step_yaw_score
            + self.slam_odom_consistency_weight * odom_consistency_score
        ) / weight_sum
        score = max(self.slam_min_score, min(1.0, score))
        scale = min(self.slam_max_scale, max(1.0, 1.0 / max(score, 1e-6)))

        self.previous_slam_measurement = measurement.copy()
        self.slam_history.append(measurement.copy())
        return score, scale

    def safe_variance(self, variance: float, fallback: float) -> float:
        if not math.isfinite(variance) or variance < 0.0:
            return max(fallback, self.min_measurement_variance)
        if variance == 0.0:
            return max(fallback, self.min_measurement_variance)
        return max(variance, self.min_measurement_variance)

    def publish_outputs(self, stamp) -> None:
        if not self.initialized:
            return

        if self.fusion_pose_publisher is not None:
            self.fusion_pose_publisher.publish(self.build_global_pose_message(stamp))

        if self.fusion_global_odom_publisher is not None:
            self.fusion_global_odom_publisher.publish(
                self.build_global_odom_message(stamp)
            )

        if self.fusion_odom_publisher is not None and self.latest_raw_odom_pose is not None:
            self.fusion_odom_publisher.publish(self.build_local_odom_message(stamp))

        if self.publish_map_odom_tf and self.latest_raw_odom_pose is not None:
            self.tf_broadcaster.sendTransform(self.build_map_to_odom_transform(stamp))

        self.path_counter += 1
        if self.path_counter % max(1, self.path_publish_stride) == 0:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = stamp
            pose_stamped.header.frame_id = self.world_frame
            pose_stamped.pose.position.x = float(self.state[0])
            pose_stamped.pose.position.y = float(self.state[1])
            pose_stamped.pose.position.z = 0.0
            qx, qy, qz, qw = yaw_to_quaternion(float(self.state[2]))
            pose_stamped.pose.orientation.x = qx
            pose_stamped.pose.orientation.y = qy
            pose_stamped.pose.orientation.z = qz
            pose_stamped.pose.orientation.w = qw
            self.path_msg.header.stamp = stamp
            self.path_msg.poses.append(pose_stamped)
            if self.path_max_length > 0 and len(self.path_msg.poses) > self.path_max_length:
                self.path_msg.poses = self.path_msg.poses[-self.path_max_length :]
            self.fusion_path_publisher.publish(self.path_msg)

        debug_msg = Float64MultiArray()
        debug_msg.data = [
            float(self.state[0]),
            float(self.state[1]),
            float(self.state[2]),
            float(self.covariance[0, 0]),
            float(self.covariance[1, 1]),
            float(self.covariance[2, 2]),
            float(self.last_velocity_x),
            float(self.last_velocity_yaw),
            float(self.predict_count),
            float(self.uwb_update_count),
            float(self.slam_update_count),
            float(self.imu_update_count),
            self.source_to_debug_id(self.last_update_source),
            self.source_to_debug_id(self.initialization_source),
            1.0 if self.latest_raw_odom_pose is not None else 0.0,
            1.0 if self.publish_map_odom_tf else 0.0,
            float(self.latest_uwb_score),
            float(self.latest_uwb_scale),
            float(self.latest_slam_score),
            float(self.latest_slam_scale),
        ]
        self.debug_publisher.publish(debug_msg)

    def build_global_pose_message(self, stamp) -> PoseWithCovarianceStamped:
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.world_frame
        pose_msg.pose.pose.position.x = float(self.state[0])
        pose_msg.pose.pose.position.y = float(self.state[1])
        pose_msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(float(self.state[2]))
        pose_msg.pose.pose.orientation.x = qx
        pose_msg.pose.pose.orientation.y = qy
        pose_msg.pose.pose.orientation.z = qz
        pose_msg.pose.pose.orientation.w = qw
        pose_msg.pose.covariance = self.to_covariance_6x6(self.covariance)
        return pose_msg

    def build_global_odom_message(self, stamp) -> Odometry:
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.world_frame
        odom_msg.child_frame_id = self.base_frame
        odom_msg.pose.pose.position.x = float(self.state[0])
        odom_msg.pose.pose.position.y = float(self.state[1])
        odom_msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(float(self.state[2]))
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw
        odom_msg.pose.covariance = self.to_covariance_6x6(self.covariance)
        odom_msg.twist.twist.linear.x = float(self.last_velocity_x)
        odom_msg.twist.twist.angular.z = float(self.last_velocity_yaw)
        odom_msg.twist.covariance = list(self.latest_raw_odom_twist_covariance)
        return odom_msg

    def build_local_odom_message(self, stamp) -> Odometry:
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame
        raw_pose = self.latest_raw_odom_pose
        if raw_pose is None:
            return odom_msg

        odom_msg.pose.pose.position.x = float(raw_pose[0])
        odom_msg.pose.pose.position.y = float(raw_pose[1])
        odom_msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(float(raw_pose[2]))
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw
        odom_msg.pose.covariance = list(self.latest_raw_odom_pose_covariance)
        odom_msg.twist.twist.linear.x = float(self.last_velocity_x)
        odom_msg.twist.twist.angular.z = float(self.last_velocity_yaw)
        odom_msg.twist.covariance = list(self.latest_raw_odom_twist_covariance)
        return odom_msg

    def build_map_to_odom_transform(self, stamp) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.world_frame
        transform.child_frame_id = self.odom_frame

        raw_pose = self.latest_raw_odom_pose
        if raw_pose is None:
            return transform

        translation_x, translation_y, rotation_yaw = self.compute_map_to_odom_transform(
            fused_pose=self.state,
            raw_odom_pose=raw_pose,
        )
        qx, qy, qz, qw = yaw_to_quaternion(rotation_yaw)
        transform.transform.translation.x = float(translation_x)
        transform.transform.translation.y = float(translation_y)
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        return transform

    def compute_map_to_odom_transform(
        self, fused_pose: np.ndarray, raw_odom_pose: np.ndarray
    ) -> tuple[float, float, float]:
        rotation_yaw = normalize_angle(float(fused_pose[2] - raw_odom_pose[2]))
        cos_yaw = math.cos(rotation_yaw)
        sin_yaw = math.sin(rotation_yaw)
        translation_x = float(
            fused_pose[0] - (cos_yaw * raw_odom_pose[0] - sin_yaw * raw_odom_pose[1])
        )
        translation_y = float(
            fused_pose[1] - (sin_yaw * raw_odom_pose[0] + cos_yaw * raw_odom_pose[1])
        )
        return translation_x, translation_y, rotation_yaw

    def to_covariance_6x6(self, covariance_3x3: np.ndarray) -> list[float]:
        covariance = [0.0] * 36
        covariance[0] = float(covariance_3x3[0, 0])
        covariance[1] = float(covariance_3x3[0, 1])
        covariance[5] = float(covariance_3x3[0, 2])
        covariance[6] = float(covariance_3x3[1, 0])
        covariance[7] = float(covariance_3x3[1, 1])
        covariance[11] = float(covariance_3x3[1, 2])
        covariance[30] = float(covariance_3x3[2, 0])
        covariance[31] = float(covariance_3x3[2, 1])
        covariance[35] = float(covariance_3x3[2, 2])
        return covariance

    def source_to_debug_id(self, source: str) -> float:
        source_map = {
            "none": 0.0,
            "odom": 1.0,
            "odom_init": 2.0,
            "predict": 3.0,
            "uwb": 4.0,
            "uwb_init": 5.0,
            "slam": 6.0,
            "slam_init": 7.0,
            "imu": 8.0,
            "imu_seed": 9.0,
        }
        return source_map.get(source, -1.0)

    def stamp_to_seconds(self, sec: int, nanosec: int) -> float:
        return float(sec) + float(nanosec) * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EkfFusionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
