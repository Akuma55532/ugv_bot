import math
from collections import deque
from typing import Deque, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return sum((value - avg) * (value - avg) for value in values) / float(len(values))


class SlamPoseAdapterNode(Node):
    def __init__(self) -> None:
        super().__init__("slam_pose_adapter_node")

        self.declare_parameter("input_mode", "tracked_pose")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_footprint")
        self.declare_parameter("tracked_pose_topic", "/tracked_pose")
        self.declare_parameter("slam_pose_topic", "/slam/pose")
        self.declare_parameter("reference_odom_topic", "/odom")
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("position_covariance_xy", 0.05)
        self.declare_parameter("position_covariance_z", 9999.0)
        self.declare_parameter("roll_pitch_covariance", 9999.0)
        self.declare_parameter("yaw_covariance", 0.05)
        self.declare_parameter("adaptive_covariance", True)
        self.declare_parameter("consistency_window_size", 10)
        self.declare_parameter("consistency_min_score", 0.05)
        self.declare_parameter("consistency_max_scale", 20.0)
        self.declare_parameter("translation_residual_beta", 0.12)
        self.declare_parameter("yaw_residual_beta", 0.20)
        self.declare_parameter("jitter_translation_beta", 0.10)
        self.declare_parameter("jitter_yaw_beta", 0.15)
        self.declare_parameter("translation_residual_weight", 0.35)
        self.declare_parameter("yaw_residual_weight", 0.25)
        self.declare_parameter("jitter_translation_weight", 0.20)
        self.declare_parameter("jitter_yaw_weight", 0.20)
        self.declare_parameter("lookup_timeout_sec", 0.2)

        self.input_mode = (
            self.get_parameter("input_mode").get_parameter_value().string_value
        )
        self.map_frame = (
            self.get_parameter("map_frame").get_parameter_value().string_value
        )
        self.robot_frame = (
            self.get_parameter("robot_frame").get_parameter_value().string_value
        )
        self.tracked_pose_topic = (
            self.get_parameter("tracked_pose_topic").get_parameter_value().string_value
        )
        self.slam_pose_topic = (
            self.get_parameter("slam_pose_topic").get_parameter_value().string_value
        )
        self.reference_odom_topic = (
            self.get_parameter("reference_odom_topic")
            .get_parameter_value()
            .string_value
        )
        self.publish_rate = (
            self.get_parameter("publish_rate").get_parameter_value().double_value
        )
        self.position_covariance_xy = (
            self.get_parameter("position_covariance_xy")
            .get_parameter_value()
            .double_value
        )
        self.position_covariance_z = (
            self.get_parameter("position_covariance_z")
            .get_parameter_value()
            .double_value
        )
        self.roll_pitch_covariance = (
            self.get_parameter("roll_pitch_covariance")
            .get_parameter_value()
            .double_value
        )
        self.yaw_covariance = (
            self.get_parameter("yaw_covariance").get_parameter_value().double_value
        )
        self.adaptive_covariance = (
            self.get_parameter("adaptive_covariance").get_parameter_value().bool_value
        )
        self.consistency_min_score = (
            self.get_parameter("consistency_min_score")
            .get_parameter_value()
            .double_value
        )
        self.consistency_max_scale = (
            self.get_parameter("consistency_max_scale")
            .get_parameter_value()
            .double_value
        )
        self.translation_residual_beta = (
            self.get_parameter("translation_residual_beta")
            .get_parameter_value()
            .double_value
        )
        self.yaw_residual_beta = (
            self.get_parameter("yaw_residual_beta").get_parameter_value().double_value
        )
        self.jitter_translation_beta = (
            self.get_parameter("jitter_translation_beta")
            .get_parameter_value()
            .double_value
        )
        self.jitter_yaw_beta = (
            self.get_parameter("jitter_yaw_beta").get_parameter_value().double_value
        )
        self.translation_residual_weight = (
            self.get_parameter("translation_residual_weight")
            .get_parameter_value()
            .double_value
        )
        self.yaw_residual_weight = (
            self.get_parameter("yaw_residual_weight")
            .get_parameter_value()
            .double_value
        )
        self.jitter_translation_weight = (
            self.get_parameter("jitter_translation_weight")
            .get_parameter_value()
            .double_value
        )
        self.jitter_yaw_weight = (
            self.get_parameter("jitter_yaw_weight")
            .get_parameter_value()
            .double_value
        )
        self.lookup_timeout = Duration(
            seconds=self.get_parameter("lookup_timeout_sec")
            .get_parameter_value()
            .double_value
        )

        window_size = max(
            2,
            self.get_parameter("consistency_window_size")
            .get_parameter_value()
            .integer_value,
        )
        self.recent_translation_steps: Deque[float] = deque(maxlen=window_size)
        self.recent_yaw_steps: Deque[float] = deque(maxlen=window_size)

        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, self.slam_pose_topic, 10
        )
        self.odom_subscription = self.create_subscription(
            Odometry, self.reference_odom_topic, self.odom_callback, 20
        )

        self.last_warning: Optional[str] = None
        self.tf_buffer = None
        self.tf_listener = None
        self.tracked_pose_subscription = None
        self.timer = None

        self.latest_odom_pose: Optional[Tuple[float, float, float]] = None
        self.previous_odom_pose_for_slam: Optional[Tuple[float, float, float]] = None
        self.previous_slam_pose: Optional[Tuple[float, float, float]] = None
        self.latest_position_covariance_xy = self.position_covariance_xy
        self.latest_yaw_covariance = self.yaw_covariance
        self.latest_consistency_score = 1.0

        if self.input_mode == "tracked_pose":
            self.tracked_pose_subscription = self.create_subscription(
                PoseStamped, self.tracked_pose_topic, self.tracked_pose_callback, 10
            )
        else:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            timer_period = 1.0 / max(self.publish_rate, 1e-3)
            self.timer = self.create_timer(timer_period, self.publish_pose_from_tf)

        self.get_logger().info(
            "SLAM pose adapter started. "
            f"input_mode={self.input_mode}, map_frame={self.map_frame}, "
            f"robot_frame={self.robot_frame}, tracked_pose_topic={self.tracked_pose_topic}, "
            f"reference_odom_topic={self.reference_odom_topic}, "
            f"slam_pose_topic={self.slam_pose_topic}, publish_rate={self.publish_rate:.2f} Hz, "
            f"adaptive_covariance={self.adaptive_covariance}"
        )

    def odom_callback(self, msg: Odometry) -> None:
        yaw = quaternion_to_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        self.latest_odom_pose = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            yaw,
        )

    def tracked_pose_callback(self, msg: PoseStamped) -> None:
        pose_msg = self.build_pose_message(
            stamp=msg.header.stamp,
            frame_id=msg.header.frame_id or self.map_frame,
            position_x=msg.pose.position.x,
            position_y=msg.pose.position.y,
            position_z=msg.pose.position.z,
            orientation=msg.pose.orientation,
        )
        self.pose_publisher.publish(pose_msg)

    def publish_pose_from_tf(self) -> None:
        if self.tf_buffer is None:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=self.lookup_timeout,
            )
        except TransformException as error:
            warning_text = (
                f"Waiting for transform {self.map_frame} -> {self.robot_frame}: {error}"
            )
            if warning_text != self.last_warning:
                self.get_logger().warning(warning_text)
                self.last_warning = warning_text
            return

        self.last_warning = None
        pose_msg = self.build_pose_message(
            stamp=transform.header.stamp,
            frame_id=self.map_frame,
            position_x=transform.transform.translation.x,
            position_y=transform.transform.translation.y,
            position_z=transform.transform.translation.z,
            orientation=transform.transform.rotation,
        )
        self.pose_publisher.publish(pose_msg)

    def evaluate_pose_quality(
        self,
        position_x: float,
        position_y: float,
        yaw: float,
    ) -> Tuple[float, float, float]:
        if not self.adaptive_covariance:
            return 1.0, self.position_covariance_xy, self.yaw_covariance

        current_pose = (position_x, position_y, yaw)

        translation_residual = 0.0
        yaw_residual = 0.0
        translation_step = 0.0
        yaw_step = 0.0

        if self.previous_slam_pose is not None:
            translation_step = math.hypot(
                current_pose[0] - self.previous_slam_pose[0],
                current_pose[1] - self.previous_slam_pose[1],
            )
            yaw_step = abs(normalize_angle(current_pose[2] - self.previous_slam_pose[2]))

        if (
            self.previous_slam_pose is not None
            and self.latest_odom_pose is not None
            and self.previous_odom_pose_for_slam is not None
        ):
            odom_translation = math.hypot(
                self.latest_odom_pose[0] - self.previous_odom_pose_for_slam[0],
                self.latest_odom_pose[1] - self.previous_odom_pose_for_slam[1],
            )
            odom_yaw = abs(
                normalize_angle(
                    self.latest_odom_pose[2] - self.previous_odom_pose_for_slam[2]
                )
            )
            translation_residual = abs(translation_step - odom_translation)
            yaw_residual = abs(yaw_step - odom_yaw)

        translation_history = list(self.recent_translation_steps)
        yaw_history = list(self.recent_yaw_steps)
        translation_history.append(translation_step)
        yaw_history.append(yaw_step)

        translation_jitter = math.sqrt(variance(translation_history))
        yaw_jitter = math.sqrt(variance(yaw_history))

        translation_score = math.exp(
            -(translation_residual * translation_residual)
            / max(self.translation_residual_beta * self.translation_residual_beta, 1e-6)
        )
        yaw_score = math.exp(
            -(yaw_residual * yaw_residual)
            / max(self.yaw_residual_beta * self.yaw_residual_beta, 1e-6)
        )
        translation_jitter_score = math.exp(
            -(translation_jitter * translation_jitter)
            / max(self.jitter_translation_beta * self.jitter_translation_beta, 1e-6)
        )
        yaw_jitter_score = math.exp(
            -(yaw_jitter * yaw_jitter)
            / max(self.jitter_yaw_beta * self.jitter_yaw_beta, 1e-6)
        )

        weight_sum = (
            self.translation_residual_weight
            + self.yaw_residual_weight
            + self.jitter_translation_weight
            + self.jitter_yaw_weight
        )
        if weight_sum <= 1e-9:
            weight_sum = 1.0

        consistency_score = (
            self.translation_residual_weight * translation_score
            + self.yaw_residual_weight * yaw_score
            + self.jitter_translation_weight * translation_jitter_score
            + self.jitter_yaw_weight * yaw_jitter_score
        ) / weight_sum
        consistency_score = max(
            self.consistency_min_score, min(1.0, consistency_score)
        )

        covariance_scale = min(
            self.consistency_max_scale,
            max(1.0, 1.0 / max(consistency_score, 1e-6)),
        )
        position_covariance_xy = self.position_covariance_xy * covariance_scale
        yaw_covariance = self.yaw_covariance * covariance_scale

        self.latest_consistency_score = consistency_score
        self.latest_position_covariance_xy = position_covariance_xy
        self.latest_yaw_covariance = yaw_covariance
        self.previous_slam_pose = current_pose
        if self.latest_odom_pose is not None:
            self.previous_odom_pose_for_slam = self.latest_odom_pose
        self.recent_translation_steps.append(translation_step)
        self.recent_yaw_steps.append(yaw_step)
        return consistency_score, position_covariance_xy, yaw_covariance

    def build_pose_message(
        self,
        stamp,
        frame_id: str,
        position_x: float,
        position_y: float,
        position_z: float,
        orientation,
    ) -> PoseWithCovarianceStamped:
        yaw = quaternion_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        _, position_covariance_xy, yaw_covariance = self.evaluate_pose_quality(
            position_x, position_y, yaw
        )

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.pose.position.x = position_x
        pose_msg.pose.pose.position.y = position_y
        pose_msg.pose.pose.position.z = position_z
        pose_msg.pose.pose.orientation = orientation

        covariance = [0.0] * 36
        covariance[0] = position_covariance_xy
        covariance[7] = position_covariance_xy
        covariance[14] = self.position_covariance_z
        covariance[21] = self.roll_pitch_covariance
        covariance[28] = self.roll_pitch_covariance
        covariance[35] = yaw_covariance
        pose_msg.pose.covariance = covariance
        return pose_msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SlamPoseAdapterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
