from typing import Optional

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformException, TransformListener


class SlamPoseAdapterNode(Node):
    def __init__(self) -> None:
        super().__init__("slam_pose_adapter_node")

        self.declare_parameter("input_mode", "tracked_pose")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_footprint")
        self.declare_parameter("tracked_pose_topic", "/tracked_pose")
        self.declare_parameter("slam_pose_topic", "/slam/pose")
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("position_covariance_xy", 0.05)
        self.declare_parameter("position_covariance_z", 9999.0)
        self.declare_parameter("orientation_covariance_rpy", 9999.0)
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
        self.orientation_covariance_rpy = (
            self.get_parameter("orientation_covariance_rpy")
            .get_parameter_value()
            .double_value
        )
        self.lookup_timeout = Duration(
            seconds=self.get_parameter("lookup_timeout_sec")
            .get_parameter_value()
            .double_value
        )

        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, self.slam_pose_topic, 10
        )
        self.last_warning: Optional[str] = None

        self.tf_buffer = None
        self.tf_listener = None
        self.tracked_pose_subscription = None
        self.timer = None

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
            f"slam_pose_topic={self.slam_pose_topic}, publish_rate={self.publish_rate:.2f} Hz"
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

    def build_pose_message(
        self,
        stamp,
        frame_id: str,
        position_x: float,
        position_y: float,
        position_z: float,
        orientation,
    ) -> PoseWithCovarianceStamped:
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.pose.position.x = position_x
        pose_msg.pose.pose.position.y = position_y
        pose_msg.pose.pose.position.z = position_z
        pose_msg.pose.pose.orientation = orientation

        covariance = [0.0] * 36
        covariance[0] = self.position_covariance_xy
        covariance[7] = self.position_covariance_xy
        covariance[14] = self.position_covariance_z
        covariance[21] = self.orientation_covariance_rpy
        covariance[28] = self.orientation_covariance_rpy
        covariance[35] = self.orientation_covariance_rpy
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
