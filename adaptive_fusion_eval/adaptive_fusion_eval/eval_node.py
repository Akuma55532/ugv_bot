import csv
import math
import os
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class Pose2DState:
    stamp_sec: float
    x: float
    y: float
    yaw: float


@dataclass
class AlignmentState:
    gt: Pose2DState
    fusion: Pose2DState
    yaw_offset: float


class EvalNode(Node):
    def __init__(self) -> None:
        super().__init__("adaptive_fusion_eval_node")

        self.declare_parameter("ground_truth_topic", "/ground_truth/odom")
        self.declare_parameter("fusion_pose_topic", "/fusion/pose")
        self.declare_parameter("debug_topic", "/fusion/debug")
        self.declare_parameter("method_name", "unknown")
        self.declare_parameter("output_dir", "/tmp/adaptive_fusion_eval")
        self.declare_parameter("summary_filename", "summary.csv")
        self.declare_parameter("timeseries_filename", "timeseries.csv")
        self.declare_parameter("flush_period_sec", 2.0)
        self.declare_parameter("status_period_sec", 2.0)
        self.declare_parameter("sync_tolerance_sec", 0.20)
        self.declare_parameter("alignment_mode", "initial_pose")
        self.declare_parameter("eval_start_delay_sec", 0.0)
        self.declare_parameter("eval_duration_sec", 0.0)

        self.ground_truth_topic = (
            self.get_parameter("ground_truth_topic").get_parameter_value().string_value
        )
        self.fusion_pose_topic = (
            self.get_parameter("fusion_pose_topic").get_parameter_value().string_value
        )
        self.debug_topic = (
            self.get_parameter("debug_topic").get_parameter_value().string_value
        )
        self.method_name = (
            self.get_parameter("method_name").get_parameter_value().string_value
        )
        self.output_dir = (
            self.get_parameter("output_dir").get_parameter_value().string_value
        )
        self.summary_filename = (
            self.get_parameter("summary_filename").get_parameter_value().string_value
        )
        self.timeseries_filename = (
            self.get_parameter("timeseries_filename").get_parameter_value().string_value
        )
        self.flush_period_sec = (
            self.get_parameter("flush_period_sec").get_parameter_value().double_value
        )
        self.status_period_sec = (
            self.get_parameter("status_period_sec").get_parameter_value().double_value
        )
        self.sync_tolerance_sec = (
            self.get_parameter("sync_tolerance_sec").get_parameter_value().double_value
        )
        self.alignment_mode = (
            self.get_parameter("alignment_mode").get_parameter_value().string_value
        )
        self.eval_start_delay_sec = (
            self.get_parameter("eval_start_delay_sec").get_parameter_value().double_value
        )
        self.eval_duration_sec = (
            self.get_parameter("eval_duration_sec").get_parameter_value().double_value
        )

        os.makedirs(self.output_dir, exist_ok=True)
        self.summary_path = os.path.join(
            self.output_dir, f"{self.method_name}_{self.summary_filename}"
        )
        self.timeseries_path = os.path.join(
            self.output_dir, f"{self.method_name}_{self.timeseries_filename}"
        )

        self.latest_ground_truth: Optional[Pose2DState] = None
        self.latest_debug: Optional[list[float]] = None
        self.latest_debug_stamp: Optional[float] = None
        self.first_eval_stamp: Optional[float] = None
        self.alignment: Optional[AlignmentState] = None

        self.sample_count = 0
        self.sum_position_error = 0.0
        self.sum_squared_position_error = 0.0
        self.max_position_error = 0.0
        self.sum_yaw_error = 0.0
        self.sum_squared_yaw_error = 0.0
        self.max_yaw_error = 0.0

        self.timeseries_header_written = os.path.exists(self.timeseries_path) and os.path.getsize(
            self.timeseries_path
        ) > 0

        self.gt_subscription = self.create_subscription(
            Odometry, self.ground_truth_topic, self.ground_truth_callback, 20
        )
        self.fusion_subscription = self.create_subscription(
            PoseWithCovarianceStamped, self.fusion_pose_topic, self.fusion_pose_callback, 20
        )
        self.debug_subscription = self.create_subscription(
            Float64MultiArray, self.debug_topic, self.debug_callback, 20
        )

        self.status_timer = self.create_timer(
            max(self.status_period_sec, 0.2), self.report_status
        )
        self.flush_timer = self.create_timer(
            max(self.flush_period_sec, 0.5), self.flush_summary
        )

        self.get_logger().info(
            "Adaptive fusion eval node started. "
            f"method_name={self.method_name}, output_dir={self.output_dir}, "
            f"ground_truth_topic={self.ground_truth_topic}, "
            f"fusion_pose_topic={self.fusion_pose_topic}, debug_topic={self.debug_topic}, "
            f"alignment_mode={self.alignment_mode}, "
            f"eval_start_delay_sec={self.eval_start_delay_sec:.2f}, "
            f"eval_duration_sec={self.eval_duration_sec:.2f}"
        )

    def ground_truth_callback(self, msg: Odometry) -> None:
        self.latest_ground_truth = Pose2DState(
            stamp_sec=self.stamp_to_sec(msg.header.stamp.sec, msg.header.stamp.nanosec),
            x=float(msg.pose.pose.position.x),
            y=float(msg.pose.pose.position.y),
            yaw=quaternion_to_yaw(
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ),
        )

    def debug_callback(self, msg: Float64MultiArray) -> None:
        self.latest_debug = list(msg.data)
        self.latest_debug_stamp = self.get_clock().now().nanoseconds * 1e-9

    def fusion_pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        if self.latest_ground_truth is None:
            return

        fusion_stamp = self.stamp_to_sec(msg.header.stamp.sec, msg.header.stamp.nanosec)
        if abs(fusion_stamp - self.latest_ground_truth.stamp_sec) > self.sync_tolerance_sec:
            return
        if not self.is_inside_eval_window(fusion_stamp):
            return

        raw_fusion_x = float(msg.pose.pose.position.x)
        raw_fusion_y = float(msg.pose.pose.position.y)
        raw_fusion_yaw = quaternion_to_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        raw_fusion = Pose2DState(
            stamp_sec=fusion_stamp,
            x=raw_fusion_x,
            y=raw_fusion_y,
            yaw=raw_fusion_yaw,
        )
        fusion = self.align_fusion_pose(raw_fusion, self.latest_ground_truth)

        dx = fusion.x - self.latest_ground_truth.x
        dy = fusion.y - self.latest_ground_truth.y
        position_error = math.hypot(dx, dy)
        yaw_error = abs(normalize_angle(fusion.yaw - self.latest_ground_truth.yaw))

        self.sample_count += 1
        self.sum_position_error += position_error
        self.sum_squared_position_error += position_error * position_error
        self.max_position_error = max(self.max_position_error, position_error)
        self.sum_yaw_error += yaw_error
        self.sum_squared_yaw_error += yaw_error * yaw_error
        self.max_yaw_error = max(self.max_yaw_error, yaw_error)

        self.append_timeseries_row(
            fusion_stamp=fusion_stamp,
            gt=self.latest_ground_truth,
            fusion_x=fusion.x,
            fusion_y=fusion.y,
            fusion_yaw=fusion.yaw,
            raw_fusion=raw_fusion,
            position_error=position_error,
            yaw_error=yaw_error,
        )

    def is_inside_eval_window(self, stamp_sec: float) -> bool:
        if self.first_eval_stamp is None:
            self.first_eval_stamp = stamp_sec

        elapsed = stamp_sec - self.first_eval_stamp
        if elapsed < self.eval_start_delay_sec:
            return False
        if self.eval_duration_sec > 0.0 and elapsed > (
            self.eval_start_delay_sec + self.eval_duration_sec
        ):
            return False
        return True

    def align_fusion_pose(
        self,
        fusion: Pose2DState,
        ground_truth: Pose2DState,
    ) -> Pose2DState:
        if self.alignment_mode in ("none", "raw"):
            return fusion

        if self.alignment_mode != "initial_pose":
            self.get_logger().warning(
                f"Unknown alignment_mode={self.alignment_mode}; using raw poses."
            )
            self.alignment_mode = "raw"
            return fusion

        if self.alignment is None:
            yaw_offset = normalize_angle(ground_truth.yaw - fusion.yaw)
            self.alignment = AlignmentState(
                gt=ground_truth,
                fusion=fusion,
                yaw_offset=yaw_offset,
            )
            self.get_logger().info(
                "Initialized trajectory alignment: "
                f"gt=({ground_truth.x:.3f}, {ground_truth.y:.3f}, {ground_truth.yaw:.3f}), "
                f"fusion=({fusion.x:.3f}, {fusion.y:.3f}, {fusion.yaw:.3f}), "
                f"yaw_offset={yaw_offset:.3f} rad"
            )

        assert self.alignment is not None
        rel_x = fusion.x - self.alignment.fusion.x
        rel_y = fusion.y - self.alignment.fusion.y
        cos_yaw = math.cos(self.alignment.yaw_offset)
        sin_yaw = math.sin(self.alignment.yaw_offset)
        aligned_x = self.alignment.gt.x + cos_yaw * rel_x - sin_yaw * rel_y
        aligned_y = self.alignment.gt.y + sin_yaw * rel_x + cos_yaw * rel_y
        aligned_yaw = normalize_angle(fusion.yaw + self.alignment.yaw_offset)
        return Pose2DState(
            stamp_sec=fusion.stamp_sec,
            x=aligned_x,
            y=aligned_y,
            yaw=aligned_yaw,
        )

    def append_timeseries_row(
        self,
        fusion_stamp: float,
        gt: Pose2DState,
        fusion_x: float,
        fusion_y: float,
        fusion_yaw: float,
        raw_fusion: Pose2DState,
        position_error: float,
        yaw_error: float,
    ) -> None:
        debug = self.latest_debug or []
        row = [
            self.method_name,
            fusion_stamp,
            gt.x,
            gt.y,
            gt.yaw,
            fusion_x,
            fusion_y,
            fusion_yaw,
            position_error,
            yaw_error,
            raw_fusion.x,
            raw_fusion.y,
            raw_fusion.yaw,
        ]
        while len(debug) < 20:
            debug.append(float("nan"))
        row.extend(debug[:20])

        with open(self.timeseries_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            if not self.timeseries_header_written:
                writer.writerow(
                    [
                        "method",
                        "stamp_sec",
                        "gt_x",
                        "gt_y",
                        "gt_yaw",
                        "fusion_x",
                        "fusion_y",
                        "fusion_yaw",
                        "position_error",
                        "yaw_error",
                        "raw_fusion_x",
                        "raw_fusion_y",
                        "raw_fusion_yaw",
                        "debug_state_x",
                        "debug_state_y",
                        "debug_state_yaw",
                        "debug_cov_x",
                        "debug_cov_y",
                        "debug_cov_yaw",
                        "debug_vx",
                        "debug_wz",
                        "debug_predict_count",
                        "debug_uwb_update_count",
                        "debug_slam_update_count",
                        "debug_imu_update_count",
                        "debug_last_source",
                        "debug_init_source",
                        "debug_has_raw_odom",
                        "debug_has_map_odom_tf",
                        "debug_uwb_score",
                        "debug_uwb_scale",
                        "debug_slam_score",
                        "debug_slam_scale",
                    ]
                )
                self.timeseries_header_written = True
            writer.writerow(row)

    def report_status(self) -> None:
        if self.sample_count <= 0:
            self.get_logger().info("Waiting for synchronized fusion and ground-truth data.")
            return

        rmse = math.sqrt(self.sum_squared_position_error / float(self.sample_count))
        mean_error = self.sum_position_error / float(self.sample_count)
        yaw_rmse = math.sqrt(self.sum_squared_yaw_error / float(self.sample_count))
        self.get_logger().info(
            "Eval status: "
            f"method={self.method_name}, samples={self.sample_count}, "
            f"mean_pos_err={mean_error:.3f} m, rmse_pos={rmse:.3f} m, "
            f"max_pos_err={self.max_position_error:.3f} m, "
            f"rmse_yaw={yaw_rmse:.3f} rad"
        )

    def flush_summary(self) -> None:
        if self.sample_count <= 0:
            return

        rmse = math.sqrt(self.sum_squared_position_error / float(self.sample_count))
        mean_error = self.sum_position_error / float(self.sample_count)
        yaw_rmse = math.sqrt(self.sum_squared_yaw_error / float(self.sample_count))
        yaw_mean = self.sum_yaw_error / float(self.sample_count)

        with open(self.summary_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "method",
                    "sample_count",
                    "mean_position_error",
                    "rmse_position_error",
                    "max_position_error",
                    "mean_yaw_error",
                    "rmse_yaw_error",
                    "max_yaw_error",
                ]
            )
            writer.writerow(
                [
                    self.method_name,
                    self.sample_count,
                    mean_error,
                    rmse,
                    self.max_position_error,
                    yaw_mean,
                    yaw_rmse,
                    self.max_yaw_error,
                ]
            )

    def destroy_node(self):
        self.flush_summary()
        return super().destroy_node()

    def stamp_to_sec(self, sec: int, nanosec: int) -> float:
        return float(sec) + float(nanosec) * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EvalNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
