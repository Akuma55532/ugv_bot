import math
import random
from typing import Any, Dict, List, Optional, Tuple

import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
import rclpy


class UwbRangeNode(Node):
    def __init__(self) -> None:
        super().__init__("uwb_range_node")

        default_pose_file = self._resolve_default_pose_file()

        self.declare_parameter("ground_truth_topic", "/ground_truth/odom")
        self.declare_parameter("uwb_pose_file", default_pose_file)
        self.declare_parameter("noise_stddev", 0.1)
        self.declare_parameter("use_3d_distance", True)
        self.declare_parameter("min_range", 0.0)
        self.declare_parameter("random_seed", -1)
        self.declare_parameter("publish_pose_topic", "/uwb/pose")
        self.declare_parameter("publish_pose", True)
        self.declare_parameter("pose_frame", "")
        self.declare_parameter("default_pose_z", 0.0)
        self.declare_parameter("position_covariance_xy", 0.01)
        self.declare_parameter("position_covariance_z", 9999.0)
        self.declare_parameter("orientation_covariance_rpy", 9999.0)

        self.ground_truth_topic = (
            self.get_parameter("ground_truth_topic").get_parameter_value().string_value
        )
        self.uwb_pose_file = (
            self.get_parameter("uwb_pose_file").get_parameter_value().string_value
        )
        if not self.uwb_pose_file:
            raise RuntimeError(
                "Parameter 'uwb_pose_file' is empty. "
                "Please provide a valid UWB anchor configuration YAML path."
            )

        self.noise_stddev = (
            self.get_parameter("noise_stddev").get_parameter_value().double_value
        )
        self.use_3d_distance = (
            self.get_parameter("use_3d_distance").get_parameter_value().bool_value
        )
        self.min_range = (
            self.get_parameter("min_range").get_parameter_value().double_value
        )
        self.publish_pose_topic = (
            self.get_parameter("publish_pose_topic")
            .get_parameter_value()
            .string_value
        )
        self.publish_pose = (
            self.get_parameter("publish_pose").get_parameter_value().bool_value
        )
        self.pose_frame = (
            self.get_parameter("pose_frame").get_parameter_value().string_value
        )
        self.default_pose_z = (
            self.get_parameter("default_pose_z").get_parameter_value().double_value
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

        seed = self.get_parameter("random_seed").get_parameter_value().integer_value
        self.random_generator = random.Random()
        if seed >= 0:
            self.random_generator.seed(seed)

        config = self._load_config(self.uwb_pose_file)
        self.anchor_poses = self._parse_anchor_poses(config)
        self.anchor_names: List[str] = list(self.anchor_poses.keys())
        if not self.anchor_names:
            raise RuntimeError(
                f"No UWB anchors found in config file: {self.uwb_pose_file}"
            )
        self.zone_disturbances = self._parse_zone_disturbances(config)

        self.latest_robot_pose: Optional[Tuple[float, float, float]] = None
        self.latest_truth_ranges: Dict[str, float] = {}
        self.latest_noisy_ranges: Dict[str, float] = {}
        self.latest_anchor_status: Dict[str, Dict[str, Any]] = {}
        self.latest_estimated_pose: Optional[Dict[str, Any]] = None
        self.latest_measurement_ready = False
        self.pose_publisher = None
        if self.publish_pose:
            self.pose_publisher = self.create_publisher(
                PoseWithCovarianceStamped, self.publish_pose_topic, 10
            )

        self.odom_subscription = self.create_subscription(
            Odometry, self.ground_truth_topic, self.odom_callback, 10
        )

        self.get_logger().info(
            "UWB range node started. "
            f"ground_truth_topic={self.ground_truth_topic}, "
            f"uwb_pose_file={self.uwb_pose_file}, "
            f"noise_stddev={self.noise_stddev:.3f} m, "
            f"use_3d_distance={self.use_3d_distance}, "
            f"publish_pose={self.publish_pose}, "
            f"zone_disturbances={len(self.zone_disturbances)}, "
            f"anchors={self.anchor_names}"
        )

    def _resolve_default_pose_file(self) -> str:
        try:
            return (
                get_package_share_directory("adaptive_fusion_gazebo")
                + "/config/uwb_pose.yaml"
            )
        except Exception:
            return ""

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        with open(config_file, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def _parse_anchor_poses(self, config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        anchors = config.get("uwb_anchors", {})
        parsed_anchors: Dict[str, Dict[str, float]] = {}
        for anchor_name, anchor_pose in anchors.items():
            parsed_anchors[anchor_name] = {
                "x": float(anchor_pose.get("x", 0.0)),
                "y": float(anchor_pose.get("y", 0.0)),
                "z": float(anchor_pose.get("z", 0.0)),
            }
        return parsed_anchors

    def _parse_zone_disturbances(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        zones = config.get("disturbance_zones", [])
        parsed_zones: List[Dict[str, Any]] = []
        for zone in zones:
            parsed_zones.append(
                {
                    "name": str(zone.get("name", "unnamed_zone")),
                    "enabled": bool(zone.get("enabled", True)),
                    "min_x": float(zone.get("min_x", float("-inf"))),
                    "max_x": float(zone.get("max_x", float("inf"))),
                    "min_y": float(zone.get("min_y", float("-inf"))),
                    "max_y": float(zone.get("max_y", float("inf"))),
                    "affected_anchors": list(zone.get("affected_anchors", [])),
                    "extra_noise_stddev": float(zone.get("extra_noise_stddev", 0.0)),
                    "bias_mean": float(zone.get("bias_mean", 0.0)),
                    "bias_stddev": float(zone.get("bias_stddev", 0.0)),
                    "dropout_prob": float(zone.get("dropout_prob", 0.0)),
                }
            )
        return parsed_zones

    def odom_callback(self, msg: Odometry) -> None:
        robot_x = msg.pose.pose.position.x
        robot_y = msg.pose.pose.position.y
        robot_z = msg.pose.pose.position.z
        self.latest_robot_pose = (robot_x, robot_y, robot_z)

        truth_ranges, noisy_ranges, anchor_status = self.compute_ranges_from_pose(
            robot_x, robot_y, robot_z
        )
        self.latest_truth_ranges = truth_ranges
        self.latest_noisy_ranges = noisy_ranges
        self.latest_anchor_status = anchor_status
        self.latest_estimated_pose = self.solve_position_from_ranges(noisy_ranges)
        self.latest_measurement_ready = True
        if self.publish_pose and self.pose_publisher is not None:
            pose_msg = self._build_pose_message(msg)
            if pose_msg is not None:
                self.pose_publisher.publish(pose_msg)

    def compute_ranges_from_pose(
        self, robot_x: float, robot_y: float, robot_z: float
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, Any]]]:
        truth_ranges: Dict[str, float] = {}
        noisy_ranges: Dict[str, float] = {}
        anchor_status: Dict[str, Dict[str, Any]] = {}

        for anchor_name in self.anchor_names:
            anchor_pose = self.anchor_poses[anchor_name]
            truth_range = self._compute_range(
                robot_x,
                robot_y,
                robot_z,
                anchor_pose["x"],
                anchor_pose["y"],
                anchor_pose["z"],
            )
            noisy_range, status = self._apply_disturbance(
                anchor_name, truth_range, robot_x, robot_y
            )
            truth_ranges[anchor_name] = truth_range
            anchor_status[anchor_name] = status
            if noisy_range is not None:
                noisy_ranges[anchor_name] = noisy_range

        return truth_ranges, noisy_ranges, anchor_status

    def get_latest_noisy_ranges(self) -> Dict[str, float]:
        return dict(self.latest_noisy_ranges)

    def get_latest_truth_ranges(self) -> Dict[str, float]:
        return dict(self.latest_truth_ranges)

    def get_latest_anchor_status(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.latest_anchor_status)

    def get_latest_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        return self.latest_robot_pose

    def get_latest_estimated_pose(self) -> Optional[Dict[str, Any]]:
        if self.latest_estimated_pose is None:
            return None
        return dict(self.latest_estimated_pose)

    def is_measurement_ready(self) -> bool:
        return self.latest_measurement_ready

    def get_anchor_names(self) -> List[str]:
        return list(self.anchor_names)

    def solve_position_from_ranges(
        self, range_measurements: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        valid_measurements = [
            (
                anchor_name,
                self.anchor_poses[anchor_name]["x"],
                self.anchor_poses[anchor_name]["y"],
                range_value,
            )
            for anchor_name, range_value in range_measurements.items()
            if anchor_name in self.anchor_poses
        ]

        if len(valid_measurements) < 3:
            return None

        ref_name, ref_x, ref_y, ref_range = valid_measurements[0]
        normal_a00 = 0.0
        normal_a01 = 0.0
        normal_a11 = 0.0
        normal_b0 = 0.0
        normal_b1 = 0.0

        for _, anchor_x, anchor_y, anchor_range in valid_measurements[1:]:
            a0 = 2.0 * (anchor_x - ref_x)
            a1 = 2.0 * (anchor_y - ref_y)
            b = (
                ref_range * ref_range
                - anchor_range * anchor_range
                - ref_x * ref_x
                + anchor_x * anchor_x
                - ref_y * ref_y
                + anchor_y * anchor_y
            )
            normal_a00 += a0 * a0
            normal_a01 += a0 * a1
            normal_a11 += a1 * a1
            normal_b0 += a0 * b
            normal_b1 += a1 * b

        determinant = normal_a00 * normal_a11 - normal_a01 * normal_a01
        if abs(determinant) < 1e-9:
            self.get_logger().warning(
                "Trilateration failed because the anchor geometry is singular."
            )
            return None

        position_x = (
            normal_b0 * normal_a11 - normal_b1 * normal_a01
        ) / determinant
        position_y = (
            normal_a00 * normal_b1 - normal_a01 * normal_b0
        ) / determinant

        return {
            "x": position_x,
            "y": position_y,
            "z": self.default_pose_z,
            "anchor_count": float(len(valid_measurements)),
            "reference_anchor": ref_name,
        }

    def _compute_range(
        self,
        robot_x: float,
        robot_y: float,
        robot_z: float,
        anchor_x: float,
        anchor_y: float,
        anchor_z: float,
    ) -> float:
        dx = robot_x - anchor_x
        dy = robot_y - anchor_y
        dz = robot_z - anchor_z

        if self.use_3d_distance:
            return math.sqrt(dx * dx + dy * dy + dz * dz)

        return math.sqrt(dx * dx + dy * dy)

    def _apply_disturbance(
        self, anchor_name: str, truth_range: float, robot_x: float, robot_y: float
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        active_zone_names: List[str] = []
        total_bias = 0.0
        extra_noise_stddev = 0.0
        dropout_prob = 0.0

        for zone in self.zone_disturbances:
            if not zone["enabled"]:
                continue
            if zone["affected_anchors"] and anchor_name not in zone["affected_anchors"]:
                continue
            if not (
                zone["min_x"] <= robot_x <= zone["max_x"]
                and zone["min_y"] <= robot_y <= zone["max_y"]
            ):
                continue

            active_zone_names.append(zone["name"])
            total_bias += zone["bias_mean"] + self.random_generator.gauss(
                0.0, zone["bias_stddev"]
            )
            extra_noise_stddev += zone["extra_noise_stddev"]
            dropout_prob = max(dropout_prob, zone["dropout_prob"])

        if dropout_prob > 0.0 and self.random_generator.random() < dropout_prob:
            return None, {
                "valid": False,
                "truth_range": truth_range,
                "active_zones": active_zone_names,
                "dropout": True,
                "bias": total_bias,
                "noise_stddev": self.noise_stddev + extra_noise_stddev,
            }

        total_noise_stddev = self.noise_stddev + extra_noise_stddev
        noisy_range = max(
            self.min_range,
            truth_range + total_bias + self.random_generator.gauss(0.0, total_noise_stddev),
        )
        return noisy_range, {
            "valid": True,
            "truth_range": truth_range,
            "active_zones": active_zone_names,
            "dropout": False,
            "bias": total_bias,
            "noise_stddev": total_noise_stddev,
            "measured_range": noisy_range,
        }

    def _build_pose_message(
        self, odom_msg: Odometry
    ) -> Optional[PoseWithCovarianceStamped]:
        if self.latest_estimated_pose is None:
            return None

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = odom_msg.header.stamp
        pose_msg.header.frame_id = self.pose_frame or odom_msg.header.frame_id or "world"
        pose_msg.pose.pose.position.x = self.latest_estimated_pose["x"]
        pose_msg.pose.pose.position.y = self.latest_estimated_pose["y"]
        pose_msg.pose.pose.position.z = self.latest_estimated_pose["z"]
        pose_msg.pose.pose.orientation.w = 1.0
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
    node = UwbRangeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
