#!/usr/bin/env python3
import numpy as np
import cv2

import rclpy
import rclpy.node

from cv_bridge import CvBridge

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

import hbba_lite


class TooCloseReactionNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('too_close_reaction_node')

        self._max_offset_m = self.declare_parameter('max_offset_m', 0.01).get_parameter_value().double_value
        self._too_close_start_distance_m = self.declare_parameter('too_close_start_distance_m', 0.5).get_parameter_value().double_value
        self._too_close_end_distance_m = self.declare_parameter('too_close_end_distance_m', 0.25).get_parameter_value().double_value
        self._pixel_ratio = self.declare_parameter('pixel_ratio', 0.01).get_parameter_value().double_value

        self._cv_bridge = CvBridge()

        self._current_offset_m = 0.0

        self._head_pose_pub = self.create_publisher(PoseStamped, 'too_close_reaction/set_head_pose', 1)
        self._depth_image_sub = hbba_lite.OnOffHbbaSubscriber(self, Image, 'depth_image_raw', self._image_cb, 1)
        self._depth_image_sub.on_filter_state_changed(self._hbba_filter_state_cb)

    def _image_cb(self, msg):
        if msg.encoding != '16UC1':
            self.get_logger().error('Invalid depth image encoding')

        depth_image = self._cv_bridge.imgmsg_to_cv2(msg, '16UC1')
        distance_m = self._compute_distance(depth_image) - self._current_offset_m

        if distance_m > self._too_close_start_distance_m:
            self._current_offset_m = 0.0
        else:
            ratio = 1.0 - max(0.0, distance_m - self._too_close_end_distance_m) / (self._too_close_start_distance_m - self._too_close_end_distance_m)
            self._current_offset_m = self._max_offset_m * ratio

        self._send_pose(self._current_offset_m)

    def _hbba_filter_state_cb(self, previous_is_filtering_all_messages, new_is_filtering_all_messages):
        if not previous_is_filtering_all_messages and new_is_filtering_all_messages:
            self._current_offset_m = 0.0
            self._send_pose(self._current_offset_m)

    def _compute_distance(self, img):
        iu16 = np.iinfo(np.uint16)
        img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5, interpolation = cv2.INTER_NEAREST)

        hist = cv2.calcHist([img], [0], None, [iu16.max + 1], [iu16.min, iu16.max])
        hist = hist.astype(np.float32)
        hist /= img.size
        hist[0] = 0.0 # Remove invalid depth

        cum_hist = np.cumsum(hist)

        return np.argmax(cum_hist > self._pixel_ratio) / 1000

    def _send_pose(self, offset_m):
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = 'stewart_base'

        pose_msg.pose.position.x = float(-offset_m)
        pose_msg.pose.position.y = 0.0
        pose_msg.pose.position.z = 0.0

        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0

        self._head_pose_pub.publish(pose_msg)

    def run(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    too_close_reaction_node = TooCloseReactionNode()

    try:
        too_close_reaction_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        too_close_reaction_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
