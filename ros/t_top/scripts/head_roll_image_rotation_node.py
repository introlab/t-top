#!/usr/bin/env python3

import math

import cv2

import rclpy
import rclpy.node


from cv_bridge import CvBridge

from tf_transformations import euler_from_quaternion

from sensor_msgs.msg import Image
from daemon_ros_client.msg import MotorStatus


class HeadRollImageRotationNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('head_roll_image_rotation_node')
        self._roll_angle_rad = 0.0
        self._cv_bridge = CvBridge()

        self._image_pub = self.create_publisher(Image, 'output_image', 1)

        self._motor_status_sub = self.create_subscription(MotorStatus, 'daemon/motor_status', self._motor_status_cb, 1)
        self._image_sub = self.create_subscription(Image, 'input_image', self._image_cb, 1)

    def _motor_status_cb(self, msg):
        head_angles = euler_from_quaternion([msg.head_pose.orientation.x,
                                             msg.head_pose.orientation.y,
                                             msg.head_pose.orientation.z,
                                             msg.head_pose.orientation.w])
        self._roll_angle_rad = head_angles[0]

    def _image_cb(self, color_image_msg):
        color_image = self._cv_bridge.imgmsg_to_cv2(color_image_msg, 'bgr8')

        h, w, _ = color_image.shape
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -math.degrees(self._roll_angle_rad), 1.0)
        rotated_color_image = cv2.warpAffine(color_image, rotation_matrix, (w, h))

        rotated_color_image_msg = self._cv_bridge.cv2_to_imgmsg(rotated_color_image, 'bgr8')
        rotated_color_image_msg.header.stamp = color_image_msg.header.stamp
        self._image_pub.publish(rotated_color_image_msg)

    def run(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    head_roll_image_rotation_node = HeadRollImageRotationNode()

    try:
        head_roll_image_rotation_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        head_roll_image_rotation_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
