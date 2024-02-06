#!/usr/bin/env python3

import math
import threading

import cv2

import rospy
from cv_bridge import CvBridge

from tf.transformations import euler_from_quaternion

from sensor_msgs.msg import Image
from daemon_ros_client.msg import MotorStatus


class HeadRollImageRotationNode:
    def __init__(self):
        import math
        self._roll_angle_lock = threading.Lock()
        self._roll_angle_rad = 0.0

        self._cv_bridge = CvBridge()

        self._image_pub = rospy.Publisher('output_image', Image, queue_size=1)

        self._motor_status_sub = rospy.Subscriber('daemon/motor_status', MotorStatus, self._motor_status_cb, queue_size=1)
        self._image_sub = rospy.Subscriber('input_image', Image, self._image_cb, queue_size=1)

    def _motor_status_cb(self, msg):
        head_angles = euler_from_quaternion([msg.head_pose.orientation.x,
                                             msg.head_pose.orientation.y,
                                             msg.head_pose.orientation.z,
                                             msg.head_pose.orientation.w])
        with self._roll_angle_lock:
            self._roll_angle_rad = head_angles[0]

    def _image_cb(self, color_image_msg):
        color_image = self._cv_bridge.imgmsg_to_cv2(color_image_msg, 'bgr8')

        with self._roll_angle_lock:
            roll_angle_rad = self._roll_angle_rad

        h, w, _ = color_image.shape
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -math.degrees(roll_angle_rad), 1.0)
        rotated_color_image = cv2.warpAffine(color_image, rotation_matrix, (w, h))

        rotated_color_image_msg = self._cv_bridge.cv2_to_imgmsg(rotated_color_image, 'bgr8')
        rotated_color_image_msg.header.seq = color_image_msg.header.seq
        rotated_color_image_msg.header.stamp = color_image_msg.header.stamp
        self._image_pub.publish(rotated_color_image_msg)

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('head_roll_image_rotation_node')
    head_roll_image_rotation_node = HeadRollImageRotationNode()
    head_roll_image_rotation_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
