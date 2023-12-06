#!/usr/bin/env python3
import numpy as np
import cv2

import rospy
from cv_bridge import CvBridge

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

import hbba_lite


class TooCloseReactionNode:
    def __init__(self):
        self._max_offset_m = rospy.get_param('~max_offset_m', 0.01)
        self._too_close_start_distance_m = rospy.get_param('~too_close_start_distance_m', 0.5)
        self._too_close_end_distance_m = rospy.get_param('~too_close_end_distance_m', 0.25)
        self._pixel_ratio = rospy.get_param('~pixel_ratio', 0.01)

        self._cv_bridge = CvBridge()

        self._current_offset_m = 0.0

        self._head_pose_pub = rospy.Publisher('too_close_reaction/set_head_pose', PoseStamped, queue_size=1)
        self._depth_image_sub = hbba_lite.OnOffHbbaSubscriber('depth_image_raw', Image, self._image_cb, queue_size=1)
        self._depth_image_sub.on_filter_state_changed(self._hbba_filter_state_cb)

    def _image_cb(self, msg):
        if msg.encoding != '16UC1':
            rospy.logerr('Invalid depth image encoding')

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

        pose_msg.pose.position.x = -offset_m
        pose_msg.pose.position.y = 0
        pose_msg.pose.position.z = 0

        pose_msg.pose.orientation.x = 0
        pose_msg.pose.orientation.y = 0
        pose_msg.pose.orientation.z = 0
        pose_msg.pose.orientation.w = 1

        self._head_pose_pub.publish(pose_msg)

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('too_close_reaction_node')
    too_close_reaction_node = TooCloseReactionNode()
    too_close_reaction_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
