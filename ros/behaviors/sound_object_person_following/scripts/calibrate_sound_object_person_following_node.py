#!/usr/bin/env python3

import threading
import math

import numpy as np
import cv2

import rospy
import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import Image

from sound_object_person_following import Camera3dCalibration


MIN_MATCH_COUNT = 10
LOWES_RATIO_THRESHOLD = 0.7
INACTIVE_SLEEP_DURATION = 0.1


class Match:
    def __init__(self, source_points, destination_points,
                 source_width, source_height,
                 destination_width, destination_height):
        self.source_points = source_points
        self.destination_points = destination_points
        self.source_width = source_width
        self.source_height = source_height
        self.destination_width = destination_width
        self.destination_height = destination_height


class CalibrateSoundObjectPersonFollowingNode:
    def __init__(self):
        self._match_count = rospy.get_param('~match_count')
        if self._match_count < 1:
            raise ValueError('match_count must be at least 1.')

        self._cv_bridge = CvBridge()
        self._sift = cv2.SIFT_create()
        self._matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

        self._matches_lock = threading.Lock()
        self._matches = []

        camera_3d_image_sub = message_filters.Subscriber('camera_3d/color/image_raw', Image)
        camera_2d_wide_image_sub = message_filters.Subscriber('camera_2d_wide/image_rect', Image)
        self._image_ts = message_filters.ApproximateTimeSynchronizer([camera_3d_image_sub, camera_2d_wide_image_sub], 10, 0.1)
        self._image_ts.registerCallback(self._image_cb)

    def _image_cb(self, camera_3d_image_msg, camera_2d_wide_image_msg):
        camera_3d_image = self._cv_bridge.imgmsg_to_cv2(camera_3d_image_msg, 'bgr8')
        camera_2d_wide_image = self._cv_bridge.imgmsg_to_cv2(camera_2d_wide_image_msg, 'bgr8')

        match = self._match_images(camera_3d_image, camera_2d_wide_image)
        if match is not None:
            with self._matches_lock:
                self._matches.append(match)

    def _match_images(self, source_image, destination_image):
        source_keypoints, source_descriptors = self._sift.detectAndCompute(source_image, None)
        destination_keypoints, destination_descriptors = self._sift.detectAndCompute(destination_image, None)

        matches = self._matcher.knnMatch(source_descriptors, destination_descriptors, 2)
        good_matches = []
        for m,n in matches:
            if m.distance < LOWES_RATIO_THRESHOLD * n.distance:
                good_matches.append(m)

        if len(good_matches) < MIN_MATCH_COUNT:
            rospy.logwarn('Not enough SIFT feature matches')
            return None
        else:
            source_points = np.float32([source_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            destination_points = np.float32([destination_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        return Match(source_points, destination_points)

    def run(self):
        while not rospy.is_shutdown():
            with self._matches_lock:
                if len(self._matches) == self._match_count:
                    self._find_transform(self._matches)
                    return
                else:
                    rospy.sleep(INACTIVE_SLEEP_DURATION)

    def _find_transform(self, matches):
        source_points = np.concatenate([x.source_points for x in self._matches])
        destination_points = np.concatenate([x.destination_points for x in self._matches])
        M, _ = cv2.estimateAffinePartial2D(source_points, destination_points)

        source_width = matches[0].source_width
        source_height = matches[0].source_height
        destination_width = matches[0].destination_width
        destination_height = matches[0].destination_height

        scale = math.sqrt(M[0, 0]**2 + M[1, 0]**2)
        center_x = (M[0, 2] + source_width) / destination_width
        center_y = (M[1, 2] + source_height) / destination_height
        rotation = math.acos(M[0, 0] / scale)
        width = source_width / destination_width * scale
        height = source_height / destination_height * scale

        Camera3dCalibration(center_x, center_y, width, height).save_json_file()

        rospy.loginfo('******* Results *******')
        rospy.loginfo(f'scale={scale}, center_x={center_x}, center_y={center_y}, width={width}, height={height}, rotation={rotation}')


def main():
    rospy.init_node('calibrate_sound_object_person_following_node')
    explore_node = CalibrateSoundObjectPersonFollowingNode()
    explore_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
