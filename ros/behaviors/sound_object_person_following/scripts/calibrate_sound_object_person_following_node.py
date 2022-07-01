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


MIN_MATCH_COUNT = 5
LOWES_RATIO_THRESHOLD = 0.7
INACTIVE_SLEEP_DURATION = 1.0


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
        self._orb = cv2.ORB_create()
        self._matcher = cv2.BFMatcher()

        self._matches_lock = threading.Lock()
        self._matches = []

        self._calibration_lock = threading.Lock()
        try:
            self._calibration = Camera3dCalibration.load_json_file()
        except FileNotFoundError:
            self._calibration = None

        self._camera_2d_wide_cropped_image_pub = rospy.Publisher('camera_2d_wide/cropped_image', Image, queue_size=1)

        camera_3d_image_sub = message_filters.Subscriber('camera_3d/color/image_raw', Image)
        camera_2d_wide_image_sub = message_filters.Subscriber('camera_2d_wide/image_rect', Image)
        self._image_ts = message_filters.ApproximateTimeSynchronizer([camera_3d_image_sub, camera_2d_wide_image_sub], 10, 0.1)
        self._image_ts.registerCallback(self._image_cb)

    def _image_cb(self, camera_3d_image_msg, camera_2d_wide_image_msg):
        camera_3d_image = self._cv_bridge.imgmsg_to_cv2(camera_3d_image_msg, 'bgr8')
        camera_2d_wide_image = self._cv_bridge.imgmsg_to_cv2(camera_2d_wide_image_msg, 'bgr8')

        with self._calibration_lock:
            calibration = self._calibration

        if calibration is None:
            match = self._match_images(camera_3d_image, camera_2d_wide_image)
            if match is not None:
                with self._matches_lock:
                    self._matches.append(match)
        else:
            self._publish_camera_2d_wide_cropped_image(camera_2d_wide_image, calibration)

    def _match_images(self, source_image, destination_image):
        source_keypoints, source_descriptors = self._orb.detectAndCompute(source_image, None)
        destination_keypoints, destination_descriptors = self._orb.detectAndCompute(destination_image, None)

        matches = self._matcher.knnMatch(source_descriptors, destination_descriptors, 2)
        good_matches = []
        for m,n in matches:
            if m.distance < LOWES_RATIO_THRESHOLD * n.distance:
                good_matches.append(m)

        if len(good_matches) < MIN_MATCH_COUNT:
            rospy.logwarn(f'Not enough ORB feature matches (count={len(good_matches)})')
            return None
        else:
            source_points = np.float32([source_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            destination_points = np.float32([destination_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        source_height, source_width, _ = source_image.shape
        destination_height, destination_width, _ = destination_image.shape

        return Match(source_points, destination_points, source_width, source_height, destination_width, destination_height)

    def _publish_camera_2d_wide_cropped_image(self, image, calibration):
        height, width, _ = image.shape
        x0 = int((calibration.center_x - calibration.half_width) * width)
        y0 = int((calibration.center_y - calibration.half_height) * height)
        x1 = int((calibration.center_x + calibration.half_width) * width + 1)
        y1 = int((calibration.center_y + calibration.half_height) * height + 1)

        camera_2d_wide_image_cropped = image[y0:y1, x0:x1, :]
        self._camera_2d_wide_cropped_image_pub.publish(self._cv_bridge.cv2_to_imgmsg(camera_2d_wide_image_cropped, 'bgr8'))

    def run(self):
        while not rospy.is_shutdown():
            with self._matches_lock, self._calibration_lock:
                match_count = len(self._matches)
                has_calibration = self._calibration is not None

            if match_count >= self._match_count and not has_calibration:
                self._calibration = self._find_transform(self._matches)
            else:
                if not has_calibration:
                    rospy.loginfo(f'Match count: {len(self._matches)}')

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
        center_x = (M[0, 2] + source_width * scale / 2) / destination_width
        center_y = (M[1, 2] + source_height * scale / 2) / destination_height
        rotation = math.acos(M[0, 0] / scale)
        width = source_width / destination_width * scale
        height = source_height / destination_height * scale

        rospy.loginfo('******* Results *******')
        rospy.loginfo(f'scale={scale}, center_x={center_x}, center_y={center_y}, width={width}, height={height}, rotation={rotation}')

        calibration = Camera3dCalibration(center_x, center_y, width, height)
        calibration.save_json_file()
        return calibration


def main():
    rospy.init_node('calibrate_sound_object_person_following_node')
    explore_node = CalibrateSoundObjectPersonFollowingNode()
    explore_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
