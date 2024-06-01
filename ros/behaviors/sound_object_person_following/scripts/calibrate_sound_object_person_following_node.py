#!/usr/bin/env python3

import math

import numpy as np
import cv2

import rclpy
import rclpy.node

import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import Image

from sound_object_person_following import Camera3dCalibration


MIN_MATCH_COUNT = 5
LOWES_RATIO_THRESHOLD = 0.7


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


class CalibrateSoundObjectPersonFollowingNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('calibrate_sound_object_person_following_node')

        self._match_count = self.declare_parameter('match_count', 100).get_parameter_value().integer_value
        if self._match_count < 1:
            raise ValueError('match_count must be at least 1.')

        self._cv_bridge = CvBridge()
        self._orb = cv2.ORB_create()
        self._matcher = cv2.BFMatcher()

        self._matches = []

        try:
            self._calibration = Camera3dCalibration.load_json_file()
        except FileNotFoundError:
            self._calibration = None

        self._camera_2d_wide_cropped_image_pub = self.create_publisher(Image, 'camera_2d_wide/cropped_image', 1)

        camera_3d_image_sub = message_filters.Subscriber(self, Image, 'camera_3d/color/image_raw')
        camera_2d_wide_image_sub = message_filters.Subscriber(self, Image, 'camera_2d_wide/image_rect')
        self._image_ts = message_filters.ApproximateTimeSynchronizer([camera_3d_image_sub, camera_2d_wide_image_sub], 10, 0.1)
        self._image_ts.registerCallback(self._image_cb)

    def _image_cb(self, camera_3d_image_msg, camera_2d_wide_image_msg):
        camera_3d_image = self._cv_bridge.imgmsg_to_cv2(camera_3d_image_msg, 'bgr8')
        camera_2d_wide_image = self._cv_bridge.imgmsg_to_cv2(camera_2d_wide_image_msg, 'bgr8')

        if self._calibration is None:
            match = self._match_images(camera_3d_image, camera_2d_wide_image)
            if match is not None:
                self._matches.append(match)
        else:
            self._publish_camera_2d_wide_cropped_image(camera_2d_wide_image, self._calibration)

    def _match_images(self, source_image, destination_image):
        source_keypoints, source_descriptors = self._orb.detectAndCompute(source_image, None)
        destination_keypoints, destination_descriptors = self._orb.detectAndCompute(destination_image, None)

        matches = self._matcher.knnMatch(source_descriptors, destination_descriptors, 2)
        good_matches = []
        for m,n in matches:
            if m.distance < LOWES_RATIO_THRESHOLD * n.distance:
                good_matches.append(m)

        if len(good_matches) < MIN_MATCH_COUNT:
            self.get_logger().warn(f'Not enough ORB feature matches (count={len(good_matches)})')
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
        while rclpy.ok():
            has_calibration = self._calibration is not None

            if len(self._matches) >= self._match_count and not has_calibration:
                self._calibration = self._find_transform(self._matches)
            else:
                if not has_calibration:
                    self.get_logger().info(f'Match count: {len(self._matches)}')

                rclpy.spin_once(self)

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

        self.get_logger().info('******* Results *******')
        self.get_logger().info(f'scale={scale}, center_x={center_x}, center_y={center_y}, width={width}, height={height}, rotation={rotation}')

        calibration = Camera3dCalibration(center_x, center_y, width, height)
        calibration.save_json_file()
        return calibration


def main():
    rclpy.init()
    calibrate_sound_object_person_following_node = CalibrateSoundObjectPersonFollowingNode()

    try:
        calibrate_sound_object_person_following_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        calibrate_sound_object_person_following_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
