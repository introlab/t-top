#!/usr/bin/env python3

import traceback

import numpy as np
import cv2

import rospy
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from video_analyzer.msg import VideoAnalysis


def concatenate_horizontal(image0, image1):
    h0, w0, _ = image0.shape
    h1, w1, _ = image1.shape

    output = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    output[0:h0, 0:w0, :] = image0
    output[0:h1, w0:, :] = image1
    return output


def concatenate_vertical(image0, image1):
    h0, w0, _ = image0.shape
    h1, w1, _ = image1.shape

    output = np.zeros((h0 + h1, max(w0, w1), 3), dtype=np.uint8)
    output[0:h0, 0:w0, :] = image0
    output[h0:, 0:w1:, :] = image1
    return output


class VideoAnalysisVisualizerNode:
    def __init__(self):
        self._cv_bridge = CvBridge()

        self._video_analysis_sub = rospy.Subscriber('video_analysis', VideoAnalysis, self._video_analysis_cb, queue_size=10)
        self._video_analysis_mosaic_pub = rospy.Publisher('video_analysis_mosaic', Image, queue_size=10)

    def _video_analysis_cb(self, msg):
        objects = msg.objects
        self._sort_objects(objects)

        mosaic = np.zeros((1, 1, 3), dtype=np.uint8)
        for o in objects:
            mosaic = concatenate_vertical(mosaic, self._concatenate_object_images(o))

        header = msg.header
        msg = self._cv_bridge.cv2_to_imgmsg(mosaic, 'rgb8')
        msg.header = header
        self._video_analysis_mosaic_pub.publish(msg)

    def _sort_objects(self, objects):
        def key(o):
            return [o.center_2d.x, o.center_2d.y]
        objects.sort(key=key)

    def _concatenate_object_images(self, o):
        object_image = self._cv_bridge.imgmsg_to_cv2(o.object_image, 'rgb8')
        pose_image = None
        face_image = None
        if len(o.person_pose_2d) > 0:
            pose_image = self._cv_bridge.imgmsg_to_cv2(o.person_pose_image, 'rgb8')
            face_image = self._cv_bridge.imgmsg_to_cv2(o.face_image, 'rgb8')

            output = concatenate_horizontal(object_image, pose_image)
            output = concatenate_horizontal(output, face_image)
        else:
            output = object_image

        return output

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('video_analysis_visualizer_node')
    video_analysis_visualizer_node = VideoAnalysisVisualizerNode()
    video_analysis_visualizer_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
