#!/usr/bin/env python3

import numpy as np

import rclpy
import rclpy.node

from hbba_lite.srv import SetThrottlingFilterState
from video_analyzer.msg import VideoAnalysis

import person_identification


INACTIVE_SLEEP_DURATION = 0.1


class CaptureFaceNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('capture_face_node')

        self._name = self.declare_parameter('name', '').get_parameter_value().string_value
        self._mean_size = self.declare_parameter('mean_size', 10).get_parameter_value().integer_value
        self._face_sharpness_score_threshold = self.declare_parameter('face_sharpness_score_threshold', 0.5).get_parameter_value().double_value

        self._descriptors = []
        self._video_analysis_sub = self.create_subscription(VideoAnalysis, 'video_analysis', self._video_analysis_cb, 1)

    def _video_analysis_cb(self, msg):
        face_object = None
        for object in msg.objects:
            if len(object.face_descriptor) > 0 and face_object is not None:
                self.get_logger().warn('Only one face must be present in the image.')
            elif len(object.face_descriptor) > 0:
                face_object = object

        if face_object is not None and face_object.face_sharpness_score >= self._face_sharpness_score_threshold:
            self._descriptors.append(face_object.face_descriptor)

    def run(self):
        self.enable_video_analyzer()

        while rclpy.ok():
            if len(self._descriptors) == self._mean_size:
                self._video_analysis_sub.unregister()
                self._save_new_descriptor()
                return
            else:
                rclpy.spin_once(self)

    def enable_video_analyzer(self):
        VIDEO_ANALYZER_FILTER_STATE_SERVICE = 'video_analyzer/image_raw/filter_state'

        client = self.create_client(SetThrottlingFilterState, VIDEO_ANALYZER_FILTER_STATE_SERVICE)
        client.wait_for_service()

        request = SetThrottlingFilterState.Request()
        request.is_filtering_all_messages = False
        request.rate = 1

        future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, future)

    def _save_new_descriptor(self):
        descriptor = np.array(self._descriptors).mean(axis=0).tolist()
        people = person_identification.load_people()
        person_identification.add_person(people, self._name, {'face': descriptor})
        person_identification.save_people(people)

        self.get_logger().info('*********************** FINISHED ***********************')


def main():
    rclpy.init()
    capture_face_node = CaptureFaceNode()

    try:
        capture_face_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        capture_face_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
