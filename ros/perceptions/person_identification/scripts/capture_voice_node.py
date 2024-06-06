#!/usr/bin/env python3

import numpy as np

import rclpy
import rclpy.node

from hbba_lite_srvs.srv import SetOnOffFilterState
from perception_msgs.msg import AudioAnalysis

import person_identification


class CaptureVoiceNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('capture_voice_node')

        self._name = self.declare_parameter('name', '').get_parameter_value().string_value
        self._mean_size = self.declare_parameter('mean_size', 10).get_parameter_value().integer_value

        self._descriptors = []
        self.audio_analysis = self.create_subscription(AudioAnalysis, 'audio_analysis', self._audio_analysis_cb, 1)

    def _audio_analysis_cb(self, msg):
        if len(msg.voice_descriptor) > 0:
            self._descriptors.append(msg.voice_descriptor)
            self.get_logger().info('voice')
        else:
            self.get_logger().info('no voice')

    def run(self):
        self.enable_audio_analyzer()

        while rclpy.ok():
            size = len(self._descriptors)

            if size == self._mean_size:
                self.audio_analysis.unregister()
                self._save_new_descriptor()
                return
            else:
                rclpy.spin_once(self)

    def enable_audio_analyzer(self):
        AUDIO_ANALYZER_FILTER_STATE_SERVICE = 'audio_analyzer/filter_state'

        client = self.create_client(SetOnOffFilterState, AUDIO_ANALYZER_FILTER_STATE_SERVICE)
        client.wait_for_service()

        request = SetOnOffFilterState.Request()
        request.is_filtering_all_messages = False

        future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, future)

    def _save_new_descriptor(self):
        descriptor = np.array(self._descriptors).mean(axis=0).tolist()
        people = person_identification.load_people()
        person_identification.add_person(people, self._name, {'voice': descriptor})
        person_identification.save_people(people)

        self.get_logger().info('*********************** FINISHED ***********************')


def main():
    rclpy.init()
    capture_voice_node = CaptureVoiceNode()

    try:
        capture_voice_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        capture_voice_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
