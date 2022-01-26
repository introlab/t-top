#!/usr/bin/env python3

import threading

import numpy as np

import torch

import rospy
from std_msgs.msg import Float32, Bool, Empty
from audio_utils.msg import AudioFrame

from dnn_utils import TTopKeywordSpotter
import hbba_lite


SUPPORTED_AUDIO_FORMAT = 'signed_16'
SUPPORTED_CHANNEL_COUNT = 1


class RobotNameDetectorNode:
    def __init__(self):
        self._lock = threading.Lock()

        self._message_rate_value = rospy.get_param('~message_rate', 10)
        self._sound_rms_attack = rospy.get_param('~sound_rms_attack', 0.05)
        self._sound_rms_release = rospy.get_param('~sound_rms_release', 0.99)
        self._sound_rms_presence_threshold = rospy.get_param('~sound_rms_presence_threshold', 0.05)

        self._inference_type = rospy.get_param('~inference_type', None)

        self._robot_name_model_probability_threshold = rospy.get_param('~robot_name_model_probability_threshold')
        self._robot_name_model_interval = rospy.get_param('~robot_name_model_interval')
        self._robot_name_model_analysis_delay = rospy.get_param('~robot_name_model_analysis_delay')
        self._robot_name_model_analysis_count = rospy.get_param('~robot_name_model_analysis_count')

        self._robot_name_model = TTopKeywordSpotter(inference_type=self._inference_type)
        self._robot_name_model_output_index = self._robot_name_model.get_class_names().index('T-Top')

        self._sound_rms = 0
        self._robot_name_model_buffer = torch.zeros(self._robot_name_model.get_supported_duration(), dtype=torch.float32)
        self._robot_name_model_interval_count = self._robot_name_model_interval - self._robot_name_model_analysis_delay
        self._robot_name_model_analysis_enabled = False
        self._robot_name_model_analysis_waiting = False
        self._robot_name_model_probabilities = []

        self._message_rate = rospy.Rate(self._message_rate_value)
        self._sound_rms_pub = rospy.Publisher('sound_rms', Float32, queue_size=10)
        self._sound_presence_pub = rospy.Publisher('sound_presence', Bool, queue_size=10)
        self._robot_name_detected_pub = rospy.Publisher('robot_name_detected', Empty, queue_size=10)

        self._hbba_filter_state = hbba_lite.OnOffHbbaFilterState('audio_in/filter_state')
        self._audio_sub = rospy.Subscriber('audio_in', AudioFrame, self._audio_cb, queue_size=100)

    def _audio_cb(self, msg):
        if msg.format != SUPPORTED_AUDIO_FORMAT or \
                msg.channel_count != SUPPORTED_CHANNEL_COUNT or \
                msg.sampling_frequency != self._robot_name_model.get_supported_sampling_frequency():
            rospy.logerr('Invalid audio frame (msg.format={}, msg.channel_count={}, msg.sampling_frequency={}})'
                .format(msg.format, msg.channel_count, msg.sampling_frequency))
            return

        audio_frame = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / -np.iinfo(np.int16).min
        new_sound_rms = self._calculate_sound_rms(audio_frame)

        if not self._hbba_filter_state.is_filtering_all_messages:
            self._detect_robot_name(audio_frame, new_sound_rms)

        with self._lock:
            self._sound_rms = new_sound_rms

    def _calculate_sound_rms(self, audio_frame):
        frame_rms = np.sqrt(np.mean(audio_frame**2))
        if frame_rms > self._sound_rms:
            return self._sound_rms_attack * self._sound_rms + (1 - self._sound_rms_attack) * frame_rms
        else:
            return self._sound_rms_release * self._sound_rms + (1 - self._sound_rms_release) * frame_rms

    def _detect_robot_name(self, audio_frame, sound_rms):
        self._update_robot_name_buffer(audio_frame)

        if sound_rms > self._sound_rms_presence_threshold and not self._robot_name_model_analysis_enabled:
            self._robot_name_model_analysis_enabled = True
            self._robot_name_model_analysis_waiting = False

        if len(self._robot_name_model_probabilities) >= self._robot_name_model_analysis_count:
            self._publish_robot_name_detected_if_needed()
            self._robot_name_model_interval_count = self._robot_name_model_interval - self._robot_name_model_analysis_delay
            self._robot_name_model_analysis_waiting = True
            self._robot_name_model_probabilities = []
        elif self._robot_name_model_analysis_enabled and not self._robot_name_model_analysis_waiting:
            self._robot_name_model_interval_count += audio_frame.shape[0]
            self._detect_robot_name_if_needed()
        elif self._robot_name_model_analysis_waiting and sound_rms < self._sound_rms_presence_threshold:
            self._robot_name_model_analysis_waiting = False
            self._robot_name_model_analysis_enabled = False

    def _update_robot_name_buffer(self, audio_frame):
        self._robot_name_model_buffer = torch.roll(self._robot_name_model_buffer, -audio_frame.shape[0])
        self._robot_name_model_buffer[-audio_frame.shape[0]:] = torch.from_numpy(audio_frame.copy())

    def _publish_robot_name_detected_if_needed(self):
        joint_probability = np.prod(np.array(self._robot_name_model_probabilities))
        if joint_probability > self._robot_name_model_probability_threshold:
            self._robot_name_detected_pub.publish(Empty())

    def _detect_robot_name_if_needed(self):
        if self._robot_name_model_interval_count >= self._robot_name_model_interval:
            probabilities = self._robot_name_model(self._robot_name_model_buffer)

            self._robot_name_model_interval_count = 0
            self._robot_name_model_probabilities.append(probabilities[self._robot_name_model_output_index].item())

    def run(self):
        while not rospy.is_shutdown():
            with self._lock:
                sound_rms = self._sound_rms

            self._publish_sound_rms_messages(sound_rms)

            self._message_rate.sleep()

    def _publish_sound_rms_messages(self, sound_rms):
        sound_rms_msg = Float32()
        sound_rms_msg.data = sound_rms
        self._sound_rms_pub.publish(sound_rms_msg)

        sound_presence_msg = Bool()
        sound_presence_msg.data = sound_rms > self._sound_rms_presence_threshold
        self._sound_presence_pub.publish(sound_presence_msg)


def main():
    rospy.init_node('robot_name_detector_node')
    robot_name_detector_node = RobotNameDetectorNode()
    robot_name_detector_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
