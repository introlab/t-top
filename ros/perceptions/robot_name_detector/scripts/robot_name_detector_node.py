#!/usr/bin/env python3

import numpy as np

import torch

import rospy
from std_msgs.msg import Float32, Bool, Empty
from audio_utils.msg import AudioFrame
from daemon_ros_client.msg import LedColors

from dnn_utils import TTopKeywordSpotter
import hbba_lite


SUPPORTED_AUDIO_FORMAT = 'signed_16'
SUPPORTED_CHANNEL_COUNT = 1

NONE_LED_COLORS = LedColors()
for led_color in NONE_LED_COLORS.colors:
    led_color.red = 0
    led_color.green = 0
    led_color.blue = 0


class SoundRmsFilter:
    def __init__(self, attack, release, initial_sound_rms=0.0):
        self._attack = attack
        self._release = release
        self._sound_rms = initial_sound_rms

    def update(self, frame_rms):
        if frame_rms > self._sound_rms:
            self._sound_rms = self._attack * self._sound_rms + (1 - self._attack) * frame_rms
        else:
            self._sound_rms = self._release * self._sound_rms + (1 - self._release) * frame_rms

        return self._sound_rms

    def reset(self):
        self._sound_rms = 0.0


class RobotNameDetectorNode:
    def __init__(self):
        self._led_status_duration_s = rospy.get_param('~led_status_duration_s', 1.0)

        self._sound_presence_relative_threshold = rospy.get_param('~sound_presence_relative_threshold', 1.05)

        self._robot_name_model_probability_threshold = rospy.get_param('~robot_name_model_probability_threshold')
        self._robot_name_model_interval = rospy.get_param('~robot_name_model_interval')
        self._robot_name_model_analysis_delay = rospy.get_param('~robot_name_model_analysis_delay')
        self._robot_name_model_analysis_count = rospy.get_param('~robot_name_model_analysis_count')

        self._fast_sound_rms_filter = SoundRmsFilter(rospy.get_param('~fast_sound_rms_attack', 0.05),
                                                     rospy.get_param('~fast_sound_rms_release', 0.99))
        self._slow_sound_rms_filter = SoundRmsFilter(rospy.get_param('~slow_sound_rms_attack', 0.9),
                                                     rospy.get_param('~slow_sound_rms_release', 0.9),
                                                     initial_sound_rms=0.1)

        self._robot_name_model = TTopKeywordSpotter(inference_type=rospy.get_param('~inference_type', None))
        self._robot_name_model_output_index = self._robot_name_model.get_class_names().index('T-Top')

        self._robot_name_model_buffer = torch.zeros(self._robot_name_model.get_supported_duration(), dtype=torch.float32)
        self._robot_name_model_interval_count = self._robot_name_model_interval - self._robot_name_model_analysis_delay
        self._robot_name_model_analysis_enabled = False
        self._robot_name_model_analysis_waiting = False
        self._robot_name_model_probabilities = []

        self._detection_status = None
        self._detection_time_s = 0.0

        # Warm up, to avoid inference delays
        self._robot_name_model(self._robot_name_model_buffer)

        self._fast_sound_rms_pub = rospy.Publisher('fast_sound_rms', Float32, queue_size=10)
        self._slow_sound_rms_pub = rospy.Publisher('slow_sound_rms', Float32, queue_size=10)
        self._sound_presence_pub = rospy.Publisher('sound_presence', Bool, queue_size=10)
        self._robot_name_detected_pub = rospy.Publisher('robot_name_detected', Empty, queue_size=10)
        self._led_colors_pub = hbba_lite.OnOffHbbaPublisher('robot_name_detector/set_led_colors', LedColors, queue_size=1,
                                                            state_service_name='led_status/filter_state')
        self._led_colors_pub.on_filter_state_changing(self._led_colors_hbba_filter_state_cb)

        self._hbba_filter_state = hbba_lite.OnOffHbbaFilterState('audio_in/filter_state')
        self._hbba_filter_state.on_changed(self._robot_name_detector_hbba_state_changed_cb)
        self._audio_sub = rospy.Subscriber('audio_in', AudioFrame, self._audio_cb, queue_size=100)

    def _led_colors_hbba_filter_state_cb(self, publish_forced,
                                         previous_is_filtering_all_messages, new_is_filtering_all_messages):
        if not previous_is_filtering_all_messages and new_is_filtering_all_messages:
            publish_forced(NONE_LED_COLORS)

    def _robot_name_detector_hbba_state_changed_cb(self, previous_is_filtering_all_messages, new_is_filtering_all_messages):
        self._fast_sound_rms_filter.reset()

    def _audio_cb(self, msg):
        if msg.format != SUPPORTED_AUDIO_FORMAT or \
                msg.channel_count != SUPPORTED_CHANNEL_COUNT or \
                msg.sampling_frequency != self._robot_name_model.get_supported_sampling_frequency():
            rospy.logerr('Invalid audio frame (msg.format={}, msg.channel_count={}, msg.sampling_frequency={}})'
                .format(msg.format, msg.channel_count, msg.sampling_frequency))
            return

        audio_frame = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / -np.iinfo(np.int16).min
        frame_rms = np.sqrt(np.mean(audio_frame**2))
        fast_sound_rms = self._fast_sound_rms_filter.update(frame_rms)
        slow_sound_rms = self._slow_sound_rms_filter.update(frame_rms)
        presence = fast_sound_rms > slow_sound_rms * self._sound_presence_relative_threshold

        self._publish_sound_rms_messages(fast_sound_rms, slow_sound_rms, presence)

        if not self._hbba_filter_state.is_filtering_all_messages:
            self._detect_robot_name(audio_frame, presence)
        if not self._led_colors_pub.is_filtering_all_messages:
            self._publish_led_status(fast_sound_rms, slow_sound_rms)

    def _publish_sound_rms_messages(self, fast_sound_rms, slow_sound_rms, presence):
        sound_rms_msg = Float32()

        sound_rms_msg.data = fast_sound_rms
        self._fast_sound_rms_pub.publish(sound_rms_msg)

        sound_rms_msg.data = slow_sound_rms
        self._slow_sound_rms_pub.publish(sound_rms_msg)

        sound_presence_msg = Bool()
        sound_presence_msg.data = presence
        self._sound_presence_pub.publish(sound_presence_msg)

    def _detect_robot_name(self, audio_frame, presence):
        self._update_robot_name_buffer(audio_frame)

        if presence and not self._robot_name_model_analysis_enabled:
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
        elif self._robot_name_model_analysis_waiting and not presence:
            self._robot_name_model_analysis_waiting = False
            self._robot_name_model_analysis_enabled = False

    def _update_robot_name_buffer(self, audio_frame):
        self._robot_name_model_buffer = torch.roll(self._robot_name_model_buffer, -audio_frame.shape[0])
        self._robot_name_model_buffer[-audio_frame.shape[0]:] = torch.from_numpy(audio_frame.copy())

    def _publish_robot_name_detected_if_needed(self):
        joint_probability = np.prod(np.array(self._robot_name_model_probabilities))

        if joint_probability > self._robot_name_model_probability_threshold:
            self._robot_name_detected_pub.publish(Empty())
            self._detection_status = True
        else:
            self._detection_status = False

        self._detection_time_s = rospy.get_time()

    def _detect_robot_name_if_needed(self):
        if self._robot_name_model_interval_count >= self._robot_name_model_interval:
            probabilities = self._robot_name_model(self._robot_name_model_buffer)

            self._robot_name_model_interval_count = 0
            self._robot_name_model_probabilities.append(probabilities[self._robot_name_model_output_index].item())

    def _publish_led_status(self, fast_sound_rms, slow_sound_rms, eps=1e-6):
        if rospy.get_time() - self._detection_time_s < self._led_status_duration_s and self._detection_status is not None:
            if self._detection_status:
                self._publish_led_colors(0, 255, 0)
            else:
                self._publish_led_colors(255, 0, 0)
        else:
            one_level = slow_sound_rms * self._sound_presence_relative_threshold
            zero_level = slow_sound_rms
            one_zero_diff = one_level - zero_level
            level = np.clip((fast_sound_rms - zero_level) / (one_zero_diff + eps), a_min=0.0, a_max=1.0)

            self._publish_led_colors(int(255 * level), int(255 * level), int(255 * level))

    def _publish_led_colors(self, red, green, blue):
        msg = LedColors()
        for led_color in msg.colors:
            led_color.red = red
            led_color.green = green
            led_color.blue = blue

        self._led_colors_pub.publish(msg)

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('robot_name_detector_node')
    robot_name_detector_node = RobotNameDetectorNode()
    robot_name_detector_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
