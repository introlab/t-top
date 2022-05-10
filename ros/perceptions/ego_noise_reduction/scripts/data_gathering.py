#!/usr/bin/env python3

import os
import threading

import rospy
import rospkg

from std_msgs.msg import Float32, Int32, Empty
from audio_utils.msg import AudioFrame


class DataGatheringNode:
    def __init__(self):
        self._torso_orientation_deg = 0
        self._moving_servo_id = 0
        self._moving_servo_speed = 0
        self._is_head_servo_audio_recording = False
        self._is_torso_servo_audio_recording = False
        self._lock = threading.Lock()

        self._rospack = rospkg.RosPack()
        self._pkg_path = self._rospack.get_path('ego_noise_reduction')
        self._data_directory_path = os.path.join(self._pkg_path, 'data')

        self._current_torso_orientation_sub = rospy.Subscriber('opencr/current_torso_orientation', Float32,
                                                               self._current_torso_orientation_cb, queue_size=10)
        self._moving_servo_id_sub = rospy.Subscriber('opencr/moving_servo_id', Int32,
                                                     self._moving_servo_id_cb, queue_size=10)
        self._moving_servo_speed = rospy.Subscriber('opencr/moving_servo_speed', Int32,
                                                    self._moving_servo_speed_cb, queue_size=10)

        self._start_head_servo_audio_recording_sub = rospy.Subscriber('opencr/start_head_servo_audio_recording', Empty,
                                                                      self._start_head_servo_audio_recording_cb, queue_size=10)
        self._stop_head_servo_audio_recording_sub = rospy.Subscriber('opencr/stop_head_servo_audio_recording', Empty,
                                                                     self._stop_head_servo_audio_recording_cb, queue_size=10)

        self._start_torso_servo_audio_recording_sub = rospy.Subscriber('opencr/start_torso_servo_audio_recording', Empty,
                                                                       self._start_torso_servo_audio_recording_cb, queue_size=10)
        self._stop_torso_servo_audio_recording_sub = rospy.Subscriber('opencr/stop_torso_servo_audio_recording', Empty,
                                                                       self._stop_torso_servo_audio_recording_cb, queue_size=10)

        self._audio_sub = rospy.Subscriber('audio_in', AudioFrame, self._audio_cb, queue_size=100)

    def current_torso_orientation_cb(self, msg):
        with self._lock:
            self._torso_orientation_deg = int(round(msg.data))

    def _moving_servo_id_cb(self, msg):
        with self._lock:
            self._moving_servo_id = msg.data

    def _moving_servo_speed_cb(self, msg):
        with self._lock:
            self._moving_servo_speed = msg.data

    def _start_head_servo_audio_recording_cb(self, msg):
        with self._lock:
            self._is_head_servo_audio_recording = True

    def _stop_head_servo_audio_recording_cb(self, msg):
        with self._lock:
            self._is_head_servo_audio_recording = False

    def _start_torso_servo_audio_recording_cb(self, msg):
        with self._lock:
            self._is_torso_servo_audio_recording = True

    def _stop_torso_servo_audio_recording_cb(self, msg):
        with self._lock:
            self._is_torso_servo_audio_recording = False

    def _audio_cb(self, msg):
        with self._lock:
            if self._is_head_servo_audio_recording:
                path = os.path.join(
                    self._data_directory_path,
                    f'head_servo_id{self._moving_servo_id}_deg{self._torso_orientation_deg}_speed{self._moving_servo_speed}.raw')
                self._append_to(path, msg.data)
            if self._is_torso_servo_audio_recording:
                path = os.path.join(
                    self._data_directory_path,
                    f'torso_servo_deg{self._torso_orientation_deg}_speed{self._moving_servo_speed}.raw')
                self._append_to(path, msg.data)

    def _append_to(path, data):
        with open(path, 'a+b') as file:
            file.write(data)

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('data_gathering')
    split_channel_node = DataGatheringNode()
    split_channel_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
