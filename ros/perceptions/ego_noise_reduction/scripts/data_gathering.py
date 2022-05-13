#!/usr/bin/env python3

import os
import math
import threading

import numpy as np
import librosa
import scipy.signal

import rospy
import rospkg

from std_msgs.msg import Float32, Int32, Empty
from audio_utils.msg import AudioFrame


STARTUP_DELAY_S = 10.
SUPPORTED_AUDIO_FORMAT = 'signed_32'


def sqrt_hann(M):
    return np.sqrt(scipy.signal.windows.hann(M))


class DataGatheringNode:
    def __init__(self):
        self._torso_orientation_deg = 0
        self._moving_servo_id = 0
        self._moving_servo_speed = 0
        self._is_head_servo_audio_recording = False
        self._is_torso_servo_audio_recording = False
        self._lock = threading.Lock()

        self._n_fft = rospy.get_param('~n_fft')
        self._sampling_frequency = rospy.get_param('~sampling_frequency')
        self._channel_count = rospy.get_param('~channel_count')

        self._rospack = rospkg.RosPack()
        self._pkg_path = self._rospack.get_path('ego_noise_reduction')
        self._audio_data_directory_path = os.path.join(self._pkg_path, 'audio_data')
        self._noise_data_directory_path = os.path.join(self._pkg_path, 'noise_data')

        os.makedirs(self._audio_data_directory_path, exist_ok=True)
        os.makedirs(self._noise_data_directory_path, exist_ok=True)

        self._write_noise_data_info()

        self._start_ego_noise_data_gathering_pub = rospy.Publisher('opencr/start_ego_noise_data_gathering', Empty, queue_size=10)

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

        self._ego_noise_data_gathering_finished_sub = rospy.Subscriber('opencr/ego_noise_data_gathering_finished', Empty,
                                                                       self._ego_noise_data_gathering_finished_cb, queue_size=10)

    def _write_noise_data_info(self):
        with open(os.path.join(self._noise_data_directory_path, 'info.txt'), 'w') as f:
            f.write(f'{self._n_fft}\n')
            f.write(f'{self._sampling_frequency}\n')
            f.write(f'{self._channel_count}\n')

    def _current_torso_orientation_cb(self, msg):
        with self._lock:
            self._torso_orientation_deg = math.degrees(msg.data)

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
        if msg.format != SUPPORTED_AUDIO_FORMAT or \
                msg.channel_count != self._channel_count or \
                msg.sampling_frequency != self._sampling_frequency:
            rospy.logerr('Invalid audio frame (msg.format={}, msg.channel_count={}, msg.sampling_frequency={}})'
                .format(msg.format, msg.channel_count, msg.sampling_frequency))
            return

        with self._lock:
            if self._is_head_servo_audio_recording:
                self._append_to(self._get_head_servo_path(), msg.data)
            if self._is_torso_servo_audio_recording:
                self._append_to(self._get_torso_servo_path(), msg.data)

    def _get_head_servo_path(self):
        deg = round(self._torso_orientation_deg)
        name = f'head_servo_id{self._moving_servo_id}_deg{deg}_speed{self._moving_servo_speed}.raw'
        return os.path.join(self._audio_data_directory_path, name)

    def _get_torso_servo_path(self):
        base = 2
        deg = base * round(self._torso_orientation_deg / base)
        name = f'torso_servo_deg{deg}_speed{self._moving_servo_speed}.raw'
        return os.path.join(self._audio_data_directory_path, name)

    def _append_to(self, path, data):
        with open(path, 'a+b') as file:
            file.write(data)

    def _ego_noise_data_gathering_finished_cb(self, msg):
        for file in os.listdir(self._audio_data_directory_path):
            self._convert_audio_data_to_noise_data(file)

    # TODO extract transfert function

    def _convert_audio_data_to_noise_data(self, file):
        input_path = os.path.join(self._audio_data_directory_path, file)
        output_path = os.path.join(self._noise_data_directory_path, os.path.splitext(file)[0] + '.txt')

        x = np.fromfile(input_path, dtype=np.int32).astype(np.float32) / -np.iinfo(np.int32).min
        x = x.reshape(-1, self._channel_count)
        X_ampl_mean = np.zeros((self._n_fft // 2 + 1, self._channel_count))
        for c in range(self._channel_count):
            X = librosa.stft(x[:, c], n_fft=self._n_fft, hop_length=self._n_fft // 2, window=sqrt_hann, center=False)
            X_ampl_mean[:, c] = np.abs(X).mean(axis=1)

        np.savetxt(output_path, X_ampl_mean)

    def run(self):
        rospy.sleep(STARTUP_DELAY_S)
        self._start_ego_noise_data_gathering_pub.publish(Empty())

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
