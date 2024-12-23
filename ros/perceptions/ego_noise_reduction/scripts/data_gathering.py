#!/usr/bin/env python3

import os
import math
import time
from collections import defaultdict

import numpy as np
import librosa
import scipy.signal

import rclpy
import rclpy.node

from std_msgs.msg import Float32, Int32, Empty
from audio_utils_msgs.msg import AudioFrame


STARTUP_DELAY_S = 10.
SUPPORTED_AUDIO_FORMAT = 'signed_32'

CIRCLE_DEGREES = 360


def sqrt_hann(M):
    return np.sqrt(scipy.signal.windows.hann(M))


class AudioAnalyser:
    def __init__(self, node, channel_count, n_fft) -> None:
        self._node = node
        self._channel_count = channel_count
        self._n_fft = n_fft

    def analyse(self, input_directory, output_directory):
        self._node.get_logger().info('Analysing the torso_servo files')
        self._analyze_specific_type(input_directory, output_directory, 'torso_servo')

        self._node.get_logger().info('Analysing the head_servo_id1 files')
        self._analyze_specific_type(input_directory, output_directory, 'head_servo_id1')

        self._node.get_logger().info('Analysing the head_servo_id2 files')
        self._analyze_specific_type(input_directory, output_directory, 'head_servo_id2')

        self._node.get_logger().info('Analysing the head_servo_id3 files')
        self._analyze_specific_type(input_directory, output_directory, 'head_servo_id3')

        self._node.get_logger().info('Analysing the head_servo_id4 files')
        self._analyze_specific_type(input_directory, output_directory, 'head_servo_id4')

        self._node.get_logger().info('Analysing the head_servo_id5 files')
        self._analyze_specific_type(input_directory, output_directory, 'head_servo_id5')

        self._node.get_logger().info('Analysing the head_servo_id6 files')
        self._analyze_specific_type(input_directory, output_directory, 'head_servo_id6')

    def _analyze_specific_type(self, input_directory, output_directory, prefix):
        paths_by_speed = self._list_paths_by_speed(input_directory, prefix)
        paths_by_speed_orientation = self._list_paths_by_speed_orientation(input_directory, prefix)

        self._node.get_logger().info('\tCalculating base noise magnitudes')
        base_noise_magnitudes_by_speed = self._get_base_noise_magnitudes(paths_by_speed)

        self._node.get_logger().info('\tCalculating orientation tf')
        orientation_tf = self._get_orientation_tf(paths_by_speed_orientation, base_noise_magnitudes_by_speed)

        self._node.get_logger().info('\tCalculating channel tf')
        channel_tf = self._get_channel_tf(paths_by_speed_orientation, base_noise_magnitudes_by_speed, orientation_tf)

        self._write_dict_of_array(base_noise_magnitudes_by_speed,
                                  os.path.join(output_directory, f'{prefix}_base_noise_magnitudes.txt'))
        self._write_dict_of_array(orientation_tf,
                                  os.path.join(output_directory, f'{prefix}_orientation_tf.txt'))
        self._write_dict_of_array(channel_tf,
                                  os.path.join(output_directory, f'{prefix}_channel_tf.txt'))

    def _list_audio_paths(self, directory, prefix):
        def predicate(x):
            return x.endswith('.raw') and x.startswith(prefix)

        return [os.path.join(directory, x)
                for x in filter(predicate, os.listdir(directory))]

    def _list_paths_by_speed(self, directory, prefix):
        paths_by_speed = defaultdict(list)
        for path in self._list_audio_paths(directory, prefix):
            speed = int(path[path.rfind('speed') + 5: path.rfind('.raw')])
            paths_by_speed[speed].append(path)
        return dict(paths_by_speed)

    def _list_paths_by_speed_orientation(self, directory, prefix):
        paths_by_speed_orientation = defaultdict(list)
        for path in self._list_audio_paths(directory, prefix):
            orientation = int(path[path.find('deg') + 3: path.find('_speed')])
            speed = int(path[path.find('speed') + 5: path.find('.raw')])
            paths_by_speed_orientation[(speed, orientation)].append(path)
        return dict(paths_by_speed_orientation)

    def _load_raw_audio_file(self, path):
        return np.fromfile(path, dtype=np.int32).reshape(-1, self._channel_count).astype(np.float32) / -np.iinfo(np.int32).min

    def _stft(self, x):
        hop_length = self._n_fft // 2
        return librosa.stft(x, n_fft=self._n_fft, hop_length=hop_length, window=sqrt_hann, center=False)

    def _get_base_noise_magnitudes(self, paths_by_speed):
        stfts_by_speed = defaultdict(list)
        for i, (speed, paths) in enumerate(paths_by_speed.items()):
            self._node.get_logger().info(f'\t\t{i+1}/{len(paths_by_speed)}')
            for path in paths:
                x = self._load_raw_audio_file(path)
                for c in range(x.shape[1]):
                    X = self._stft(x[:, c])
                    stfts_by_speed[speed].append(X)

        return {speed: np.abs(np.concatenate(X, axis=1)).mean(axis=1) for speed, X in stfts_by_speed.items()}

    def _get_orientation_tf(self, paths_by_speed_orientation, base_noise_magnitudes_by_speed):
        stfts_by_speed_orientation = defaultdict(list)
        for (speed, orientation), paths in paths_by_speed_orientation.items():
            for path in paths:
                x = self._load_raw_audio_file(path)
                for c in range(x.shape[1]):
                    X = self._stft(x[:, c])
                    stfts_by_speed_orientation[(speed, orientation)].append(X)

        tf_by_speed_orientation = {}
        for (speed, orientation), X in stfts_by_speed_orientation.items():
            X = np.concatenate(X, axis=1)
            tf_by_speed_orientation[(speed, orientation)] = np.abs(X).mean(axis=1) / base_noise_magnitudes_by_speed[speed]

        tf_by_orientation = {}
        count_by_orientation = {}
        for (speed, orientation), tf in tf_by_speed_orientation.items():
            if orientation in tf_by_orientation:
                tf_by_orientation[orientation] += tf
                count_by_orientation[orientation] += 1
            else:
                tf_by_orientation[orientation] = tf
                count_by_orientation[orientation] = 1

        for orientation in tf_by_orientation.keys():
            tf_by_orientation[orientation] /= count_by_orientation[orientation]
            tf_by_orientation[orientation] = tf_by_orientation[orientation]

        return tf_by_orientation

    def _get_channel_tf(self, paths_by_speed_orientation, base_noise_magnitudes_by_speed, orientation_tf):
        stfts_by_speed_orientation_channel = defaultdict(list)
        for (speed, orientation), paths in paths_by_speed_orientation.items():
            for path in paths:
                x = self._load_raw_audio_file(path)
                for c in range(x.shape[1]):
                    X = self._stft(x[:, c])
                    stfts_by_speed_orientation_channel[(speed, orientation, c)].append(X)

        tf_by_speed_orientation_channel = {}
        for (speed, orientation, c), X in stfts_by_speed_orientation_channel.items():
            X = np.concatenate(X, axis=1)
            tf_by_speed_orientation_channel[(speed, orientation, c)] = \
                np.abs(X).mean(axis=1) / (base_noise_magnitudes_by_speed[speed] * orientation_tf[orientation])

        tf_by_channel = {}
        count_by_channel = {}
        for (speed, orientation, c), tf in tf_by_speed_orientation_channel.items():
            if orientation in tf_by_channel:
                tf_by_channel[c] += tf
                count_by_channel[c] += 1
            else:
                tf_by_channel[c] = tf
                count_by_channel[c] = 1

        for c in tf_by_channel.keys():
            tf_by_channel[c] /= count_by_channel[c]
            tf_by_channel[c] = tf_by_channel[c]

        return tf_by_channel

    def _write_dict_of_array(self, dict_of_array, path):
        with open(path, 'w') as file:
            for key, value in dict_of_array.items():
                file.write(f'{key}|')
                file.write(' '.join(map(str, value)))
                file.write('\n')

class DataGatheringNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('data_gathering')

        self._torso_orientation_deg = 0
        self._moving_servo_id = 0
        self._moving_servo_speed = 0
        self._is_head_servo_audio_recording = False
        self._is_torso_servo_audio_recording = False

        self._n_fft = self.declare_parameter('n_fft', 1024).get_parameter_value().integer_value
        self._sampling_frequency = self.declare_parameter('sampling_frequency', 16000).get_parameter_value().integer_value
        self._channel_count = self.declare_parameter('channel_count', 16).get_parameter_value().integer_value
        self._audio_analyzer = AudioAnalyser(self._channel_count, self._n_fft)

        self._pkg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        self._audio_data_directory_path = os.path.join(self._pkg_path, 'audio_data')
        self._noise_data_directory_path = os.path.join(self._pkg_path, 'noise_data')

        os.makedirs(self._audio_data_directory_path, exist_ok=True)
        os.makedirs(self._noise_data_directory_path, exist_ok=True)

        self._write_noise_data_info()

        self._start_ego_noise_data_gathering_pub = self.create_publisher(Empty, 'opencr/start_ego_noise_data_gathering', 10)

        self._current_torso_orientation_sub = self.create_subscription(Float32, 'opencr/current_torso_orientation',
                                                                       self._current_torso_orientation_cb, 10)
        self._moving_servo_id_sub = self.create_subscription(Int32, 'opencr/moving_servo_id',
                                                             self._moving_servo_id_cb, 10)
        self._moving_servo_speed = self.create_subscription(Int32, 'opencr/moving_servo_speed',
                                                            self._moving_servo_speed_cb, 10)

        self._start_head_servo_audio_recording_sub = self.create_subscription(Empty, 'opencr/start_head_servo_audio_recording',
                                                                              self._start_head_servo_audio_recording_cb, 10)
        self._stop_head_servo_audio_recording_sub = self.create_subscription(Empty, 'opencr/stop_head_servo_audio_recording',
                                                                             self._stop_head_servo_audio_recording_cb, 10)

        self._start_torso_servo_audio_recording_sub = self.create_subscription(Empty, 'opencr/start_torso_servo_audio_recording',
                                                                               self._start_torso_servo_audio_recording_cb, 10)
        self._stop_torso_servo_audio_recording_sub = self.create_subscription(Empty, 'opencr/stop_torso_servo_audio_recording',
                                                                              self._stop_torso_servo_audio_recording_cb, 10)

        self._audio_sub = self.create_subscription(AudioFrame, 'audio_in', self._audio_cb, 100)

        self._ego_noise_data_gathering_finished_sub = self.create_subscription(Empty, 'opencr/ego_noise_data_gathering_finished',
                                                                               self._ego_noise_data_gathering_finished_cb, 10)

    def _write_noise_data_info(self):
        with open(os.path.join(self._noise_data_directory_path, 'info.txt'), 'w') as f:
            f.write(f'{self._n_fft}\n')
            f.write(f'{self._sampling_frequency}\n')
            f.write(f'{self._channel_count}\n')

    def _current_torso_orientation_cb(self, msg):
        self._torso_orientation_deg = math.degrees(msg.data)

    def _moving_servo_id_cb(self, msg):
        self._moving_servo_id = msg.data

    def _moving_servo_speed_cb(self, msg):
        base = 5
        self._moving_servo_speed = base * round(msg.data / base)

    def _start_head_servo_audio_recording_cb(self, msg):
        self._is_head_servo_audio_recording = True

    def _stop_head_servo_audio_recording_cb(self, msg):
        self._is_head_servo_audio_recording = False

    def _start_torso_servo_audio_recording_cb(self, msg):
        self._is_torso_servo_audio_recording = True

    def _stop_torso_servo_audio_recording_cb(self, msg):
        self._is_torso_servo_audio_recording = False

    def _audio_cb(self, msg):
        if msg.format != SUPPORTED_AUDIO_FORMAT or \
                msg.channel_count != self._channel_count or \
                msg.sampling_frequency != self._sampling_frequency:
            self.get_logger().error('Invalid audio frame (msg.format={}, msg.channel_count={}, msg.sampling_frequency={}})'
                .format(msg.format, msg.channel_count, msg.sampling_frequency))
            return

        if self._is_head_servo_audio_recording:
            self._append_to(self._get_head_servo_path(), msg.data)
        if self._is_torso_servo_audio_recording:
            self._append_to(self._get_torso_servo_path(), msg.data)

    def _get_head_servo_path(self):
        deg = round(self._torso_orientation_deg) % CIRCLE_DEGREES
        name = f'head_servo_id{self._moving_servo_id}_deg{deg}_speed{self._moving_servo_speed}.raw'
        return os.path.join(self._audio_data_directory_path, name)

    def _get_torso_servo_path(self):
        base = 10
        deg = base * round(self._torso_orientation_deg / base)
        deg %= CIRCLE_DEGREES
        name = f'torso_servo_deg{deg}_speed{self._moving_servo_speed}.raw'
        return os.path.join(self._audio_data_directory_path, name)

    def _append_to(self, path, data):
        with open(path, 'a+b') as file:
            file.write(data)

    def _ego_noise_data_gathering_finished_cb(self, msg):
        self._audio_analyzer.analyse(self._audio_data_directory_path, self._noise_data_directory_path)

    def run(self):
        time.sleep(STARTUP_DELAY_S)
        self._start_ego_noise_data_gathering_pub.publish(Empty())

        rclpy.spin(self)


def main():
    rclpy.init()
    data_gathering = DataGatheringNode()

    try:
        data_gathering.run()
    except KeyboardInterrupt:
        pass
    finally:
        data_gathering.destroy_node()
        
    rclpy.shutdown()


if __name__ == '__main__':
    main()
