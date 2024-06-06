#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os
import time
from datetime import datetime

import numpy as np
from scipy import signal

import librosa

import rclpy
import rclpy.node

from std_msgs.msg import Float32
from talk.msg import Text, Done, Statistics
from audio_utils_msgs.msg import AudioFrame

import hbba_lite

from talk.lib_voice_generator import Language, Gender
from talk.lib_voice_generator import GoogleVoiceGenerator, PiperVoiceGenerator, CachedVoiceGenerator


class TalkNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('talk_node')

        language = Language.from_name(self.declare_parameter('language', 'en').get_parameter_value().string_value)
        gender = Gender.from_name(self.declare_parameter('gender', 'male').get_parameter_value().string_value)
        speaking_rate = self.declare_parameter('speaking_rate', 1.0).get_parameter_value().double_value
        generator_type = self.declare_parameter('generator_type', 'piper').get_parameter_value().string_value
        cache_size = self.declare_parameter('cache_size', 2000).get_parameter_value().integer_value

        self._mouth_signal_gain = self.declare_parameter('mouth_signal_gain', 0.04).get_parameter_value().double_value
        self._sampling_frequency = self.declare_parameter('sampling_frequency', 16000).get_parameter_value().integer_value
        self._frame_sample_count = self.declare_parameter('frame_sample_count', 1024).get_parameter_value().integer_value
        self._done_delay_s = self.declare_parameter('done_delay_s', 0.5).get_parameter_value().double_value

        audio_directory_path = os.path.join(os.environ['HOME'], '.ros', 't-top', 'talk', 'audio_files')

        if generator_type == 'google':
            self._voice_generator = GoogleVoiceGenerator(audio_directory_path, language, gender, speaking_rate)
        elif generator_type == 'piper':
            self._voice_generator = PiperVoiceGenerator(self, audio_directory_path, language, gender, speaking_rate)
        else:
            raise ValueError(f'Invalid generator type ({generator_type})')

        if cache_size > 0:
            self._voice_generator = CachedVoiceGenerator(self._voice_generator, cache_size)

        self._mouth_signal_scale_pub = self.create_publisher(Float32, 'face/mouth_signal_scale', 5)
        self._audio_pub = hbba_lite.OnOffHbbaPublisher(self, AudioFrame, 'audio_out', 5)
        self._done_talking_pub = self.create_publisher(Done, 'talk/done', 5)
        self._stats_pub = self.create_publisher(Statistics, 'talk/statistics', 5)

        self._text_sub = self.create_subscription(Text, 'talk/text', self._on_text_received_cb, 1)

    def _on_text_received_cb(self, msg):
        if self._audio_pub.is_filtering_all_messages:
            return

        try:
            if msg.text != '':
                start_time = datetime.now()
                file_path = self._voice_generator.generate(msg.text)
                frames = self._load_frames(file_path)
                processing_time_s = (datetime.now() - start_time).total_seconds()

                self._publish_stats(msg.text, frames, processing_time_s)

                self._play_audio(frames)
                self._voice_generator.delete_generated_file(file_path)
                time.sleep(self._done_delay_s)

            ok = True
        except Exception as e:
            self.get_logger().error(f'Unable to talk ({e})')
            ok = False

        self._done_talking_pub.publish(Done(id=msg.id, ok=ok))

    def _publish_stats(self, text, frames, processing_time_s):
        stats = Statistics()
        stats.text = text
        stats.processing_time_s = processing_time_s
        stats.header.stamp = self.get_clock().now().to_msg()
        stats.total_samples_count = 0

        for frame in frames:
            stats.total_samples_count += frame.shape[0]

        self._stats_pub.publish(stats)

    def _play_audio(self, frames):
        global_energy_filter_sos, global_energy_filter_zi = self._initialize_global_energy_filter()
        current_energy_filter_sos, current_energy_filter_zi = self._initialize_current_energy_filter()
        mouth_signal_filter_sos, mouth_signal_filter_zi = self._initialize_mouth_signal_filter()

        mouth_signal_msg = Float32()
        audio_frame = AudioFrame()
        audio_frame.format = 'float'
        audio_frame.channel_count = 1
        audio_frame.sampling_frequency = self._sampling_frequency
        audio_frame.frame_sample_count = self._frame_sample_count

        sleep_duration = self._frame_sample_count / self._sampling_frequency
        for frame in frames:
            if self._audio_pub.is_filtering_all_messages:
                break

            abs_frame = np.abs(frame)
            global_energy, global_energy_filter_zi = signal.sosfilt(global_energy_filter_sos, abs_frame, zi=global_energy_filter_zi)
            current_energy, current_energy_filter_zi = signal.sosfilt(current_energy_filter_sos, abs_frame, zi=current_energy_filter_zi)

            global_energy = global_energy.sum()
            current_energy = current_energy.sum()

            mouth_signal = np.array([current_energy - global_energy])
            mouth_signal, mouth_signal_filter_zi = signal.sosfilt(mouth_signal_filter_sos, mouth_signal, zi=mouth_signal_filter_zi)

            mouth_signal_msg.data = max(0.0, min(mouth_signal[0] * self._mouth_signal_gain, 1.0))
            self._mouth_signal_scale_pub.publish(mouth_signal_msg)

            audio_frame.header.stamp = self.get_clock().now().to_msg()
            audio_frame.data = frame.tobytes()
            self._audio_pub.publish(audio_frame)

            time.sleep(sleep_duration)

        mouth_signal_msg.data = 0.0
        self._mouth_signal_scale_pub.publish(mouth_signal_msg)

    def _load_frames(self, file_path):
        waveform, _ = librosa.load(file_path, sr=self._sampling_frequency, res_type='kaiser_fast')
        waveform = librosa.to_mono(waveform)
        pad = (self._frame_sample_count - (waveform.shape[0] % self._frame_sample_count)) % self._frame_sample_count
        waveform = np.pad(waveform, (0, pad), 'constant', constant_values=0)
        frames = np.split(waveform, np.arange(self._frame_sample_count, len(waveform), self._frame_sample_count))
        return frames

    def _initialize_global_energy_filter(self):
        global_energy_filter_sos =  signal.butter(1, 2, btype='lowpass', fs=self._sampling_frequency, output='sos')
        global_energy_filter_zi = np.zeros((global_energy_filter_sos.shape[0], 2))
        return global_energy_filter_sos, global_energy_filter_zi

    def _initialize_current_energy_filter(self):
        current_energy_filter_sos =  signal.butter(1, 10, btype='lowpass', fs=self._sampling_frequency, output='sos')
        current_energy_filter_zi = np.zeros((current_energy_filter_sos.shape[0], 2))
        return current_energy_filter_sos, current_energy_filter_zi

    def _initialize_mouth_signal_filter(self):
        mouth_signal_filter_sos =  signal.butter(1, 2.5, btype='lowpass',
                                                 fs=self._sampling_frequency // self._frame_sample_count, output='sos')
        mouth_signal_filter_zi = np.zeros((mouth_signal_filter_sos.shape[0], 2))
        return mouth_signal_filter_sos, mouth_signal_filter_zi

    def run(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    talk_node = TalkNode()

    try:
        talk_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        talk_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
