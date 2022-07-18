#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os
import threading

import numpy as np
from scipy import signal

import librosa

import rospy
import rospkg
from std_msgs.msg import Float32
from talk.msg import Text, Done
from audio_utils.msg import AudioFrame

import hbba_lite

from talk.lib_voice_generator import GoogleVoiceGenerator, CachedVoiceGenerator


class TalkNode:
    def __init__(self):
        language = rospy.get_param('~language')
        speaking_rate = rospy.get_param('~speaking_rate')
        cache_size = rospy.get_param('~cache_size')

        self._mouth_signal_gain = rospy.get_param('~mouth_signal_gain')
        self._sampling_frequency = rospy.get_param('~sampling_frequency')
        self._frame_sample_count = rospy.get_param('~frame_sample_count')

        self._rospack = rospkg.RosPack()
        self._pkg_path = self._rospack.get_path('talk')
        audio_directory_path = os.path.join(self._pkg_path, 'audio_files')

        google_voice_generator = GoogleVoiceGenerator(audio_directory_path, language, speaking_rate)
        self._voice_generator = CachedVoiceGenerator(google_voice_generator, cache_size)

        self._mouth_signal_scale_pub = rospy.Publisher('face/mouth_signal_scale', Float32, queue_size=5)
        self._audio_pub = hbba_lite.OnOffHbbaPublisher('audio_out', AudioFrame, queue_size=5)
        self._done_talking_pub = rospy.Publisher('talk/done', Done, queue_size=5)

        self._text_sub_lock = threading.Lock()
        self._text_sub = rospy.Subscriber('talk/text', Text, self._on_text_received_cb, queue_size=1)

    def _on_text_received_cb(self, msg):
        with self._text_sub_lock:
            if self._audio_pub.is_filtering_all_messages:
                return

            try:
                if msg.text != '':
                    file_path = self._voice_generator.generate(msg.text)
                    self._play_audio(file_path)
                ok = True
            except Exception as e:
                rospy.logerr(f'Unable to talk ({e})')
                ok = False

            self._done_talking_pub.publish(Done(id=msg.id, ok=ok))

    def _play_audio(self, file_path):
        frames = self._load_frames(file_path)

        global_energy_filter_sos, global_energy_filter_zi = self._initialize_global_energy_filter()
        current_energy_filter_sos, current_energy_filter_zi = self._initialize_current_energy_filter()
        mouth_signal_filter_sos, mouth_signal_filter_zi = self._initialize_mouth_signal_filter()

        mouth_signal_msg = Float32()
        audio_frame = AudioFrame()
        audio_frame.format = 'float'
        audio_frame.channel_count = 1
        audio_frame.sampling_frequency = self._sampling_frequency
        audio_frame.frame_sample_count = self._frame_sample_count

        rate = rospy.Rate(self._sampling_frequency / self._frame_sample_count)
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

            audio_frame.header.stamp = rospy.Time.now()
            audio_frame.data = frame.tobytes()
            self._audio_pub.publish(audio_frame)

            rate.sleep()

        mouth_signal_msg.data = 0.0
        self._mouth_signal_scale_pub.publish(mouth_signal_msg)

    def _load_frames(self, file_path):
        waveform, _ = librosa.load(file_path, sr=self._sampling_frequency, res_type='kaiser_fast')
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
        rospy.spin()


def main():
    rospy.init_node('talk_node')
    talk_node = TalkNode()
    talk_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
