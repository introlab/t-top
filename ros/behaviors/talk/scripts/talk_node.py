#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os

from google.cloud import texttospeech
from librosa.core import audio

import numpy as np
from scipy import signal

import librosa

import rospy
import rospkg
from std_msgs.msg import String, Float32, Empty
from audio_utils.msg import AudioFrame

import hbba_lite

SUPPORTED_SAMPLE_RATE = 22050


class TalkNode:
    def __init__(self):
        self._language = rospy.get_param('~language')
        self._mouth_signal_gain = rospy.get_param('~mouth_signal_gain')
        self._frame_sample_count = rospy.get_param('~frame_sample_count')
        self._rospack = rospkg.RosPack()
        self._pkg_path = self._rospack.get_path('talk')

        self._mouth_signal_scale_pub = rospy.Publisher('face/mouth_signal_scale', Float32, queue_size=5)
        self._audio_pub = hbba_lite.OnOffHbbaPublisher('audio_out', AudioFrame, queue_size=5)
        self._done_talking_pub = rospy.Publisher('talk/done', Empty, queue_size=5)

        self._text_sub = rospy.Subscriber('talk/text', String, self._on_text_received_cb)

    def _on_text_received_cb(self, msg):
        if self._audio_pub.is_filtering_all_messages:
            return

        mp3_audio_content = self._generate_mp3_audio_content(msg.data)
        file_path = self._write_mp3_audio_content(mp3_audio_content)
        self._play_audio(file_path)
        self._done_talking_pub.publish(Empty())

    def _generate_mp3_audio_content(self, text):
        language_code = self._convert_language_to_language_code(self._language)

        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code=language_code, ssml_gender=texttospeech.SsmlVoiceGender.MALE)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return response.audio_content

    def _convert_language_to_language_code(self, language):
        if language == 'en':
            return 'en-US'
        elif language == 'fr':
            return 'fr-CA'

    def _write_mp3_audio_content(self, mp3_audio_content):
        directory_path = os.path.join(self._pkg_path, 'audio_files')
        file_path = os.path.join(directory_path, 'text.mp3')
        os.makedirs(directory_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            file.write(mp3_audio_content)

        return file_path

    def _play_audio(self, file_path):
        frames, sample_rate = self._load_frames(file_path)

        if sample_rate != SUPPORTED_SAMPLE_RATE:
            rospy.logerr("Sample rate is not equal to " + str(SUPPORTED_SAMPLE_RATE) + ".\nSound will not be played")
            return

        global_energy_filter_sos, global_energy_filter_zi = self._initialize_global_energy_filter(sample_rate)
        current_energy_filter_sos, current_energy_filter_zi = self._initialize_current_energy_filter(sample_rate)
        mouth_signal_filter_sos, mouth_signal_filter_zi = self._initialize_mouth_signal_filter(sample_rate)

        mouth_signal_msg = Float32()
        audio_frame = AudioFrame()
        audio_frame.format = 'float'
        audio_frame.channel_count = 1
        audio_frame.sampling_frequency = SUPPORTED_SAMPLE_RATE
        audio_frame.frame_sample_count = self._frame_sample_count

        rate = rospy.Rate(SUPPORTED_SAMPLE_RATE / self._frame_sample_count)
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

            audio_frame.data = frame.tobytes()
            self._audio_pub.publish(audio_frame)

            rate.sleep()

        mouth_signal_msg.data = 0.0
        self._mouth_signal_scale_pub.publish(mouth_signal_msg)

    def _load_frames(self, file_path):
        waveform, sample_rate = librosa.load(file_path)
        frames = np.split(waveform, np.arange(self._frame_sample_count, len(waveform), self._frame_sample_count))
        return frames, sample_rate

    def _initialize_global_energy_filter(self, sample_rate):
        global_energy_filter_sos =  signal.butter(1, 2, btype='lowpass', fs=sample_rate, output='sos')
        global_energy_filter_zi = np.zeros((global_energy_filter_sos.shape[0], 2))
        return global_energy_filter_sos, global_energy_filter_zi

    def _initialize_current_energy_filter(self, sample_rate):
        current_energy_filter_sos =  signal.butter(1, 10, btype='lowpass', fs=sample_rate, output='sos')
        current_energy_filter_zi = np.zeros((current_energy_filter_sos.shape[0], 2))
        return current_energy_filter_sos, current_energy_filter_zi

    def _initialize_mouth_signal_filter(self, sample_rate):
        mouth_signal_filter_sos =  signal.butter(1, 2.5, btype='lowpass', fs=sample_rate // self._frame_sample_count, output='sos')
        mouth_signal_filter_zi = np.zeros((mouth_signal_filter_sos.shape[0], 2))
        return mouth_signal_filter_sos, mouth_signal_filter_zi

    def run(self):
        rospy.spin()
        self._pyaudio.terminate()


def main():
    rospy.init_node('talk_node')
    talk_node = TalkNode()
    talk_node.run()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
