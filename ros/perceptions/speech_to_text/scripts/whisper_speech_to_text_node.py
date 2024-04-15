#!/usr/bin/env python3

import queue
import threading
import datetime

import numpy as np

from faster_whisper import WhisperModel

import rospy
from speech_to_text.msg import Transcript

from audio_utils import get_format_information, convert_audio_data_to_numpy_frames
from audio_utils.msg import AudioFrame, VoiceActivity

import hbba_lite


SUPPORTED_LANGUAGES = {'en', 'fr'}
SUPPORTED_CHANNEL_COUNT = 1
SUPPORTED_SAMPLING_FREQUENCY = 16000


class WhisperSpeechToTextNode:
    def __init__(self):
        self._language = rospy.get_param('~language')
        self._model_size = rospy.get_param('~model_size')
        self._device = rospy.get_param('~device')
        self._compute_type = rospy.get_param('~compute_type')

        self._prebuffering_frame_count = rospy.get_param('~prebuffering_frame_count', 4)
        self._minimum_voice_sequence_size = rospy.get_param('~minimum_voice_sequence_size', 8000)

        if self._language not in SUPPORTED_LANGUAGES:
            raise ValueError(f'Invalid language ({self._language})')

        self._model = WhisperModel(self._model_size, device=self._device, compute_type=self._compute_type)

        self._is_voice = False
        self._frames_lock = threading.Lock()
        self._frames = []
        self._voice_sequence_queue = queue.Queue()

        rospy.on_shutdown(self._shutdown_cb)

        self._text_pub = rospy.Publisher('transcript', Transcript, queue_size=10)
        self._voice_activity_pub = rospy.Subscriber('voice_activity', VoiceActivity, self._voice_activity_cb, queue_size=10)
        self._audio_sub = hbba_lite.OnOffHbbaSubscriber('audio_in', AudioFrame, self._audio_cb, queue_size=10)
        self._audio_sub.on_filter_state_changed(self._filter_state_changed_cb)

    def _voice_activity_cb(self, msg):
        last_is_voice = self._is_voice
        self._is_voice = msg.is_voice

        if last_is_voice and not self._is_voice:
            self._put_frames_in_voice_sequence_queue()

    def _audio_cb(self, msg):
        if msg.channel_count != SUPPORTED_CHANNEL_COUNT or msg.sampling_frequency != SUPPORTED_SAMPLING_FREQUENCY:
            rospy.logerr('Invalid audio frame (msg.channel_count={}, msg.sampling_frequency={}})'
                         .format(msg.channel_count, msg.sampling_frequency))
            return

        input_format_information = get_format_information(msg.format)
        frame = convert_audio_data_to_numpy_frames(input_format_information, msg.channel_count, msg.data)[0]

        with self._frames_lock:
            self._frames.append(frame.astype(np.float32))
            if not self._is_voice and len(self._frames) > self._prebuffering_frame_count:
                self._frames = self._frames[-self._prebuffering_frame_count:]

    def _filter_state_changed_cb(self, previous_is_filtering_all_messages, new_is_filtering_all_messages):
        if not previous_is_filtering_all_messages and new_is_filtering_all_messages:
            self._is_voice = False
            self._put_frames_in_voice_sequence_queue()

    def _put_frames_in_voice_sequence_queue(self):
        with self._frames_lock:
            if len(self._frames) > 0:
                self._voice_sequence_queue.put(np.concatenate(self._frames))
                self._frames.clear()

    def _shutdown_cb(self):
        self._voice_sequence_queue.put(None)

    def run(self):
        self._warm_up_model()

        while not rospy.is_shutdown():
            voice_sequence = self._voice_sequence_queue.get()
            if voice_sequence is None:
                break
            elif voice_sequence.shape[0] < self._minimum_voice_sequence_size:
                # Residual audio is flushed.
                continue

            start_timestamp = datetime.datetime.now()
            segments, _ = self._model.transcribe(voice_sequence,
                                                 beam_size=1, best_of=1, temperature=0.0, language=self._language)
            end_timestamp = datetime.datetime.now()

            msg = Transcript()
            msg.text = ' '.join((segment.text for segment in segments))
            msg.is_final = True
            msg.processing_time_s = (end_timestamp - start_timestamp).total_seconds()
            msg.total_samples_count = voice_sequence.shape[0]
            self._text_pub.publish(msg)

    def _warm_up_model(self):
        audio = np.zeros(SUPPORTED_SAMPLING_FREQUENCY, dtype=np.float32)
        segments, _ = self._model.transcribe(audio, beam_size=1, best_of=1, temperature=0.0, language=self._language)
        for _ in segments:
            pass


def main():
    rospy.init_node('whisper_speech_to_text_node')
    whisper_speech_to_text_node = WhisperSpeechToTextNode()
    whisper_speech_to_text_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
