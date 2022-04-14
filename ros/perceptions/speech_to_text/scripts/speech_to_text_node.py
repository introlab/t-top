#!/usr/bin/env python3

import queue
import threading
import time

import numpy as np

import rospy
from std_msgs.msg import String
from audio_utils.msg import AudioFrame
import hbba_lite

from google.cloud import speech
from google.api_core import exceptions as core_exceptions


SUPPORTED_AUDIO_FORMAT = 'signed_16'
SUPPORTED_CHANNEL_COUNT = 1


class SpeechToTextNode:
    def __init__(self):
        self._sampling_frequency = rospy.get_param('~sampling_frequency', 16000)
        self._frame_sample_count = rospy.get_param('~frame_sample_count', 92)
        self._request_frame_count = rospy.get_param('~request_frame_count', 20)
        self._language_code = self._convert_language_to_language_code(rospy.get_param('~language'))

        self._sleeping_duration = (self._request_frame_count *
                                   self._frame_sample_count / self._sampling_frequency)

        self._is_enabled = False

        self._buffer_lock = threading.Lock()
        self._buffer = self._create_request_frame_buffer()
        self._current_buffer_index = 0
        self._request_frame_queue = queue.Queue()

        rospy.on_shutdown(self._shutdown_cb)

        self._text_pub = rospy.Publisher('text', String, queue_size=10)
        self._audio_sub = hbba_lite.OnOffHbbaSubscriber('audio_in', AudioFrame, self._audio_cb, queue_size=10)
        self._audio_sub.on_filter_state_changed(self._filter_state_changed_cb)

        self._speech_client = speech.SpeechClient()

    def _convert_language_to_language_code(self, language):
        if language == 'en':
            return 'en-US'
        elif language == 'fr':
            return 'fr-CA'

    def _create_request_frame_buffer(self):
        return np.zeros(self._frame_sample_count * self._request_frame_count, dtype=np.int16)

    def _audio_cb(self, msg):
        if msg.format != SUPPORTED_AUDIO_FORMAT or \
                msg.channel_count != SUPPORTED_CHANNEL_COUNT or \
                msg.sampling_frequency != self._sampling_frequency or \
                msg.frame_sample_count != self._frame_sample_count:
            rospy.logerr('Invalid audio frame (msg.format={}, msg.channel_count={}, msg.sampling_frequency={}, msg.frame_sample_count={}})'
                         .format(msg.format, msg.channel_count, msg.sampling_frequency, msg.frame_sample_count))
            return

        with self._buffer_lock:
            audio_frame = np.frombuffer(msg.data, dtype=np.int16)
            self._buffer[self._current_buffer_index:self._current_buffer_index +
                         self._frame_sample_count] = audio_frame

            self._current_buffer_index += self._frame_sample_count

            if self._current_buffer_index >= self._buffer.shape[0]:
                self._current_buffer_index = 0
                self._request_frame_queue.put(self._buffer.copy())

    def _filter_state_changed_cb(self, previous_is_filtering_all_messages, new_is_filtering_all_messages):
        with self._buffer_lock:
            if previous_is_filtering_all_messages and not new_is_filtering_all_messages:
                self._current_buffer_index = 0
                while not self._request_frame_queue.empty():
                    self._request_frame_queue.get_nowait()
                self._is_enabled = True
            elif not previous_is_filtering_all_messages and new_is_filtering_all_messages:
                self._is_enabled = False
                self._request_frame_queue.put(None)

    def _shutdown_cb(self):
        self._filter_state_changed_cb(
            self._audio_sub.is_filtering_all_messages, True)

    def run(self):
        while not rospy.is_shutdown():
            if not self._is_enabled:
                time.sleep(self._sleeping_duration)
                continue

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self._sampling_frequency,
                language_code=self._language_code)
            streaming_config = speech.StreamingRecognitionConfig(config=config,
                                                                 single_utterance=False,
                                                                 interim_results=False)

            requests = self._request_frame_generator()
            responses = self._speech_client.streaming_recognize(config=streaming_config, requests=requests)
            for response in responses:
                if response.results and response.results[0].is_final:
                    msg = String()
                    msg.data = response.results[0].alternatives[0].transcript
                    self._text_pub.publish(msg)

    def _request_frame_generator(self):
        while self._is_enabled:
            audio_content = self._request_frame_queue.get()
            if audio_content is None:
                break

            yield speech.StreamingRecognizeRequest(audio_content=audio_content.tobytes())


def main():
    rospy.init_node('speech_to_text_node')
    speech_to_text_node = SpeechToTextNode()

    while not rospy.is_shutdown():
        try:
            speech_to_text_node.run()
        except (core_exceptions.InvalidArgument,
                core_exceptions.Unknown,
                core_exceptions.DeadlineExceeded,
                core_exceptions.OutOfRange) as e:
            rospy.logerr(f'speech_to_text_node has failed ({e})')


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
