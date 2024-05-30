#!/usr/bin/env python3

import queue
import threading
import time
import datetime
import numpy as np

import rclpy
import rclpy.node

from speech_to_text.msg import Transcript
from audio_utils.msg import AudioFrame
import hbba_lite

from google.cloud import speech
from google.api_core import exceptions as core_exceptions


SUPPORTED_AUDIO_FORMAT = 'signed_16'
SUPPORTED_CHANNEL_COUNT = 1


class GoogleSpeechToTextNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('google_speech_to_text_node')

        self._sampling_frequency = self.declare_parameter('sampling_frequency', 16000).get_parameter_value().integer_value
        self._frame_sample_count = self.declare_parameter('frame_sample_count', 92).get_parameter_value().integer_value
        self._request_frame_count = self.declare_parameter('request_frame_count', 20).get_parameter_value().integer_value

        language = self.declare_parameter('language', 'en').get_parameter_value().string_value
        self._language_code = self._convert_language_to_language_code(language)

        self._sleeping_duration = (self._request_frame_count *
                                   self._frame_sample_count / self._sampling_frequency)

        self._is_enabled = False

        self._buffer = self._create_request_frame_buffer()
        self._current_buffer_index = 0
        self._request_frame_queue = queue.Queue()
        self._total_samples_count = 0

        self._text_pub = self.create_publisher(Transcript, 'transcript', 10)
        self._audio_sub = hbba_lite.OnOffHbbaSubscriber(self, AudioFrame, 'audio_in', self._audio_cb, 10)
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
            self.get_logger().error(
                f'Invalid audio frame (msg.format={msg.format}, msg.channel_count={msg.channel_count}' +
                f', msg.sampling_frequency={msg.sampling_frequency}, msg.frame_sample_count={msg.frame_sample_count})')
            return

        audio_frame = np.frombuffer(msg.data, dtype=np.int16)
        self._buffer[self._current_buffer_index:self._current_buffer_index +
                        self._frame_sample_count] = audio_frame

        self._current_buffer_index += self._frame_sample_count

        if self._current_buffer_index >= self._buffer.shape[0]:
            self._current_buffer_index = 0
            self._request_frame_queue.put(self._buffer.copy())

    def _filter_state_changed_cb(self, previous_is_filtering_all_messages, new_is_filtering_all_messages):
        if previous_is_filtering_all_messages and not new_is_filtering_all_messages:
            self._current_buffer_index = 0
            while not self._request_frame_queue.empty():
                self._request_frame_queue.get_nowait()
            self._is_enabled = True
        elif not previous_is_filtering_all_messages and new_is_filtering_all_messages:
            self._is_enabled = False
            self._request_frame_queue.put(None)

    def run(self):
        speech_to_text_thread = threading.Thread(target=self._speech_to_text_thread_run)
        speech_to_text_thread.start()

        rclpy.spin(self)

        self._filter_state_changed_cb(self._audio_sub.is_filtering_all_messages, True)
        speech_to_text_thread.join()

    def _speech_to_text_thread_run(self):
        while rclpy.ok():
            if not self._is_enabled:
                time.sleep(self._sleeping_duration)
                continue

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self._sampling_frequency,
                language_code=self._language_code)
            streaming_config = speech.StreamingRecognitionConfig(config=config,
                                                                 single_utterance=False,
                                                                 interim_results=True)

            requests = self._request_frame_generator()
            start_timestamp = datetime.datetime.now()
            responses = self._speech_client.streaming_recognize(config=streaming_config, requests=requests)

            processing_time = 0

            for response in responses:
                processing_time += (datetime.datetime.now() - start_timestamp).total_seconds()
                if response.results:
                    msg = Transcript()
                    msg.text = response.results[0].alternatives[0].transcript
                    msg.is_final = response.results[0].is_final
                    msg.processing_time_s = processing_time
                    msg.total_samples_count = self._total_samples_count
                    self._text_pub.publish(msg)

                    # Reset samples and processing time when message is final
                    if msg.is_final:
                        self._total_samples_count = 0
                        processing_time = 0

                # Reset time for this iteration
                start_timestamp = datetime.datetime.now()

    def _request_frame_generator(self):
        self._total_samples_count = 0
        while self._is_enabled:
            audio_content = self._request_frame_queue.get()
            self._total_samples_count += audio_content.shape[0]
            if audio_content is None:
                break

            yield speech.StreamingRecognizeRequest(audio_content=audio_content.tobytes())


def main():
    rclpy.init()

    google_speech_to_text_node = GoogleSpeechToTextNode()

    try:
        while rclpy.ok():
            try:
                google_speech_to_text_node.run()
            except (core_exceptions.InvalidArgument,
                    core_exceptions.Unknown,
                    core_exceptions.DeadlineExceeded,
                    core_exceptions.OutOfRange) as e:
                google_speech_to_text_node.get_logger().error(f'google_speech_to_text_node has failed ({e})')
    except KeyboardInterrupt:
        pass
    finally:
        google_speech_to_text_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
