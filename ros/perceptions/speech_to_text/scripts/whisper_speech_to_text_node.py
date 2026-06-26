#!/usr/bin/env python3

import queue
import threading
import datetime

import numpy as np

from faster_whisper import WhisperModel

import rclpy
import rclpy.node

from perception_msgs.msg import Transcript

from audio_utils import get_format_information, convert_audio_data_to_numpy_frames
from audio_utils_msgs.msg import AudioFrame, VoiceActivity, CompleteUtterance

import hbba_lite


SUPPORTED_LANGUAGES = {'en', 'fr'}
SUPPORTED_CHANNEL_COUNT = 1
SUPPORTED_SAMPLING_FREQUENCY = 16000


class WhisperSpeechToTextNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('whisper_speech_to_text_node')

        self._language = self.declare_parameter('language', 'en').get_parameter_value().string_value
        self._model_size = self.declare_parameter('model_size', 'base.en').get_parameter_value().string_value
        self._device = self.declare_parameter('device', 'cpu').get_parameter_value().string_value
        self._compute_type = self.declare_parameter('compute_type', 'float32').get_parameter_value().string_value

        self._prebuffering_frame_count = self.declare_parameter('prebuffering_frame_count', 4).get_parameter_value().integer_value
        self._minimum_voice_sequence_size = self.declare_parameter('minimum_voice_sequence_size', 8000).get_parameter_value().integer_value

        if self._language not in SUPPORTED_LANGUAGES:
            raise ValueError(f'Invalid language ({self._language})')

        self._model = WhisperModel(self._model_size, device=self._device, compute_type=self._compute_type)

        self._is_voice = False
        self._is_complete_sentence = False
        self._frames = []
        self._pending_frames = []
        self._voice_sequence_queue = queue.Queue()

        self._text_pub = self.create_publisher(Transcript, 'transcript', 10)
        self._voice_activity_sub = self.create_subscription(VoiceActivity, 'voice_activity', self._voice_activity_cb, 10)
        self._semantic_analysis_sub = self.create_subscription(CompleteUtterance, 'semantic_analysis', self._semantic_analysis_cb, 10)
        self._audio_sub = hbba_lite.OnOffHbbaSubscriber(self, AudioFrame, 'audio_in', self._audio_cb, 10)
        self._audio_sub.on_filter_state_changed(self._filter_state_changed_cb)

    def _voice_activity_cb(self, msg):
        """
        Callback for incoming voice activity detection (VAD) messages.

        Detects falling edges of the voice activity signal. On a falling edge,
        snapshots the current audio frames into the pending buffer and clears
        the active frame buffer.

        Args:
            msg: Incoming VAD message containing the following fields:
                - is_voice (bool): Whether voice activity is currently detected.

        Returns:
            None

        Side Effects:
            - Updates `self._is_voice`.
            - Extends `self._pending_frames` with the current `self._frames`
            contents on a falling edge.
            - Clears `self._frames` on a falling edge.
        """        
        last_is_voice = self._is_voice
        self._is_voice = msg.is_voice

        if last_is_voice and not self._is_voice:
            # Snapshot frames at the exact moment VAD drops,
            # before _audio_cb starts trimming them
            if len(self._frames) > 0:
                self._pending_frames.extend(self._frames)
            self._frames.clear()

    def _semantic_analysis_cb(self, msg):
        """
        Callback for incoming semantic analysis messages.

        Updates the sentence completion state and triggers queuing of pending
        audio frames when a complete sentence is detected.

        Args:
            msg: Incoming semantic analysis message containing the following fields:
                - sentence_complete (bool): Whether the current utterance is
                considered a complete sentence.

        Returns:
            None

        Side Effects:
            - Updates `self._is_complete_sentence`.
            - Calls `_put_frames_in_voice_sequence_queue` when
            `msg.sentence_complete` is True.
        """
        self._is_complete_sentence = msg.sentence_complete

        if msg.sentence_complete:
            self._put_frames_in_voice_sequence_queue()

    def _audio_cb(self, msg):
        """
        Callback for incoming audio messages.

        Validates the audio frame format, converts the data to a float32 numpy
        frame, and appends it to the active frame buffer. When voice activity is
        not detected, the buffer is trimmed to the configured pre-buffering size
        to limit memory usage.

        Args:
            msg: Incoming audio message containing the following fields:
                - channel_count (int): Number of audio channels.
                - sampling_frequency (int): Sample rate in Hz.
                - format: Audio sample format descriptor.
                - data (bytes): Raw audio data.

        Returns:
            None

        Side Effects:
            - Logs an error and returns early if the audio frame format is invalid.
            - Appends a float32 frame to `self._frames`.
            - Trims `self._frames` to `self._prebuffering_frame_count` when
            voice is inactive and the buffer exceeds that size.
        """
        if msg.channel_count != SUPPORTED_CHANNEL_COUNT or msg.sampling_frequency != SUPPORTED_SAMPLING_FREQUENCY:
            self.get_logger().error('Invalid audio frame (msg.channel_count={}, msg.sampling_frequency={}})'
                         .format(msg.channel_count, msg.sampling_frequency))
            return

        input_format_information = get_format_information(msg.format)
        frame = convert_audio_data_to_numpy_frames(input_format_information, msg.channel_count, msg.data)[0]

        self._frames.append(frame.astype(np.float32))
        if not self._is_voice and len(self._frames) > self._prebuffering_frame_count:
            self._frames = self._frames[-self._prebuffering_frame_count:]

    def _filter_state_changed_cb(self, previous_is_filtering_all_messages, new_is_filtering_all_messages):
        """
        Callback invoked when the HBBA filter state changes.

        When filtering transitions from inactive to active, resets the voice
        activity state and flushes any pending audio frames into the voice
        sequence queue.

        Args:
            previous_is_filtering_all_messages (bool): Filter state before the change.
            new_is_filtering_all_messages (bool): Filter state after the change.

        Returns:
            None

        Side Effects:
            - Resets `self._is_voice` to False on a low-to-high filter transition.
            - Calls `_put_frames_in_voice_sequence_queue` on a low-to-high
            filter transition.
        """
        if not previous_is_filtering_all_messages and new_is_filtering_all_messages:
            self._is_voice = False
            self._put_frames_in_voice_sequence_queue()

    def _put_frames_in_voice_sequence_queue(self):
        """
        Concatenates pending audio frames and enqueues them for transcription.

        If there are pending frames, merges them into a single numpy array and
        places it on the voice sequence queue, then clears the pending buffer.

        Returns:
            None

        Side Effects:
            - Enqueues a concatenated numpy array to `self._voice_sequence_queue`
            if `self._pending_frames` is non-empty.
            - Clears `self._pending_frames`.
        """
        if  len(self._pending_frames) > 0:
            self._voice_sequence_queue.put(np.concatenate(self._pending_frames))
            self._pending_frames.clear()

    def run(self):
        speech_to_text_thread = threading.Thread(target=self._speech_to_text_thread_run)
        speech_to_text_thread.start()

        try:
            rclpy.spin(self)
        finally:
            self._voice_sequence_queue.put(None)
            speech_to_text_thread.join()

    def _speech_to_text_thread_run(self):
        """
        Worker thread that transcribes queued voice sequences using Whisper.

        Warms up the model on startup, then continuously dequeues audio from
        the voice sequence queue and runs transcription. Sequences that are
        too short are discarded. Publishes the resulting transcript with timing
        metadata after each successful transcription.

        Stops when a None sentinel is dequeued or `rclpy.ok()` returns False.

        Returns:
            None

        Side Effects:
            - Calls `_warm_up_model` on startup.
            - Publishes a `Transcript` message via `self._text_pub` for each
            transcribed voice sequence.
        """
        self._warm_up_model()

        while rclpy.ok():
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
        """
        Runs a silent inference pass to warm up the Whisper model.

        Transcribes one second of silence to pre-initialize internal model state
        and avoid latency on the first real transcription request.

        Returns:
            None
        """
        audio = np.zeros(SUPPORTED_SAMPLING_FREQUENCY, dtype=np.float32)
        segments, _ = self._model.transcribe(audio, beam_size=1, best_of=1, temperature=0.0, language=self._language)
        for _ in segments:
            pass


def main():
    rclpy.init()
    whisper_speech_to_text_node = WhisperSpeechToTextNode()

    try:
        whisper_speech_to_text_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        whisper_speech_to_text_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
