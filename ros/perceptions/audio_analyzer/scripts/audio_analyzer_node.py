#!/usr/bin/env python3

import threading
from datetime import datetime
import numpy as np

import torch

import rclpy
import rclpy.node

from audio_utils_msgs.msg import AudioFrame
from perception_msgs.msg import AudioAnalysis
from odas_ros_msgs.msg import OdasSstArrayStamped

from dnn_utils import MulticlassAudioDescriptorExtractor, VoiceDescriptorExtractor
import hbba_lite


SUPPORTED_AUDIO_FORMAT = 'signed_16'
SUPPORTED_CHANNEL_COUNT = 1


class AudioAnalyzerNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('audio_analyzer_node')

        self._inference_type = self.declare_parameter('inference_type', 'cpu').get_parameter_value().string_value

        self._audio_descriptor_extractor = MulticlassAudioDescriptorExtractor(inference_type=self._inference_type)
        self._voice_descriptor_extractor = VoiceDescriptorExtractor(inference_type=self._inference_type)

        if self._audio_descriptor_extractor.get_supported_sampling_frequency() != self._voice_descriptor_extractor.get_supported_sampling_frequency():
            raise ValueError('Not compatible models (sampling frequency)')
        self._supported_sampling_frequency = self._audio_descriptor_extractor.get_supported_sampling_frequency()

        self._audio_analysis_interval = self.declare_parameter('interval', 16000).get_parameter_value().integer_value
        self._voice_probability_threshold = self.declare_parameter('voice_probability_threshold', 0.5).get_parameter_value().double_value
        self._class_probability_threshold = self.declare_parameter('class_probability_threshold', 0.5).get_parameter_value().double_value

        self._audio_buffer_duration = max(self._audio_descriptor_extractor.get_supported_duration(),
                                          self._voice_descriptor_extractor.get_supported_duration(),
                                          self._audio_analysis_interval)

        self._class_names = self._audio_descriptor_extractor.get_class_names()
        self._voice_class_index = self._class_names.index('Human_voice')

        self._audio_frames_lock = threading.Lock()
        self._audio_frames = []
        self._audio_analysis_count = 0

        self._audio_direction_lock = threading.Lock()
        self._audio_direction = ('', 0.0, 0.0, 0.0)

        self._audio_analysis_pub = self.create_publisher(AudioAnalysis, 'audio_analysis', 10)

        self._sst_id = -1

        self._hbba_filter_state = hbba_lite.OnOffHbbaFilterState(self, 'audio_in/filter_state')
        self._audio_sub = self.create_subscription(AudioFrame, 'audio_in', self._audio_cb, 100)

        self._sst_sub = self.create_subscription(OdasSstArrayStamped, 'sst', self._sst_cb, 1)

    def _audio_cb(self, msg):
        if msg.format != SUPPORTED_AUDIO_FORMAT or \
                msg.channel_count != SUPPORTED_CHANNEL_COUNT or \
                msg.sampling_frequency != self._supported_sampling_frequency:
            self.get_logger().error('Invalid audio frame (msg.format={}, msg.channel_count={}, msg.sampling_frequency={}})'
                .format(msg.format, msg.channel_count, msg.sampling_frequency))
            return
        if self._hbba_filter_state.is_filtering_all_messages:
            self._audio_analysis_count = 0
            self._audio_frames.clear()
            return

        with torch.no_grad():
            with self._audio_frames_lock:
                audio_frame = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / -np.iinfo(np.int16).min
                self._audio_frames.append(torch.from_numpy(audio_frame))
                if (len(self._audio_frames) - 1) * audio_frame.shape[0] >= self._audio_buffer_duration:
                    self._audio_frames.pop(0)

            if self._audio_analysis_count >= self._audio_analysis_interval:
                self._audio_analysis_count = 0
                self._analyse()
            else:
                self._audio_analysis_count += audio_frame.shape[0]

    def _analyse(self):
        start_time = datetime.now()
        audio_buffer, sst_id = self._get_audio_buffer_and_sst_id()
        audio_descriptor_buffer = audio_buffer[-self._audio_descriptor_extractor.get_supported_duration():]
        audio_descriptor, audio_class_probabilities = self._audio_descriptor_extractor(audio_descriptor_buffer)
        audio_descriptor = audio_descriptor.tolist()

        if audio_class_probabilities[self._voice_class_index].item() >= self._voice_probability_threshold:
            voice_descriptor_buffer = audio_buffer[-self._voice_descriptor_extractor.get_supported_duration():]
            voice_descriptor = self._voice_descriptor_extractor(voice_descriptor_buffer).tolist()
        else:
            voice_descriptor = []

        audio_classes = self._get_audio_classes(audio_class_probabilities)
        processing_time_s = (datetime.now() - start_time).total_seconds()
        self._publish_audio_analysis(sst_id, audio_buffer, audio_classes, audio_descriptor, voice_descriptor, processing_time_s)

    def _get_audio_buffer_and_sst_id(self):
        with self._audio_frames_lock:
            sst_id = self._sst_id
            audio_buffer = torch.cat(self._audio_frames, dim=0)
        if audio_buffer.size()[0] < self._audio_buffer_duration:
            return torch.cat([torch.zeros(self._audio_buffer_duration - audio_buffer.size()[0]), audio_buffer], dim=0), sst_id
        else:
            return audio_buffer[-self._audio_buffer_duration:], sst_id

    def _get_audio_classes(self, audio_class_probabilities):
        return [self._class_names[i] for i in range(len(self._class_names))
                if audio_class_probabilities[i].item() >= self._class_probability_threshold]

    def _publish_audio_analysis(self, sst_id, audio_buffer, audio_classes, audio_descriptor, voice_descriptor, processing_time_s=0):
        with self._audio_direction_lock:
            frame_id, direction_x, direction_y, direction_z = self._audio_direction

        if frame_id == '':
            return

        msg = AudioAnalysis()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        msg.tracking_id = sst_id

        msg.audio_frame.format = 'float'
        msg.audio_frame.channel_count = SUPPORTED_CHANNEL_COUNT
        msg.audio_frame.sampling_frequency = self._supported_sampling_frequency
        msg.audio_frame.frame_sample_count = audio_buffer.numel()
        msg.audio_frame.data = audio_buffer.cpu().detach().numpy().tobytes()

        msg.audio_classes = audio_classes
        msg.audio_descriptor = audio_descriptor
        msg.voice_descriptor = voice_descriptor

        msg.direction_x = direction_x
        msg.direction_y = direction_y
        msg.direction_z = direction_z

        msg.processing_time_s = processing_time_s

        self._audio_analysis_pub.publish(msg)

    def _sst_cb(self, sst):
        if len(sst.sources) == 0:
            return

        if len(sst.sources) > 1:
            self.get_logger().error('Invalid sst (len(sst.sources)={})'.format(len(sst.sources)))
            return

        if sst.sources[0].id != self._sst_id:
            self._sst_id = sst.sources[0].id
            with self._audio_frames_lock:
                self._audio_frames = []

        with self._audio_direction_lock:
            self._audio_direction = (sst.header.frame_id, sst.sources[0].x, sst.sources[0].y, sst.sources[0].z)

    def run(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    audio_analyzer_node = AudioAnalyzerNode()

    try:
        audio_analyzer_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        audio_analyzer_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
