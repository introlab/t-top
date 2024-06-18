#!/usr/bin/env python3

import time
from pathlib import Path

import numpy as np

import librosa

import rclpy
import rclpy.node

from behavior_msgs.msg import SoundFile, SoundStarted, Done
from audio_utils_msgs.msg import AudioFrame

import hbba_lite


class SoundPlayerNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('sound_player_node')

        self._sampling_frequency = self.declare_parameter('sampling_frequency', 16000).get_parameter_value().integer_value
        self._frame_sample_count = self.declare_parameter('frame_sample_count', 1024).get_parameter_value().integer_value

        self._audio_pub = hbba_lite.OnOffHbbaPublisher(self, AudioFrame, 'audio_out', 5)
        self._started_pub = self.create_publisher(SoundStarted, 'sound_player/started', 5)
        self._done_pub = self.create_publisher(Done, 'sound_player/done', 5)

        self._file_sub = self.create_subscription(SoundFile, 'sound_player/file', self._on_file_received_cb, 1)

    def _on_file_received_cb(self, msg):
        if self._audio_pub.is_filtering_all_messages:
            return

        try:
            self._play_audio(msg.id, msg.path)
            ok = True
        except Exception as e:
            self.get_logger().error(f'Unable to play the sound ({e})')
            ok = False

        self._done_pub.publish(Done(id=msg.id, ok=ok))

    def _play_audio(self, id, path):
        frames = self._load_frames(Path(path).expanduser().resolve())

        self._started_pub.publish(Started(id=id))

        audio_frame = AudioFrame()
        audio_frame.format = 'float'
        audio_frame.channel_count = 1
        audio_frame.sampling_frequency = self._sampling_frequency
        audio_frame.frame_sample_count = self._frame_sample_count

        sleep_duration = self._frame_sample_count / self._sampling_frequency
        for frame in frames:
            if self._audio_pub.is_filtering_all_messages:
                break

            audio_frame.header.stamp = self.get_clock().now().to_msg()
            audio_frame.data = frame.tobytes()
            self._audio_pub.publish(audio_frame)

            time.sleep(sleep_duration)

    def _load_frames(self, file_path):
        waveform, _ = librosa.load(file_path, sr=self._sampling_frequency, res_type='kaiser_fast')
        waveform = librosa.to_mono(waveform)
        pad = (self._frame_sample_count - (waveform.shape[0] % self._frame_sample_count)) % self._frame_sample_count
        waveform.resize(waveform.shape[0] + pad, refcheck=False)
        frames = np.split(waveform, np.arange(self._frame_sample_count, len(waveform), self._frame_sample_count))
        return frames

    def run(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    sound_player_node = SoundPlayerNode()

    try:
        sound_player_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        sound_player_node.destroy_node()
        
    rclpy.shutdown()


if __name__ == '__main__':
    main()
