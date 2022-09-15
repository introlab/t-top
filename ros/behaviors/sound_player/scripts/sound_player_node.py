#!/usr/bin/env python3

import threading
from pathlib import Path

import numpy as np

import librosa

import rospy
from sound_player.msg import SoundFile, Started, Done
from audio_utils.msg import AudioFrame

import hbba_lite


class SoundPlayerNode:
    def __init__(self):
        self._sampling_frequency = rospy.get_param('~sampling_frequency')
        self._frame_sample_count = rospy.get_param('~frame_sample_count')

        self._audio_pub = hbba_lite.OnOffHbbaPublisher('audio_out', AudioFrame, queue_size=5)
        self._started_pub = rospy.Publisher('sound_player/started', Started, queue_size=5)
        self._done_pub = rospy.Publisher('sound_player/done', Done, queue_size=5)

        self._file_sub_lock = threading.Lock()
        self._file_sub = rospy.Subscriber('sound_player/file', SoundFile, self._on_file_received_cb, queue_size=1)

    def _on_file_received_cb(self, msg):
        with self._file_sub_lock:
            if self._audio_pub.is_filtering_all_messages:
                return

            try:
                self._play_audio(msg.id, msg.path)
                ok = True
            except Exception as e:
                rospy.logerr(f'Unable to play the sound ({e})')
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

        rate = rospy.Rate(self._sampling_frequency / self._frame_sample_count)
        for frame in frames:
            if self._audio_pub.is_filtering_all_messages:
                break

            audio_frame.header.stamp = rospy.Time.now()
            audio_frame.data = frame.tobytes()
            self._audio_pub.publish(audio_frame)

            rate.sleep()

    def _load_frames(self, file_path):
        waveform, _ = librosa.load(file_path, sr=self._sampling_frequency, res_type='kaiser_fast')
        pad = (self._frame_sample_count - (waveform.shape[0] % self._frame_sample_count)) % self._frame_sample_count
        waveform.resize(waveform.shape[0] + pad, refcheck=False)
        frames = np.split(waveform, np.arange(self._frame_sample_count, len(waveform), self._frame_sample_count))
        return frames

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('sound_player_node')
    sound_player_node = SoundPlayerNode()
    sound_player_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
