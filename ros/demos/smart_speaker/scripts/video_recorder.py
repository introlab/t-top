#!/usr/bin/env python3

import rospy

from audio_utils.msg import AudioFrame
from sensor_msgs.msg import Image

import time
import inspect

from cv_bridge import CvBridge, CvBridgeError
import cv2

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


class Recorder:
    def __init__(self, width: int, height: int, filename: str):
        self._width = width
        self._height = height
        self._filename = filename
        self._encode_audio_rate = 44100
        self._record_start_time_ns = int(time.time() * 1e9)
        self._last_video_frame_timestamp_ns = int(time.time() * 1e9)
        self._pipeline = None
        self._video_src = None
        self._audio_src = None

    def start_recording(self):
        if self._pipeline is None:
            video_caps = f'video/x-raw,format=RGB,width={self._width},height={self._height},framerate=1/1'
            audio_caps = f'audio/x-raw,format=S16LE,channels=1,rate={self._encode_audio_rate},layout=interleaved'
            command = f'appsrc name=video_src emit-signals=True  is-live=True format=time caps={video_caps} ! ' \
                f'queue max-size-buffers=100 ! ' \
                f'videoconvert ! ' \
                f'videoscale ! ' \
                f'videorate ! ' \
                f'capsfilter caps=video/x-raw,framerate=24/1 ! ' \
                f'x264enc tune=zerolatency ! ' \
                f'capsfilter caps=video/x-h264,profile=high ! ' \
                f'mp4mux name=mux !' \
                f'filesink location={self._filename} ' \
                f'appsrc name=audio_src is-live=True format=time caps={audio_caps} !' \
                f'queue max-size-buffers=100 ! ' \
                f'audioconvert ! ' \
                f'audioresample ! ' \
                f'audiorate ! ' \
                f'avenc_aac !' \
                f'mux.'
            try:
                self._pipeline = Gst.parse_launch(command)
                self._record_start_time_ns = int(time.time() * 1e9)
                self._last_video_frame_timestamp_ns = 0
                self._video_src = self._pipeline.get_by_name('video_src')
                self._audio_src = self._pipeline.get_by_name('audio_src')
                self._pipeline.set_state(Gst.State.PLAYING)
            except gi.repository.GLib.Error as e:
                print(f'recorder Error: {e}')


    def stop_recording(self):

        print(f'recorder - stop')
        if self._pipeline:
            self._video_src.emit("end-of-stream")
            self._audio_src.emit("end-of-stream")
            bus = self._pipeline.get_bus()
            msg = bus.timed_pop_filtered(
                Gst.CLOCK_TIME_NONE,
                Gst.MessageType.ERROR | Gst.MessageType.EOS
            )
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            self._video_src = None
            self._audio_src = None


    @staticmethod
    def numpy_to_gst_buffer(image, timestamp_ns, duration_ns=0):
        buffer = Gst.Buffer.new_wrapped(image.data)
        buffer.pts = int(timestamp_ns)
        buffer.duration = int(duration_ns)
        return buffer


    def push_audio_frame(self, frame: AudioFrame):
        if self._pipeline is None:
            self.start_recording()

        if self._audio_src:
            timestamp_ns = int(time.time() *  1e9) - self._record_start_time_ns
            self._audio_src.emit("push-buffer", Recorder.numpy_to_gst_buffer(frame, timestamp_ns, 1/self._encode_audio_rate * frame.frame_sample_count))



    def push_video_frame(self, frame: Image):
        if self._pipeline is None:
            self.start_recording()

        timestamp_ns = int(time.time() * 1e9 - self._record_start_time_ns)
        duration = timestamp_ns - self._last_video_frame_timestamp_ns
        self._last_video_frame_timestamp_ns = timestamp_ns

         # Convert to GST Buffer and send to encoder
        self._video_src.emit("push-buffer", Recorder.numpy_to_gst_buffer(frame, timestamp_ns, duration))


# Recorder
recorder = Recorder(1280,720, '/tmp/test.mp4')


def image_callback(image: Image):
    # print('image callback', image.width, image.height)
    recorder.push_video_frame(image)


def audio_callback(frame: AudioFrame):
    # print('audio callback',frame.format, frame.sampling_frequency)
    recorder.push_audio_frame(frame)


def on_shutdown():
    recorder.stop_recording()


if __name__ == '__main__':

    print('init')
    rospy.init_node('recorder')
    Gst.init(None)
    Gst.debug_set_active(True)
    Gst.debug_set_default_threshold(3)

    image_topic = '/camera/color/image_raw'
    audio_topic = '/audio_signed_16_44100'

    rospy.Subscriber(image_topic, Image, image_callback)
    rospy.Subscriber(audio_topic, AudioFrame, audio_callback)


    rospy.spin()

    rospy.on_shutdown(on_shutdown)

    # Stopping recording
    recorder.stop_recording()



