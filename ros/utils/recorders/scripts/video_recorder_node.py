#!/usr/bin/env python3

import datetime
from enum import Enum, auto
import os
import threading

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import rospy

from audio_utils.msg import AudioFrame
from sensor_msgs.msg import Image

import hbba_lite


class VideoFormat(Enum):
    RGB = 'RGB'
    BGR = 'BGR'

    @staticmethod
    def from_string(x: str) -> 'VideoFormat':
        if x not in VideoFormat._STRING_TO_VIDEO_FORMAT:
            raise ValueError(f'Invalid video codec ({x})')
        else:
            return VideoFormat._STRING_TO_VIDEO_FORMAT[x]


VideoFormat._STRING_TO_VIDEO_FORMAT = {
    'rgb8': VideoFormat.RGB,
    'bgr8': VideoFormat.BGR
}


class VideoCodec(Enum):
    H264 = auto()
    H265 = auto()
    VP8 = auto()
    VP9 = auto()

    @staticmethod
    def from_string(x: str) -> 'VideoCodec':
        if x not in VideoCodec._STRING_TO_VIDEO_CODEC:
            raise ValueError(f'Invalid video codec ({x})')
        else:
            return VideoCodec._STRING_TO_VIDEO_CODEC[x]

    def to_software_encoder(self):
        return VideoCodec._VIDEO_CODEC_TO_SOFTWARE_ENCODER[self]

    def get_software_bitrate_attribute(self):
        if self == VideoCodec.H264 or self == VideoCodec.H265:
            return 'bitrate'
        elif self == VideoCodec.VP8 or self == VideoCodec.VP9:
            return 'target-bitrate'

    def to_nvidia_hardware_encoder(self):
        return VideoCodec._VIDEO_CODEC_TO_NVIDIA_HARDWARE_ENCODER[self]


VideoCodec._STRING_TO_VIDEO_CODEC = {
    'h264': VideoCodec.H264,
    'h265': VideoCodec.H265,
    'vp8': VideoCodec.VP8,
    'vp9': VideoCodec.VP9,
}


VideoCodec._VIDEO_CODEC_TO_SOFTWARE_ENCODER = {
    VideoCodec.H264: 'x264enc',
    VideoCodec.H265: 'x265enc',
    VideoCodec.VP8: 'vp8enc',
    VideoCodec.VP9: 'vp9enc',
}


VideoCodec._VIDEO_CODEC_TO_NVIDIA_HARDWARE_ENCODER = {
    VideoCodec.H264: 'nvv4l2h264enc',
    VideoCodec.H265: 'nvv4l2h265enc',
    VideoCodec.VP8: 'nvv4l2vp8enc',
    VideoCodec.VP9: 'nvv4l2vp9enc',
}


class AudioFormat(Enum):
    S8 = 'S8'
    S16LE = 'S16LE'
    S24LE = 'S24LE'
    S24_32LE = 'S24_32LE'
    S32LE = 'S32LE'
    U8 = 'U8'
    U16LE = 'U16LE'
    U24LE = 'U24LE'
    U24_32LE = 'U24_32LE'
    U32LE = 'U32LE'
    F32LE = 'F32LE'
    F64LE = 'F64LE'

    @staticmethod
    def from_string(x: str) -> 'AudioFormat':
        if x not in AudioFormat._STRING_TO_AUDIO_FORMAT:
            raise ValueError(f'Invalid audio format ({x})')
        else:
            return AudioFormat._STRING_TO_AUDIO_FORMAT[x]


AudioFormat._STRING_TO_AUDIO_FORMAT = {
    'signed_8': AudioFormat.S8,
    'signed_16': AudioFormat.S16LE,
    'signed_24': AudioFormat.S24LE,
    'signed_padded_24': AudioFormat.S24_32LE,
    'signed_32': AudioFormat.S32LE,
    'unsigned_8': AudioFormat.U8,
    'unsigned_16': AudioFormat.U16LE,
    'unsigned_24': AudioFormat.U24LE,
    'unsigned_padded_24': AudioFormat.U24_32LE,
    'unsigned_32': AudioFormat.U32LE,
    'float': AudioFormat.F32LE,
    'double': AudioFormat.F64LE
}


class AudioCodec(Enum):
    AAC = auto()
    MP3 = auto()

    @staticmethod
    def from_string(x: str) -> 'AudioCodec':
        if x not in AudioCodec._STRING_TO_AUDIO_CODEC:
            raise ValueError(f'Invalid audio codec ({x})')
        else:
            return AudioCodec._STRING_TO_AUDIO_CODEC[x]

    def to_encoder(self):
        return AudioCodec._AUDIO_CODEC_TO_ENCODER[self]


AudioCodec._STRING_TO_AUDIO_CODEC = {
    'aac': AudioCodec.AAC,
    'mp3': AudioCodec.MP3,
}


AudioCodec._AUDIO_CODEC_TO_ENCODER = {
    AudioCodec.AAC: 'voaacenc',
    AudioCodec.MP3: 'lamemp3enc'
}


class VideoRecorderConfiguration:
    def __init__(self, output_directory: str, filename_prefix: str,
                 video_format: VideoFormat, video_width: int, video_height: int,
                 video_codec: VideoCodec, video_bitrate: int, video_delay_s: float,
                 audio_format: AudioFormat, audio_channel_count: int,
                 audio_sampling_frequency: int, audio_codec: AudioCodec):
        self.output_directory = output_directory
        self.filename_prefix = filename_prefix

        self.video_format = video_format
        self.video_width = video_width
        self.video_height = video_height
        self.video_codec = video_codec
        self.video_bitrate = video_bitrate
        self.video_delay_s = video_delay_s
        self.video_delay_ns = int(video_delay_s * 1e9)

        self.audio_format = audio_format
        self.audio_channel_count = audio_channel_count
        self.audio_sampling_frequency = audio_sampling_frequency
        self.audio_codec = audio_codec

    @staticmethod
    def from_parameters() -> 'VideoRecorderConfiguration':
        return VideoRecorderConfiguration(
            rospy.get_param('~output_directory'),
            rospy.get_param('~filename_prefix'),

            VideoFormat.from_string(rospy.get_param('~video_format')),
            rospy.get_param('~video_width'),
            rospy.get_param('~video_height'),
            VideoCodec.from_string(rospy.get_param('~video_codec')),
            rospy.get_param('~video_bitrate'),
            rospy.get_param('~video_delay_s'),

            AudioFormat.from_string(rospy.get_param('~audio_format')),
            rospy.get_param('~audio_channel_count'),
            rospy.get_param('~audio_sampling_frequency'),
            AudioCodec.from_string(rospy.get_param('~audio_codec'))
        )


class VideoRecorder:
    def __init__(self, configuration: VideoRecorderConfiguration):
        self._configuration = configuration
        os.makedirs(self._configuration.output_directory, exist_ok=True)

        self._record_start_time_ns = 0
        self._last_video_frame_timestamp_ns = 0

        self._pipeline = None
        self._video_src = None
        self._audio_src = None

        self._image_sub = rospy.Subscriber('image', Image, self._image_cb)
        self._audio_sub = rospy.Subscriber('audio', AudioFrame, self._audio_cb)

    def close(self):
        self._image_sub.unregister()
        self._audio_sub.unregister()
        self._stop_if_started()

    def _image_cb(self, msg: Image):
        if not VideoRecorder._verify_image_msg(self._configuration, msg):
            rospy.logerr(f'Invalid image (encoding={msg.encoding}, width={msg.width}, height={msg.height})')
            return

        msg_timestamp_ns = msg.header.stamp.to_nsec()
        self._start_if_not_started(msg_timestamp_ns)
        if self._video_src is None:
            return

        timestamp_ns = msg_timestamp_ns - self._record_start_time_ns + self._configuration.video_delay_ns
        duration_ns = max(0,timestamp_ns - self._last_video_frame_timestamp_ns)
        self._last_video_frame_timestamp_ns = timestamp_ns

        if timestamp_ns >= 0:
            self._video_src.emit("push-buffer", VideoRecorder.data_to_gst_buffer(msg.data, timestamp_ns, duration_ns))

    @staticmethod
    def _verify_image_msg(configuration: VideoRecorderConfiguration, msg: Image):
        try:
            return (VideoFormat.from_string(msg.encoding) == configuration.video_format and
                    msg.width == configuration.video_width and
                    msg.height == configuration.video_height)
        except ValueError:
            return False

    def _audio_cb(self, msg: AudioFrame):
        if not VideoRecorder._verify_audio_frame_msg(self._configuration, msg):
            rospy.logerr(f'Invalid audio frame (format={msg.format}, channel_count={msg.channel_count}, sampling_frequency={msg.sampling_frequency})')
            return

        msg_timestamp_ns = msg.header.stamp.to_nsec()
        self._start_if_not_started(msg_timestamp_ns)
        if self._audio_src is None:
            return

        timestamp_ns = msg_timestamp_ns - self._record_start_time_ns
        duration_ns = 1 / msg.sampling_frequency * msg.frame_sample_count

        if timestamp_ns >= 0:
            self._audio_src.emit("push-buffer", VideoRecorder.data_to_gst_buffer(msg.data, timestamp_ns, duration_ns))

    @staticmethod
    def _verify_audio_frame_msg(configuration: VideoRecorderConfiguration, msg: AudioFrame):
        try:
            return (AudioFormat.from_string(msg.format) == configuration.audio_format and
                    msg.channel_count == configuration.audio_channel_count and
                    msg.sampling_frequency == configuration.audio_sampling_frequency)
        except ValueError:
            return False

    def _start_if_not_started(self, record_start_time_ns: int):
        if self._pipeline is not None:
            return

        self._record_start_time_ns = record_start_time_ns
        self._last_video_frame_timestamp_ns = 0

        mux_pipeline = VideoRecorder._create_mux_pipeline(self._configuration)
        video_pipeline = VideoRecorder._create_video_pipeline(self._configuration)
        audio_pipeline = VideoRecorder._create_audio_pipeline(self._configuration)

        try:
            pipeline = Gst.parse_launch(f'{video_pipeline} ! {mux_pipeline} {audio_pipeline} ! mux.')
            video_src = pipeline.get_by_name('video_src')
            audio_src = pipeline.get_by_name('audio_src')
            bus = pipeline.get_bus()
            bus.add_watch(0, self._on_bus_message_cb)
            pipeline.set_state(Gst.State.PLAYING)

            self._pipeline = pipeline
            self._video_src = video_src
            self._audio_src = audio_src
            self._bus = bus
        except gi.repository.GLib.Error as e:
            rospy.loginfo(f'GStreamer pipeline failed({e})')

    def _stop_if_started(self):
        if self._pipeline is None:
            return

        self._video_src.emit("end-of-stream")
        self._audio_src.emit("end-of-stream")
        self._bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)

        self._bus.remove_watch()
        self._pipeline.set_state(Gst.State.NULL)
        self._pipeline = None
        self._video_src = None
        self._audio_src = None
        self._bus = None

    def _on_bus_message_cb(self, bus, msg):
        rospy.log(f'Gstreamer bus message: {msg}')

    @staticmethod
    def _create_mux_pipeline(configuration: VideoRecorderConfiguration):
        path = os.path.join(configuration.output_directory, VideoRecorder._get_filename(configuration))
        if configuration.video_codec == VideoCodec.H264 or configuration.video_codec == VideoCodec.H265:
            attributes = 'reserved-bytes-per-sec=100 reserved-max-duration=20184000000000 reserved-moov-update-period=100000000'
            return f'qtmux name=mux {attributes} ! filesink location={path}.mp4'
        elif configuration.video_codec == VideoCodec.VP8 or configuration.video_codec == VideoCodec.VP9:
            return f'matroskamux name=mux ! filesink location={path}.mkv'

    @staticmethod
    def _get_filename(configuration: VideoRecorderConfiguration):
        now = datetime.datetime.utcfromtimestamp(rospy.Time.now().to_sec())
        return configuration.filename_prefix + now.strftime("%Y-%m-%d_%H-%M-%S")

    @staticmethod
    def _create_video_pipeline(configuration: VideoRecorderConfiguration):
        pipeline = VideoRecorder._create_video_input_pipeline(configuration)

        if VideoRecorder._verify_nvidia_hardware_encoder(configuration):
            encoder = configuration.video_codec.to_nvidia_hardware_encoder()
            pipeline += ' ! videoconvert ! nvvidconv ! capsfilter caps=video/x-raw(memory:NVMM),format=(string)I420'
            pipeline += f' ! {encoder} bitrate={configuration.video_bitrate}'
        else:
            rospy.logwarn('NVIDIA hardware encoder are not available.')

            encoder = configuration.video_codec.to_software_encoder()
            bitrate_attribute = configuration.video_codec.get_software_bitrate_attribute()
            pipeline += f' ! videoconvert ! capsfilter caps=video/x-raw,format=I420'
            pipeline += f' ! {encoder} {bitrate_attribute}={configuration.video_bitrate}'

        if configuration.video_codec == VideoCodec.H264:
            pipeline += ' ! h264parse'
        elif configuration.video_codec == VideoCodec.H265:
            pipeline += ' ! h265parse'

        return pipeline

    @staticmethod
    def _create_video_input_pipeline(configuration: VideoRecorderConfiguration):
        caps = f'video/x-raw,format={configuration.video_format.value}'
        caps += f',width={configuration.video_width},height={configuration.video_height}'
        return f'appsrc name=video_src emit-signals=True is-live=True format=time caps={caps} ! queue max-size-buffers=100'

    @staticmethod
    def _verify_nvidia_hardware_encoder(configuration: VideoRecorderConfiguration):
        return (Gst.ElementFactory.find('nvvidconv') is not None and
                Gst.ElementFactory.find(configuration.video_codec.to_nvidia_hardware_encoder()) is not None)

    @staticmethod
    def _create_audio_pipeline(configuration: VideoRecorderConfiguration):
        pipeline = VideoRecorder._create_audio_input_pipeline(configuration)

        audioconvert_attributes = ''
        if configuration.audio_channel_count > 1 :
            values = [f'(float){1 / configuration.audio_channel_count}'] * configuration.audio_channel_count
            audioconvert_attributes = 'mix-matrix="<<' + ','.join(values) + '>>"'

        encoder = configuration.audio_codec.to_encoder()
        pipeline += f' ! audioconvert {audioconvert_attributes}'
        pipeline += f' ! capsfilter caps=audio/x-raw,channels=1 ! audiorate ! {encoder}'
        return pipeline

    @staticmethod
    def _create_audio_input_pipeline(configuration: VideoRecorderConfiguration):
        channel_mask = ''
        if configuration.audio_channel_count > 2:
            channel_mask = ',channel-mask=(bitmask)0x0'

        caps = f'audio/x-raw,format={configuration.audio_format.value},channels={configuration.audio_channel_count}'
        caps += f',rate={configuration.audio_sampling_frequency},layout=interleaved{channel_mask}'
        return f'appsrc name=audio_src is-live=True format=time caps={caps} ! queue max-size-buffers=100'

    @staticmethod
    def data_to_gst_buffer(data: bytes, timestamp_ns: int, duration_ns: int):
        buffer = Gst.Buffer.new_wrapped(data)
        buffer.pts = int(timestamp_ns)
        buffer.duration = int(duration_ns)
        return buffer


class VideoRecorderNode:
    def __init__(self):
        self._recorder_lock = threading.Lock()
        self._recorder = None
        self._recorder_configuration = VideoRecorderConfiguration.from_parameters()

        self._filter_state = hbba_lite.OnOffHbbaFilterState('video_recorder/filter_state')
        self._filter_state.on_changed(self._on_filter_state_changed)

    def _on_filter_state_changed(self, _, next_is_filtering_all_messages):
        with self._recorder_lock:
            if next_is_filtering_all_messages and self._recorder is not None:
                self._recorder.close()
                self._recorder = None
            elif not next_is_filtering_all_messages and self._recorder is None:
                self._recorder = VideoRecorder(self._recorder_configuration)

    def run(self):
        rospy.spin()

        with self._recorder_lock:
            if self._recorder is not None:
                self._recorder.close()
                self._recorder = None


def main():
    rospy.init_node('video_recorder_node')

    Gst.init(None)

    video_recorder_node = VideoRecorderNode()
    video_recorder_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
