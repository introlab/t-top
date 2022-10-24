#!/usr/bin/env python3

import datetime
from enum import Enum
import os
import threading
from typing import Optional

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore
from gi.repository import GLib  # type: ignore

import rospy

from audio_utils.msg import AudioFrame
from sensor_msgs.msg import Image

import hbba_lite


class FormatEnum(str, Enum):
    gst_value: str

    def __new__(cls, value: str, gstreamer_value: str):
        obj = str.__new__(cls, value)
        obj._value_ = value

        obj.gst_value = gstreamer_value
        return obj


class VideoCodecEnum(str, Enum):
    sw_enc: str
    nv_hw_enc: Optional[str]
    gst_parser: str

    def __new__(
        cls,
        value: str,
        sw_enc: str,
        nv_hw_enc: Optional[str] = None,
        parser: Optional[str] = None,
    ):
        obj = str.__new__(cls, value)
        obj._value_ = value

        obj.sw_enc = sw_enc
        obj.nv_hw_enc = nv_hw_enc
        obj.gst_parser = f" ! {parser}" if parser is not None else ""
        return obj


class AudioCodecEnum(str, Enum):
    enc: str

    def __new__(cls, value: str, enc: str):
        obj = str.__new__(cls, value)
        obj._value_ = value

        obj.enc = enc
        return obj


class VideoFormat(FormatEnum):
    RGB = ("rgb8", "RGB")
    BGR = ("bgr8", "BGR")


class VideoCodec(VideoCodecEnum):
    H264 = ("h264", "x264enc", "nvv4l2h264enc", "h264parse")
    H265 = ("h265", "x265enc", "nvv4l2h265enc", "h265parse")
    VP8 = ("vp8", "vp8enc", "nvv4l2vp8enc")
    VP9 = ("vp9", "vp9enc", "nvv4l2vp9enc")

    def get_software_bitrate_attribute(self) -> str:
        if self == VideoCodec.H264 or self == VideoCodec.H265:
            return "bitrate"
        elif self == VideoCodec.VP8 or self == VideoCodec.VP9:
            return "target-bitrate"
        else:
            raise ValueError("Invalid video codec")

    def convert_software_bitrate(self, bitrate) -> int:
        if self == VideoCodec.H264 or self == VideoCodec.H265:
            return bitrate // 1000
        elif self == VideoCodec.VP8 or self == VideoCodec.VP9:
            return bitrate
        else:
            raise ValueError("Invalid video codec")

    def convert_nvidia_hardware_bitrate(self, bitrate) -> int:
        return bitrate


class AudioFormat(FormatEnum):
    S8 = ("signed_8", "S8")
    S16LE = ("signed_16", "S16LE")
    S24LE = ("signed_24", "S24LE")
    S24_32LE = ("signed_padded_24", "S24_32LE")
    S32LE = ("signed_32", "S32LE")
    U8 = ("unsigned_8", "U8")
    U16LE = ("unsigned_16", "U16LE")
    U24LE = ("unsigned_24", "U24LE")
    U24_32LE = ("unsigned_padded_24", "U24_32LE")
    U32LE = ("unsigned_32", "U32LE")
    F32LE = ("float", "F32LE")
    F64LE = ("double", "F64LE")


class AudioCodec(AudioCodecEnum):
    AAC = ("aac", "voaacenc")
    MP3 = ("mp3", "lamemp3enc")
    OPUS = ("opus", "opusenc")
    FLAC = ("flac", "flacenc")


class VideoRecorderConfiguration:
    def __init__(
        self,
        output_directory: str,
        filename_prefix: str,
        video_format: VideoFormat,
        video_width: int,
        video_height: int,
        video_codec: VideoCodec,
        video_bitrate: int,
        video_delay_s: float,
        audio_format: AudioFormat,
        audio_channel_count: int,
        audio_sampling_frequency: int,
        audio_codec: AudioCodec,
    ):
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
    def from_parameters() -> "VideoRecorderConfiguration":
        return VideoRecorderConfiguration(
            rospy.get_param("~output_directory"),  # type: ignore
            rospy.get_param("~filename_prefix"),  # type: ignore
            VideoFormat(rospy.get_param("~video_format")),
            rospy.get_param("~video_width"),  # type: ignore
            rospy.get_param("~video_height"),  # type: ignore
            VideoCodec(rospy.get_param("~video_codec")),
            rospy.get_param("~video_bitrate"),  # type: ignore
            rospy.get_param("~video_delay_s"),  # type: ignore
            AudioFormat(rospy.get_param("~audio_format")),
            rospy.get_param("~audio_channel_count"),  # type: ignore
            rospy.get_param("~audio_sampling_frequency"),  # type: ignore
            AudioCodec(rospy.get_param("~audio_codec")),
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

        self._image_sub = rospy.Subscriber("image", Image, self._image_cb)
        self._audio_sub = rospy.Subscriber("audio", AudioFrame, self._audio_cb)

    def close(self):
        self._image_sub.unregister()
        self._audio_sub.unregister()
        self._stop_if_started()

    def _image_cb(self, msg: Image):
        if not VideoRecorder._verify_image_msg(self._configuration, msg):
            rospy.logerr(
                f"Invalid image (encoding={msg.encoding}, width={msg.width}, height={msg.height})"
            )
            return

        msg_timestamp_ns = msg.header.stamp.to_nsec()
        self._start_if_not_started(msg_timestamp_ns)
        if self._video_src is None:
            return

        timestamp_ns = (
            msg_timestamp_ns
            - self._record_start_time_ns
            + self._configuration.video_delay_ns
        )
        duration_ns = max(0, timestamp_ns - self._last_video_frame_timestamp_ns)
        self._last_video_frame_timestamp_ns = timestamp_ns

        if timestamp_ns >= 0:
            self._video_src.emit(
                "push-buffer",
                VideoRecorder.data_to_gst_buffer(msg.data, timestamp_ns, duration_ns),
            )

    @staticmethod
    def _verify_image_msg(configuration: VideoRecorderConfiguration, msg: Image):
        try:
            return (
                VideoFormat(msg.encoding) == configuration.video_format
                and msg.width == configuration.video_width
                and msg.height == configuration.video_height
            )
        except ValueError:
            return False

    def _audio_cb(self, msg: AudioFrame):
        if not VideoRecorder._verify_audio_frame_msg(self._configuration, msg):
            rospy.logerr(
                f"Invalid audio frame (format={msg.format}, channel_count={msg.channel_count}, sampling_frequency={msg.sampling_frequency})"
            )
            return

        msg_timestamp_ns = msg.header.stamp.to_nsec()
        self._start_if_not_started(msg_timestamp_ns)
        if self._audio_src is None:
            return

        timestamp_ns = msg_timestamp_ns - self._record_start_time_ns
        duration_ns = int(1 / msg.sampling_frequency * msg.frame_sample_count)

        if timestamp_ns >= 0:
            self._audio_src.emit(
                "push-buffer",
                VideoRecorder.data_to_gst_buffer(msg.data, timestamp_ns, duration_ns),
            )

    @staticmethod
    def _verify_audio_frame_msg(
        configuration: VideoRecorderConfiguration, msg: AudioFrame
    ):
        try:
            return (
                AudioFormat(msg.format) == configuration.audio_format
                and msg.channel_count == configuration.audio_channel_count
                and msg.sampling_frequency == configuration.audio_sampling_frequency
            )
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
            pipeline = Gst.parse_launch(
                f"{video_pipeline} ! {mux_pipeline} {audio_pipeline} ! mux."
            )
            video_src = pipeline.get_by_name("video_src")
            audio_src = pipeline.get_by_name("audio_src")
            bus = pipeline.get_bus()
            bus.add_watch(0, self._on_bus_message_cb)
            pipeline.set_state(Gst.State.PLAYING)

            self._pipeline = pipeline
            self._video_src = video_src
            self._audio_src = audio_src
            self._bus = bus
        except GLib.Error as e:
            rospy.loginfo(f"GStreamer pipeline failed({e})")

    def _stop_if_started(self):
        if self._pipeline is None:
            return

        self._video_src.emit("end-of-stream")  # type: ignore
        self._audio_src.emit("end-of-stream")  # type: ignore
        self._bus.timed_pop_filtered(  # type: ignore
            Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS
        )

        self._bus.remove_watch()  # type: ignore
        self._pipeline.set_state(Gst.State.NULL)
        self._pipeline = None
        self._video_src = None
        self._audio_src = None
        self._bus = None

    def _on_bus_message_cb(self, bus, msg):
        rospy.loginfo(f"Gstreamer bus message: {msg}")

    @staticmethod
    def _create_mux_pipeline(configuration: VideoRecorderConfiguration) -> str:
        path = os.path.join(
            configuration.output_directory, VideoRecorder._get_filename(configuration)
        )
        if (
            configuration.video_codec == VideoCodec.H264
            or configuration.video_codec == VideoCodec.H265
        ):
            attributes = "reserved-bytes-per-sec=100 reserved-max-duration=20184000000000 reserved-moov-update-period=100000000"
            return f"qtmux name=mux {attributes} ! filesink location={path}.mp4"
        elif (
            configuration.video_codec == VideoCodec.VP8
            or configuration.video_codec == VideoCodec.VP9
        ):
            return f"matroskamux name=mux ! filesink location={path}.mkv"

        raise NotImplementedError()

    @staticmethod
    def _get_filename(configuration: VideoRecorderConfiguration) -> str:
        now = datetime.datetime.utcfromtimestamp(rospy.Time.now().to_sec())
        return configuration.filename_prefix + now.strftime("%Y-%m-%d_%H-%M-%S")

    @staticmethod
    def _create_video_pipeline(configuration: VideoRecorderConfiguration) -> str:
        pipeline = VideoRecorder._create_video_input_pipeline(configuration)

        if VideoRecorder._verify_nvidia_hardware_encoder(configuration):
            encoder = configuration.video_codec.nv_hw_enc
            bitrate = configuration.video_codec.convert_nvidia_hardware_bitrate(
                configuration.video_bitrate
            )
            pipeline += " ! videoconvert ! nvvidconv ! capsfilter caps=video/x-raw(memory:NVMM),format=(string)I420"
            pipeline += f" ! {encoder} bitrate={bitrate}"
        else:
            rospy.logwarn("NVIDIA hardware encoder are not available.")

            encoder = configuration.video_codec.sw_enc
            bitrate_attribute = (
                configuration.video_codec.get_software_bitrate_attribute()
            )
            bitrate = configuration.video_codec.convert_software_bitrate(
                configuration.video_bitrate
            )
            pipeline += f" ! videoconvert ! capsfilter caps=video/x-raw,format=I420"
            pipeline += f" ! {encoder} {bitrate_attribute}={bitrate}"

        pipeline += configuration.video_codec.gst_parser

        return pipeline

    @staticmethod
    def _create_video_input_pipeline(configuration: VideoRecorderConfiguration) -> str:
        caps = f"video/x-raw,format={configuration.video_format.value}"
        caps += (
            f",width={configuration.video_width},height={configuration.video_height}"
        )
        return f"appsrc name=video_src emit-signals=True is-live=True format=time caps={caps} ! queue max-size-buffers=100"

    @staticmethod
    def _verify_nvidia_hardware_encoder(
        configuration: VideoRecorderConfiguration,
    ) -> bool:
        return (
            Gst.ElementFactory.find("nvvidconv") is not None
            and Gst.ElementFactory.find(configuration.video_codec.nv_hw_enc) is not None
        )

    @staticmethod
    def _create_audio_pipeline(configuration: VideoRecorderConfiguration) -> str:
        pipeline = VideoRecorder._create_audio_input_pipeline(configuration)

        audioconvert_attributes = ""
        if configuration.audio_channel_count > 1:
            values = [
                f"(float){1 / configuration.audio_channel_count}"
            ] * configuration.audio_channel_count
            audioconvert_attributes = 'mix-matrix="<<' + ",".join(values) + '>>"'

        encoder = configuration.audio_codec.enc
        pipeline += f" ! audioconvert {audioconvert_attributes}"
        pipeline += f" ! capsfilter caps=audio/x-raw,channels=1 ! audiorate ! {encoder}"
        return pipeline

    @staticmethod
    def _create_audio_input_pipeline(configuration: VideoRecorderConfiguration) -> str:
        channel_mask = ""
        if configuration.audio_channel_count > 2:
            channel_mask = ",channel-mask=(bitmask)0x0"

        caps = f"audio/x-raw,format={configuration.audio_format.value},channels={configuration.audio_channel_count}"
        caps += f",rate={configuration.audio_sampling_frequency},layout=interleaved{channel_mask}"
        return f"appsrc name=audio_src is-live=True format=time caps={caps} ! queue max-size-buffers=100"

    @staticmethod
    def data_to_gst_buffer(
        data: bytes, timestamp_ns: int, duration_ns: int
    ) -> Gst.Buffer:
        buffer = Gst.Buffer.new_wrapped(data)
        buffer.pts = int(timestamp_ns)
        buffer.duration = int(duration_ns)
        return buffer


class VideoRecorderNode:
    def __init__(self):
        self._recorder_lock = threading.Lock()
        self._recorder = None
        self._recorder_configuration = VideoRecorderConfiguration.from_parameters()

        self._filter_state = hbba_lite.OnOffHbbaFilterState(  # type: ignore
            "video_recorder/filter_state"
        )
        self._filter_state.on_changed(self._on_filter_state_changed)

    def _on_filter_state_changed(self, _, next_is_filtering_all_messages):
        with self._recorder_lock:
            if next_is_filtering_all_messages and self._recorder is not None:
                self._recorder.close()
                self._recorder = None
            elif not next_is_filtering_all_messages and self._recorder is None:
                self._recorder = VideoRecorder(self._recorder_configuration)

    def run(self):
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            pass
        finally:
            with self._recorder_lock:
                if self._recorder is not None:
                    self._recorder.close()
                    self._recorder = None


def main():
    rospy.init_node("video_recorder_node")

    Gst.init(None)

    video_recorder_node = VideoRecorderNode()
    video_recorder_node.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
