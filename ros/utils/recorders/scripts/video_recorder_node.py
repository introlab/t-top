#!/usr/bin/env python3

import datetime
from enum import Enum
import os
import sys
import threading
from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import (
    Optional,
    List,
    TypedDict,
    cast,
    Hashable,
    Set,
    Iterable,
    Dict,
    Callable,
    Tuple,
)

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst  # type: ignore
from gi.repository import GLib  # type: ignore

import rospy

from audio_utils.msg import AudioFrame
from sensor_msgs.msg import Image

import hbba_lite


def apply(func: Callable, iter: Iterable) -> None:
    for x in iter:
        func(x)


def graph_export(pipeline: Gst.Pipeline, name: str):
    Gst.debug_bin_to_dot_file(pipeline, Gst.DebugGraphDetails.ALL, name)


def get_duplicates(iterable: Iterable[Hashable]) -> Set[Hashable]:
    ti = tuple(iterable)
    return {x for x in iterable if ti.count(x) > 1}


class FormatEnum(str, Enum):
    gstreamer_value: str

    def __new__(cls, value: str, gstreamer_value: str):
        obj = str.__new__(cls, value)
        obj._value_ = value

        obj.gstreamer_value = gstreamer_value
        return obj


class VideoCodecEnum(str, Enum):
    software_encoder: str
    nvidia_hardware_encoder: Optional[str]
    _gstreamer_parser: Optional[str]

    def __new__(
        cls,
        value: str,
        software_encoder: str,
        nvidia_hardware_encoder: Optional[str] = None,
        parser: Optional[str] = None,
    ):
        obj = str.__new__(cls, value)
        obj._value_ = value

        obj.software_encoder = software_encoder
        obj.nvidia_hardware_encoder = nvidia_hardware_encoder
        obj._gstreamer_parser = parser
        return obj

    def gstreamer_parser_string(self, configuration_name: str) -> str:
        if self._gstreamer_parser is None:
            return ''
        return f' ! {self._gstreamer_parser} name={self._gstreamer_parser}_{configuration_name}'


class AudioCodecEnum(str, Enum):
    encoder: str

    def __new__(cls, value: str, enc: str):
        obj = str.__new__(cls, value)
        obj._value_ = value

        obj.encoder = enc
        return obj


class VideoFormat(FormatEnum):
    RGB = ('rgb8', 'RGB')
    BGR = ('bgr8', 'BGR')


class VideoCodec(VideoCodecEnum):
    H264 = ('h264', 'x264enc', 'nvv4l2h264enc', 'h264parse')
    H265 = ('h265', 'x265enc', 'nvv4l2h265enc', 'h265parse')
    VP8 = ('vp8', 'vp8enc', 'nvv4l2vp8enc')
    VP9 = ('vp9', 'vp9enc', 'nvv4l2vp9enc')
    AV1 = ('av1', 'av1enc', 'nvv4l2av1enc')

    def get_software_bitrate_attribute(self) -> str:
        if self == VideoCodec.H264 or self == VideoCodec.H265:
            return 'bitrate'
        elif self == VideoCodec.VP8 or self == VideoCodec.VP9:
            return 'target-bitrate'
        else:
            raise ValueError('Invalid video codec')

    def convert_software_bitrate(self, bitrate) -> int:
        if self == VideoCodec.H264 or self == VideoCodec.H265:
            return bitrate // 1000
        elif self == VideoCodec.VP8 or self == VideoCodec.VP9:
            return bitrate
        else:
            raise ValueError('Invalid video codec')

    def convert_nvidia_hardware_bitrate(self, bitrate) -> int:
        return bitrate


class AudioFormat(FormatEnum):
    S8 = ('signed_8', 'S8')
    S16LE = ('signed_16', 'S16LE')
    S24LE = ('signed_24', 'S24LE')
    S24_32LE = ('signed_padded_24', 'S24_32LE')
    S32LE = ('signed_32', 'S32LE')
    U8 = ('unsigned_8', 'U8')
    U16LE = ('unsigned_16', 'U16LE')
    U24LE = ('unsigned_24', 'U24LE')
    U24_32LE = ('unsigned_padded_24', 'U24_32LE')
    U32LE = ('unsigned_32', 'U32LE')
    F32LE = ('float', 'F32LE')
    F64LE = ('double', 'F64LE')


class AudioCodec(AudioCodecEnum):
    AAC = ('aac', 'voaacenc')
    MP3 = ('mp3', 'lamemp3enc')
    OPUS = ('opus', 'opusenc')
    FLAC = ('flac', 'flacenc')


@dataclass
class VideoStreamConfiguration:
    name: str
    format: VideoFormat
    width: int
    height: int
    framerate: int
    codec: VideoCodec
    bitrate: int
    delay_s: float
    delay_ns: int
    language_code: str

    @property
    def full_name(self) -> str:
        return f'video_{self.name}'

    class VideoStreamParameters(TypedDict):
        name: str
        format: str
        width: int
        height: int
        framerate: int
        codec: str
        bitrate: int
        delay_s: float
        language_code: str

    @staticmethod
    def check_parameters(parameters: VideoStreamParameters, index: int) -> None:
        sould_exit = False
        if 'name' not in parameters:
            debug_str = f'at index |{index}|'
            rospy.logerr(f'Video stream parameters {debug_str} must have a name')
            sould_exit = True
        else:
            debug_str = f'for name |{parameters["name"]}|'

        if 'format' not in parameters:
            rospy.logerr(f'Video stream parameters {debug_str} must have a format')
            sould_exit = True
        if 'width' not in parameters:
            rospy.logerr(f'Video stream parameters {debug_str} must have a width')
            sould_exit = True
        if 'height' not in parameters:
            rospy.logerr(f'Video stream parameters {debug_str} must have a height')
            sould_exit = True
        if 'framerate' not in parameters:
            rospy.logerr(f'Video stream parameters {debug_str} must have a framerate')
            sould_exit = True
        if 'codec' not in parameters:
            rospy.logerr(f'Video stream parameters {debug_str} must have a codec')
            sould_exit = True
        if 'bitrate' not in parameters:
            rospy.logerr(f'Video stream parameters {debug_str} must have a bitrate')
            sould_exit = True

        if sould_exit:
            sys.exit(os.EX_CONFIG)

    @staticmethod
    def from_parameters(
        parameters: VideoStreamParameters,
        index: int,
    ) -> 'VideoStreamConfiguration':

        VideoStreamConfiguration.check_parameters(parameters, index)

        return VideoStreamConfiguration(
            name=parameters['name'],
            format=VideoFormat(parameters['format']),
            width=parameters['width'],
            height=parameters['height'],
            framerate=parameters['framerate'],
            codec=VideoCodec(parameters['codec']),
            bitrate=parameters['bitrate'],
            delay_s=parameters['delay_s'] if 'delay_s' in parameters else 0.0,
            delay_ns=int(parameters['delay_s'] * 1e9 if 'delay_s' in parameters else 0),
            language_code=parameters['language_code']
            if 'language_code' in parameters
            else 'eng',
        )


@dataclass
class AudioStreamConfiguration:
    name: str
    format: AudioFormat
    channel_count: int
    sampling_frequency: int
    codec: AudioCodec
    merge_channels: bool
    language_code: str

    @property
    def full_name(self) -> str:
        return f'audio_{self.name}'

    class AudioStreamParameters(TypedDict):
        name: str
        format: str
        channel_count: int
        sampling_frequency: int
        codec: str
        merge_channels: bool
        language_code: str

    @staticmethod
    def check_parameters(parameters: AudioStreamParameters, index: int) -> None:
        sould_exit = False
        if 'name' not in parameters:
            debug_str = f'at index |{index}|'
            rospy.logerr(f'Audio stream parameters {debug_str} must have a name')
            sould_exit = True
        else:
            debug_str = f'for name |{parameters["name"]}|'

        if 'format' not in parameters:
            rospy.logerr(f'Audio stream parameters {debug_str} must have a format')
            sould_exit = True
        if 'channel_count' not in parameters:
            rospy.logerr(
                f'Audio stream parameters {debug_str} must have a channel_count'
            )
            sould_exit = True
        if 'sampling_frequency' not in parameters:
            rospy.logerr(
                f'Audio stream parameters {debug_str} must have a sampling_frequency'
            )
            sould_exit = True
        if 'codec' not in parameters:
            rospy.logerr(f'Audio stream parameters {debug_str} must have a codec')
            sould_exit = True

        if sould_exit:
            sys.exit(os.EX_CONFIG)

    @staticmethod
    def from_parameters(
        parameters: AudioStreamParameters,
        index: int,
    ) -> 'AudioStreamConfiguration':

        AudioStreamConfiguration.check_parameters(parameters, index)

        return AudioStreamConfiguration(
            name=parameters['name'],
            format=AudioFormat(parameters['format']),
            channel_count=parameters['channel_count'],
            sampling_frequency=parameters['sampling_frequency'],
            codec=AudioCodec(parameters['codec']),
            merge_channels=parameters['merge_channels']
            if 'merge_channels' in parameters
            else True,
            language_code=parameters['language_code']
            if 'language_code' in parameters
            else 'eng',
        )


class VideoRecorderConfiguration:
    def __init__(
        self,
        output_directory: str,
        filename_prefix: str,
        video_streams: List[VideoStreamConfiguration],
        audio_streams: List[AudioStreamConfiguration],
    ):
        self.output_directory = output_directory
        self.filename_prefix = filename_prefix

        self.video_streams = video_streams
        self.audio_streams = audio_streams

    @staticmethod
    def _is_valid_name(name: str) -> bool:
        return name.replace('_', '').isalnum() and name.isascii()

    @staticmethod
    def _is_valid_language_code(language_code: str) -> bool:
        return (
            language_code.isalpha()
            and language_code.isascii()
            and len(language_code) == 3
        )

    @staticmethod
    def from_parameters() -> 'VideoRecorderConfiguration':

        output_directory = cast(str, rospy.get_param('~output_directory'))
        filename_prefix = cast(
            str, rospy.get_param('~filename_prefix', '')  # type: ignore
        )
        video_streams = [
            VideoStreamConfiguration.from_parameters(video_stream, index) for index, video_stream in enumerate(rospy.get_param(f'~video_streams'))  # type: ignore
        ]
        audio_streams = [
            AudioStreamConfiguration.from_parameters(audio_stream, index) for index, audio_stream in enumerate(rospy.get_param(f'~audio_streams'))  # type: ignore
        ]

        streams_count = len(video_streams) + len(audio_streams)
        if streams_count < 1:
            rospy.logerr(f'At least one video or audio stream must be specified')
            sys.exit(os.EX_CONFIG)

        video_names = [video_stream.name for video_stream in video_streams]
        if len(duplicates := get_duplicates(video_names)) > 0:
            rospy.logerr(
                f'Video stream names must be unique, duplicates found: {duplicates}'
            )
            sys.exit(os.EX_CONFIG)

        audio_names = [audio_stream.name for audio_stream in audio_streams]
        if len(duplicates := get_duplicates(audio_names)) > 0:
            rospy.logerr(
                f'Audio stream names must be unique, duplicates found: {duplicates}'
            )
            sys.exit(os.EX_CONFIG)

        video_lang_codes = [
            video_stream.language_code for video_stream in video_streams
        ]
        if len(duplicates := get_duplicates(video_lang_codes)) > 0:
            rospy.logerr(
                f'Video language codes must be unique, duplicates found: {duplicates}'
            )
            sys.exit(os.EX_CONFIG)

        audio_lang_codes = [
            audio_stream.language_code for audio_stream in audio_streams
        ]
        if len(duplicates := get_duplicates(audio_lang_codes)) > 0:
            rospy.logerr(
                f'Audio language codes must be unique, duplicates found: {duplicates}'
            )
            sys.exit(os.EX_CONFIG)

        should_exit = False
        for stream_name in itertools.chain(video_names, audio_names):
            if not VideoRecorderConfiguration._is_valid_name(stream_name):
                rospy.logerr(
                    f'Stream names must only contain alphanumeric ascii characters and underscores; invalid name: {stream_name}'
                )
                should_exit = True
        for lang_code in itertools.chain(video_lang_codes, audio_lang_codes):
            if not VideoRecorderConfiguration._is_valid_language_code(lang_code):
                rospy.logerr(
                    'Language codes must be ISO 639-3 codes, and should be a 3-letter code (see https://iso639-3.sil.org/code_tables/639/data)'
                    f'; invalid language code: {lang_code}'
                )
                should_exit = True
        if should_exit:
            sys.exit(os.EX_CONFIG)

        return VideoRecorderConfiguration(
            output_directory=output_directory,
            filename_prefix=filename_prefix,
            video_streams=video_streams,
            audio_streams=audio_streams,
        )


class VideoRecorder:
    def __init__(self, configuration: VideoRecorderConfiguration):
        self._configuration = configuration
        os.makedirs(self._configuration.output_directory, exist_ok=True)

        self._record_start_time_ns = 0
        self._clear_last_timestamps()

        self.lock = threading.Lock()

        self._bus = None

        self._glib_main_loop = GLib.MainLoop()
        self._glib_thread = threading.Thread(target=self._glib_main_loop.run)
        self._glib_thread.start()

        self._pipeline = None

        self._clear_sources_lists()

        self._image_subscribers = {
            video_stream_configuration.name: rospy.Subscriber(
                video_stream_configuration.full_name,
                Image,
                self._image_cb,
                video_stream_configuration,
                queue_size=10,
            )
            for video_stream_configuration in self._configuration.video_streams
        }
        self._audio_subscribers = {
            audio_stream_configuration.name: rospy.Subscriber(
                audio_stream_configuration.full_name,
                AudioFrame,
                self._audio_cb,
                audio_stream_configuration,
                queue_size=10,
            )
            for audio_stream_configuration in self._configuration.audio_streams
        }

    def close(self):
        apply(rospy.Subscriber.unregister, self._image_subscribers.values())
        apply(rospy.Subscriber.unregister, self._audio_subscribers.values())
        self._stop_if_started()

    def _image_cb(self, msg: Image, configuration: VideoStreamConfiguration):
        if not VideoRecorder._verify_image_msg(configuration, msg):
            rospy.logerr(
                f'Invalid image (encoding={msg.encoding}, width={msg.width}, height={msg.height})'
            )
            return

        msg_timestamp_ns = msg.header.stamp.to_nsec()
        self._start_if_not_started(msg_timestamp_ns)
        if self._video_srcs[configuration.name] is None:
            return

        timestamp_ns = (
            msg_timestamp_ns - self._record_start_time_ns + configuration.delay_ns
        )
        duration_ns = max(
            0,
            timestamp_ns - self._last_video_frame_timestamp_ns[configuration.name],
        )
        self._last_video_frame_timestamp_ns[configuration.name] = timestamp_ns

        if timestamp_ns >= 0:
            rospy.logdebug(f'Pushing image {msg.header.seq} for {configuration.name}')
            self._video_srcs[configuration.name].emit(  # type: ignore
                'push-buffer',
                VideoRecorder.data_to_gst_buffer(msg.data, timestamp_ns, duration_ns),
            )

    @staticmethod
    def _verify_image_msg(configuration: VideoStreamConfiguration, msg: Image):
        try:
            return (
                VideoFormat(msg.encoding) == configuration.format
                and msg.width == configuration.width
                and msg.height == configuration.height
            )
        except ValueError:
            return False

    def _audio_cb(self, msg: AudioFrame, configuration: AudioStreamConfiguration):
        if not VideoRecorder._verify_audio_frame_msg(configuration, msg):
            rospy.logerr(
                f'Invalid audio frame (format={msg.format}, channel_count={msg.channel_count}, sampling_frequency={msg.sampling_frequency})'
            )
            return

        msg_timestamp_ns = msg.header.stamp.to_nsec()
        self._start_if_not_started(msg_timestamp_ns)
        if self._audio_srcs[configuration.name] is None:
            return

        timestamp_ns = msg_timestamp_ns - self._record_start_time_ns
        duration_ns = int(1 / msg.sampling_frequency * msg.frame_sample_count)

        if timestamp_ns >= 0:
            rospy.logdebug(
                f'Pushing audio frame {msg.header.seq} for {configuration.name}'
            )
            self._audio_srcs[configuration.name].emit(  # type: ignore
                'push-buffer',
                VideoRecorder.data_to_gst_buffer(msg.data, timestamp_ns, duration_ns),
            )

    @staticmethod
    def _verify_audio_frame_msg(
        configuration: AudioStreamConfiguration, msg: AudioFrame
    ):
        try:
            return (
                AudioFormat(msg.format) == configuration.format
                and msg.channel_count == configuration.channel_count
                and msg.sampling_frequency == configuration.sampling_frequency
            )
        except ValueError:
            return False

    def _start_if_not_started(self, record_start_time_ns: int):
        with self.lock:
            if self._pipeline is not None:
                return

            self._record_start_time_ns = record_start_time_ns
            self._clear_last_timestamps()

            mux_pipeline = VideoRecorder._create_mux_pipeline(self._configuration)
            video_and_audio_pipelines = itertools.chain(
                (
                    VideoRecorder._create_video_pipeline(video_stream_configuration)
                    for video_stream_configuration in self._configuration.video_streams
                ),
                (
                    VideoRecorder._create_audio_pipeline(audio_stream_configuration)
                    for audio_stream_configuration in self._configuration.audio_streams
                ),
            )

            pipeline_str = (
                f'{next(video_and_audio_pipelines)} ! {mux_pipeline}  '
                + '  '.join(f'{p} ! mux.' for p in video_and_audio_pipelines)
            )

            rospy.loginfo(f'Launching gstreamer pipeline: {pipeline_str}')

            try:
                pipeline = Gst.parse_launch(pipeline_str)

                video_srcs = {
                    video_stream_configuration.name: pipeline.get_by_name(
                        f'appsrc_{video_stream_configuration.full_name}'
                    )
                    for video_stream_configuration in self._configuration.video_streams
                }
                audio_srcs = {
                    audio_stream_configuration.name: pipeline.get_by_name(
                        f'appsrc_{audio_stream_configuration.full_name}'
                    )
                    for audio_stream_configuration in self._configuration.audio_streams
                }

                bus = pipeline.get_bus()
                bus.add_watch(0, self._on_bus_message_cb)

                pipeline.set_state(Gst.State.PLAYING)

                self._pipeline = pipeline
                self._video_srcs = video_srcs
                self._audio_srcs = audio_srcs
                self._bus = bus
            except GLib.Error as e:  # type: ignore
                rospy.loginfo(f'GStreamer pipeline failed({e})')
                sys.exit(-2)

            graph_export(self._pipeline, 'started')

    def _stop_if_started(self):
        with self.lock:
            if self._pipeline is None:
                return

            graph_export(self._pipeline, 'stop')

            # We don't want the message callback to pop the messages anymore, so that we don't miss the EOS message
            self._glib_main_loop.quit()

            apply(lambda s: s.emit('end-of-stream'), self._video_srcs.values())  # type: ignore
            apply(lambda s: s.emit('end-of-stream'), self._audio_srcs.values())  # type: ignore
            rospy.loginfo('Sent EOS to all sources')

            if self._bus:
                self._bus.timed_pop_filtered(  # type: ignore
                    Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS
                )
                self._bus.remove_watch()

            rospy.loginfo('Received EOS on the bus')

            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None

            self._clear_sources_lists()

            self._bus = None

    def _on_bus_message_cb(self, bus, msg):
        def message_to_string(msg: Gst.Message) -> Tuple[str, Callable]:
            if msg.type == Gst.MessageType.EOS:
                return 'End-of-stream', rospy.loginfo
            elif msg.type == Gst.MessageType.WARNING:
                err, debug = msg.parse_warning()
                return f'Warning: |{err}|: {debug}', rospy.logwarn
            elif msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                return f'Bus call: Error: |{err}|: {debug}', rospy.logerr
            elif msg.type == Gst.MessageType.BUFFERING:
                percent = msg.parse_buffering()
                return f'Buffering ({percent}%)', rospy.loginfo
            elif msg.type == Gst.MessageType.STATE_CHANGED:
                old_state, new_state, pending_state = msg.parse_state_changed()
                return (
                    f'Element |{msg.src.get_name()}| (of type |{type(msg.src).__name__}|) state changed from |{old_state.value_nick}| to |{new_state.value_nick}| '
                    f'(pending |{pending_state.value_nick}|)'
                ), rospy.loginfo
            elif msg.type == Gst.MessageType.STREAM_START:
                return 'Stream started', rospy.loginfo
            elif msg.type == Gst.MessageType.STREAM_STATUS:
                status_type, owner = msg.parse_stream_status()
                return (
                    f'Stream status; status: |{status_type.value_nick}|, owner: |{type(owner).__name__}|',
                    rospy.loginfo,
                )
            elif msg.type == Gst.MessageType.LATENCY:
                return 'Latency', rospy.loginfo
            elif msg.type == Gst.MessageType.NEW_CLOCK:
                clock = msg.parse_new_clock()
                return (
                    f'New clock; sync: |{clock.is_synced()}|, resolution: |{clock.get_resolution()}|',
                    rospy.loginfo,
                )
            elif msg.type == Gst.MessageType.ASYNC_DONE:
                return 'Async done', rospy.loginfo
            else:
                return (
                    f'Unknow message of type |{Gst.MessageType.get_name(msg.type).upper().replace("-", "_")}|',
                    rospy.loginfo,
                )

        msg_str, log_func = message_to_string(msg)
        log_func(f'Gstreamer bus message: {msg_str}')
        return True

    @staticmethod
    def _create_mux_pipeline(configuration: VideoRecorderConfiguration) -> str:
        path = os.path.join(
            configuration.output_directory, VideoRecorder._get_filename(configuration)
        )
        return (
            f'matroskamux name=mux writing-app="{Path(__file__).stem}"'
            f' ! filesink name=filesink location={path}.mkv'
        )

    @staticmethod
    def _get_filename(configuration: VideoRecorderConfiguration) -> str:
        now = datetime.datetime.utcfromtimestamp(rospy.Time.now().to_sec())
        filename_us = configuration.filename_prefix + now.strftime('%Y-%m-%d_%H-%M-%S.%f')
        filename_ms = filename_us[:-3]
        return filename_ms

    @staticmethod
    def _create_video_pipeline(configuration: VideoStreamConfiguration) -> str:
        pipeline = VideoRecorder._create_video_input_pipeline(configuration)

        if VideoRecorder._verify_nvidia_hardware_encoder(configuration):
            encoder = configuration.codec.nvidia_hardware_encoder
            bitrate = configuration.codec.convert_nvidia_hardware_bitrate(
                configuration.bitrate
            )
            pipeline += (
                f' ! videoconvert name=videoconvert_{configuration.full_name}'
                f' ! nvvidconv name=nvvidconv_{configuration.full_name}'
                f' ! capsfilter name=capsfilter_{configuration.full_name} caps=video/x-raw(memory:NVMM),format=(string)I420'
                f' ! {encoder} name={encoder}_{configuration.full_name} bitrate={bitrate}'
            )
        else:
            rospy.logwarn('NVIDIA hardware encoder are not available.')

            encoder = configuration.codec.software_encoder
            bitrate_attribute = configuration.codec.get_software_bitrate_attribute()
            bitrate = configuration.codec.convert_software_bitrate(
                configuration.bitrate
            )
            pipeline += (
                f' ! videoconvert name=videoconvert_{configuration.full_name}'
                f' ! capsfilter name=capsfilter_{configuration.full_name} caps=video/x-raw,format=I420'
                f' ! {encoder} name={encoder}_{configuration.full_name} {bitrate_attribute}={bitrate}'
            )

        pipeline += (
            f'{configuration.codec.gstreamer_parser_string(configuration.full_name)}'
            f' ! taginject name=taginject_{configuration.full_name} tags="language-code={configuration.language_code}"'
        )

        return pipeline

    @staticmethod
    def _create_video_input_pipeline(configuration: VideoStreamConfiguration) -> str:
        caps = (
            f'video/x-raw'
            f',format={configuration.format.gstreamer_value}'
            f',width={configuration.width}'
            f',height={configuration.height}'
            f',framerate={configuration.framerate}/1'
        )
        return (
            f'appsrc name=appsrc_video_{configuration.name} emit-signals=true is-live=true format=time caps={caps}'
            f' ! queue name=queue_video_{configuration.name} max-size-buffers=100'
        )

    @staticmethod
    def _verify_nvidia_hardware_encoder(
        configuration: VideoStreamConfiguration,
    ) -> bool:
        return (
            Gst.ElementFactory.find('nvvidconv') is not None
            and Gst.ElementFactory.find(configuration.codec.nvidia_hardware_encoder)
            is not None
        )

    @staticmethod
    def _create_audio_pipeline(configuration: AudioStreamConfiguration) -> str:
        pipeline = VideoRecorder._create_audio_input_pipeline(configuration)

        audioconvert_attributes = ''
        if configuration.channel_count > 1 and configuration.merge_channels:
            values = [
                f'(float){1 / configuration.channel_count}'
            ] * configuration.channel_count
            audioconvert_attributes = ' mix-matrix="<<' + ','.join(values) + '>>"'
        elif not configuration.merge_channels:
            # If the number of channels is bigger than the max number of channels that can be encoded together,
            # we would need to split the audio into multiple streams.
            # `deinterleave` splits the audio into multiple streams, and `interleave` merges them back together.
            # The number of streams is the number of channels divided by the max number of channels that can be encoded together based on the encoder used.
            # Sample command doing this (but with only opus, specific rate, and 4 channels; also results in bad opus headers according to VLC):
            # `gst-launch-1.0 audiotestsrc ! "audio/x-raw,channels=4,layout=interleaved,rate=48000" ! deinterleave name=d  d.src_0 ! queue ! interleave name=i ! opusenc ! matroskamux name=m ! filesink location=test.mkv  d.src_1 ! queue ! i.  d.src_2 ! queue ! interleave name=ii ! opusenc ! m.  d.src_3 ! queue ! ii.`
            raise NotImplementedError('Not merging channels is not implemented yet.')

        encoder = configuration.codec.encoder
        pipeline += (
            f' ! audioconvert{audioconvert_attributes} name=audioconvert_audio_{configuration.name}'
            f' ! capsfilter name=capsfilter_audio_{configuration.name} caps=audio/x-raw,channels=1'
            f' ! audiorate name=audiorate_audio_{configuration.name}'
            f' ! {encoder} name={encoder}_audio_{configuration.name}'
            f' ! taginject name=taginject_audio_{configuration.name} tags="language-code={configuration.language_code}"'
        )
        return pipeline

    @staticmethod
    def _create_audio_input_pipeline(configuration: AudioStreamConfiguration) -> str:
        channel_mask = ''
        if configuration.channel_count > 2:
            channel_mask = ',channel-mask=(bitmask)0x0'

        caps = (
            f'audio/x-raw,format={configuration.format.gstreamer_value},channels={configuration.channel_count}'
            f',rate={configuration.sampling_frequency},layout=interleaved{channel_mask}'
        )
        return (
            f'appsrc name=appsrc_audio_{configuration.name} emit-signals=true is-live=true format=time caps={caps}'
            f' ! queue name=queue_audio_{configuration.name} max-size-buffers=100'
        )

    @staticmethod
    def data_to_gst_buffer(
        data: bytes, timestamp_ns: int, duration_ns: int
    ) -> Gst.Buffer:
        buffer = Gst.Buffer.new_wrapped(data)
        buffer.pts = int(timestamp_ns)
        buffer.duration = int(duration_ns)
        return buffer

    def _clear_sources_lists(self) -> None:
        self._video_srcs: Dict[str, Optional[Gst.Element]] = {
            video_stream_configuration.name: None
            for video_stream_configuration in self._configuration.video_streams
        }
        self._audio_srcs: Dict[str, Optional[Gst.Element]] = {
            audio_stream_configuration.name: None
            for audio_stream_configuration in self._configuration.audio_streams
        }

    def _clear_last_timestamps(self) -> None:
        self._last_video_frame_timestamp_ns = {
            video_stream_configuration.name: 0
            for video_stream_configuration in self._configuration.video_streams
        }


class VideoRecorderNode:
    def __init__(self):
        self._recorder_lock = threading.Lock()
        self._recorder = None
        self._recorder_configuration = VideoRecorderConfiguration.from_parameters()

        self._filter_state = hbba_lite.OnOffHbbaFilterState(  # type: ignore
            'video_recorder/filter_state'
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
    rospy.init_node('video_recorder_node')

    Gst.init(None)

    video_recorder_node = VideoRecorderNode()
    video_recorder_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
