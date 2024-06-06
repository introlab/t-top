# recorders
This folder contains nodes to record data.

## Nodes
### `video_recorder_node.py`
This node records an image topic and an audio topic.

#### Parameters
 - `output_directory` (string): The output directory where to save the recorded files.
 - `filename_prefix` (string): The filename prefix of the recorded files.
 - `video_stream_name` (string): The video stream name.
 - `video_stream_format` (string): The image encoding of the image topic (`rgb8` or `bgr8`).
 - `video_stream_width` (int): The image width of the image topic.
 - `video_stream_height` (int): The image height of the image topic.
 - `video_stream_framerate` (int): The video frame rate.
 - `video_stream_codec` (string): The video codec to use (`h264`, `h265`, `vp8`, `vp9` or `av1`).
 - `video_stream_bitrate` (int): The video codec bitrate (bits/s).
 - `video_stream_delay_s` (double): The video delay in seconds.
 - `video_stream_language_code` (string): The video stream language code.
 - `audio_stream_name` (string): The audio stream name
 - `audio_stream_format` (string): The audio format of the audio topic.
 - `audio_stream_channel_count` (int): The channel count of the audio topic.
 - `audio_stream_sampling_frequency` (int): The sampling frequency of the audio topic.
 - `audio_stream_codec` (string): The audio codec to use (`aac`, or `mp3`).
 - `audio_stream_merge_channels` (bool): Indicates to merge the audio channels.
 - `audio_stream_language_code` (string): The audio stream language code.

#### Subscribed Topics
 - `video_{video_stream_name}` ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)): The image topic to record.
 - `audio_{audio_stream_name}` ([audio_utils_msgs/AudioFrame](https://github.com/introlab/audio_utils/blob/main/audio_utils_msgs/msg/AudioFrame.msg)): The audio topic to record.

#### Services
 - `video_recorder/filter_state` ([hbba_lite_srvs/SetOnOffFilterState](../../utils/hbba_lite/hbba_lite_srvs/srv/SetOnOffFilterState.srv)): The HBBA filter state service to enable or disable the recording.

### `perception_logger_node`
This node logs the video analysis messages and the audio analysis messages into a SQLite database.

#### Parameters
 - `database_path` (string): The database path of the SQLite database.
 - `frame_id` (string): The frame id of the positions and directions in the database.

#### Subscribed Topics
 - `video_analysis` ([perception_msgs/VideoAnalysis](../../perceptions/perception_msgs/msg/VideoAnalysis.msg)): The video analysis containing the detected objects. The video analysis must contain 3d positions.
 - `audio_analysis` ([perception_msgs/AudioAnalysis](../../perceptions/perception_msgs/msg/AudioAnalysis.msg)): The audio analysis containing the audio classes, voice embedding and the sound direction.
 - `talk/text` ([talk/Text](../../behaviors/behavior_msgs/msg/Text.msg)): The text spoken by the robot.
 - `speech_to_text/transcript` ([perception_msgs/Transcript](../../perceptions/perception_msgs/msg/Transcript.msg)): The text spoken by the human.
