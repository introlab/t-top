# recorders
This folder contains nodes to record data.

## Nodes
### `video_recorder_node.py`
This node records an image topic and an audio topic.

#### Parameters
 - `output_directory` (string): The output directory where to save the recorded files.
 - `filename_prefix` (string): The filename prefix of the recorded files.
 - `video_format` (string): The image encoding of the image topic (`rgb8` or `bgr8`).
 - `video_width` (int): The image width of the image topic.
 - `video_height` (int): The image height of the image topic.
 - `video_codec` (string): The video codec to use (`h264`, `h265`, `vp8` or `vp9`).
 - `video_bitrate` (int): The video codec bitrate (bits/s).
 - `video_delay_s` (double): The video delay in seconds.
 - `audio_format` (string): The audio format of the audio topic.
 - `audio_channel_count` (int): The channel count of the audio topic.
 - `audio_sampling_frequency` (int): The sampling frequency of the audio topic.
 - `audio_codec` (string): The audio codec to use (`aac`, or `mp3`).

#### Subscribed Topics
 - `image` ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)): The image topic to record.
 - `audio` ([audio_utils/AudioFrame](https://github.com/introlab/audio_utils/blob/main/msg/AudioFrame.msg)): The audio topic to record.

#### Services
 - `video_recorder/filter_state` ([hbba_lite/SetOnOffFilterState](../../hbba_lite/srv/SetOnOffFilterState.srv)): The HBBA filter state service to enable or disable the recording.

### `perception_logger_node`
This node logs the video analysis messages and the audio analysis messages into a SQLite database.

#### Parameters
 - `database_path` (string): The database path of the SQLite database.
 - `frame_id` (string): The frame id of the positions and directions in the database.

#### Subscribed Topics
 - `video_analysis` ([video_analyzer/VideoAnalysis](../../perceptions/video_analyzer/msg/VideoAnalysis.msg)): The video analysis containing the detected objects. The video analysis must contain 3d positions.
 - `audio_analysis` ([audio_analyzer/AudioAnalysis](../../perceptions/audio_analyzer/msg/AudioAnalysis.msg)): The audio analysis containing the audio classes, voice embedding and the sound direction.
 - `talk/text` ([talk/Text](../../behaviors/talk/msg/Text.msg)): The text spoken by the robot.
 - `speech_to_text/text` ([speech_to_text/Text](../../perceptions/speech_to_text/msg/Transcript.msg)): The text spoken by the human.
