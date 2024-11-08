# sound_player

This folder contains the node to play sound files

## Nodes

### `sound_player_node.py`

This node plays sound files.

#### Parameters

- `sampling_frequency` (int): The output sampling frequency. The default value is 16000.
- `frame_sample_count` (double): The number of sample in each audio frame. The default value is 1024.

#### Subscribed Topics

- `sound_player/file` ([behavior_msgs/SoundFile](../behavior_msgs/msg/SoundFile.msg)): The sound file to play.

#### Published Topics

- `audio_out` ([audio_utils_msgs/AudioFrame](https://github.com/introlab/audio_utils/blob/ros2/audio_utils_msgs/msg/AudioFrame.msg)): The
  sound (mono, float format).
- `sound_player/done` ([behavior_msgs/Done](../behavior_msgs/msg/Done.msg)): Indicates that the speech is finished.

#### Services

- `audio_out/filter_state` ([hbba_lite_srvs/SetOnOffFilterState](../../utils/hbba_lite/hbba_lite_srvs/srv/SetOnOffFilterState.srv)): The HBBA
  filter state service to enable or disable the behavior.
