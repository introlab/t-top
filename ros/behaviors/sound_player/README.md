# sound_player

This folder contains the node to play sound files

## Nodes

### `sound_player_node.py`

This node plays sound files.

#### Parameters

- `sampling_frequency` (int): The output sampling frequency.
- `frame_sample_count` (double): The number of sample in each audio frame.

#### Subscribed Topics

- `sound_player/file` ([sound_player/SoundFile](msg/SoundFile.msg)): The sound file to play.

#### Published Topics

- `audio_out` ([audio_utils/AudioFrame](https://github.com/introlab/audio_utils/blob/main/msg/AudioFrame.msg)): The
  sound (mono, float format).
- `sound_player/done` ([sound_player/Done](msg/Done.msg)): Indicates that the speech is finished.

#### Services

- `audio_out/filter_state` ([hbba_lite/SetOnOffFilterState](../../hbba_lite/srv/SetOnOffFilterState.srv)): The HBBA
  filter state service to enable or disable the behavior.
