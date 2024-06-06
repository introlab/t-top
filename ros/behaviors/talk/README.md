# talk

This folder contains the node to make T-Top talk.

## Nodes

### `talk_node.py`

This node makes T-Top talk and move its lips accordingly. It uses Google Cloud Text-to-Speech.

#### Parameters

- `language` (string): The language (en or fr). The default value is en.
- `gender` (string): The gender (female or male). The default value is male.
- `speaking_rate` (doule): The speaking rate (range: [0.25, 4.0]). The default value is 1.0.
- `generator_type` (string): The generator type (google or piper). The Google generator uses the cloud and the Piper generator does not. The default value is piper.
- `mouth_signal_gain` (double): The gain to apply after the filter calculating the mouth signal. The default value is 0.04.
- `sampling_frequency` (int): The output sampling frequency. The default value is 16000.
- `frame_sample_count` (double): The number of sample in each audio frame. The default value is 1024.
- `done_delay_s` (double): The delay before sending the talk done message. The default value is 0.5.

#### Subscribed Topics

- `talk/text` ([behavior_msgs/Text](../behavior_msgs/msg/Text.msg)): The text to be said.

#### Published Topics

- `face/mouth_signal_scale` ([std_msgs/Float32](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32.html)):
  Indicates how much the mouth must be open.
- `audio_out` ([audio_utils/AudioFrame](https://github.com/introlab/audio_utils/blob/main/msg/AudioFrame.msg)): The
  speech sound (mono, float format).
- `talk/done` ([behavior_msgs/Done](../behavior_msgs/msg/Done.msg)): Indicates that the speech is finished.

#### Services

- `audio_out/filter_state` ([hbba_lite_srvs/SetOnOffFilterState](../../utils/hbba_lite/hbba_lite_srvs/srv/SetOnOffFilterState.srv)): The HBBA
  filter state service to enable or disable the behavior.
