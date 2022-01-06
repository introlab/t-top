# talk
This folder contains the node to make T-Top talk.

## Nodes
### `talk_node.py`
This node makes T-Top talk and move its lips accordingly. It uses Google Cloud Text-to-Speech.

#### Parameters
 - `language` (string): The language (en or fr).
 - `mouth_signal_gain` (double): The gain to apply after the filter calculating the mouth signal.
 - `sampling_frequency` (int): The output sampling frequency.
 - `frame_sample_count` (double): The number of sample in each audio frame.

#### Subscribed Topics
- `talk/text` ([talk/Text](msg/Text.msg)): The text to be said.

#### Published Topics
 - `face/mouth_signal_scale` ([std_msgs/Float32](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32.html)): Indicates how much the mouth must be open.
 - `audio_out` ([audio_utils/AudioFrame](https://github.com/introlab/audio_utils/blob/main/msg/AudioFrame.msg)): The speech sound.
 - `talk/done` ([talk/Done](msg/Done.msg)): Indicates that the speech is finished.

#### Services
 - `audio_out/filter_state` ([hbba_lite/SetOnOffFilterState](../../hbba_lite/srv/SetOnOffFilterState.srv)): The HBBA filter state service to enable or disable the behavior.
