# speech_to_text
This folder contains the node performing speech to text.

## Nodes
### `speech_to_text_node.py`
The node performs speech to text. It uses Google Cloud Speech-to-Text.

#### Parameters
 - `sampling_frequency` (int): The input sampling frequency.
 - `frame_sample_count` (int): The number of samples in each frame.
 - `request_frame_count` (int): The number of frame in each request.
 - `language` (string): The language (en or fr).

#### Subscribed Topics
 - `audio_in` ([audio_utils/AudioFrame](https://github.com/introlab/audio_utils/blob/main/msg/AudioFrame.msg)): The sound topic processed.

#### Published Topics
 - `transcript` ([speech_to_text/Transcript](msg/Transcript.msg)): The text said.

#### Services
 - `audio_in/filter_state` ([hbba_lite/SetOnOffFilterState](../../hbba_lite/srv/SetOnOffFilterState.srv)) :The HBBA filter state service to enable or disable the processing.
