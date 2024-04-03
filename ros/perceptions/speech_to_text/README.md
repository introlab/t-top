# speech_to_text
This folder contains the nodes performing speech to text.

## Nodes
### `google_speech_to_text_node.py`
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
 - `audio_in/filter_state` ([hbba_lite/SetOnOffFilterState](../../hbba_lite/srv/SetOnOffFilterState.srv)): The HBBA filter state service to enable or disable the processing.

### `whisper_speech_to_text_node.py`
The node performs speech to text. It uses [Faster Whisper](https://github.com/guillaumekln/faster-whisper).

#### Parameters
 - `language` (string): The language (en or fr).
 - `model_size` (string): The [Whisper model size](https://github.com/openai/whisper#available-models-and-languages)
 (tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large or large.en).
 - `device` (string): The device that executes the model (cpu or cuda)
 - `compute_type` (string): The compute type (int8, float16 or float32).
 - `prebuffering_frame_count` (int): Number of frames to be accumulated before processing (default=4)
 - `minimum_voice_sequence_size` (int): Minimum number of voice samples to be processed (default=8000)


#### Subscribed Topics
 - `voice_activity` ([audio_utils/VoiceActivity](https://github.com/introlab/audio_utils/blob/main/msg/VoiceActivity.msg)): The output of the voice activity detector.
 - `audio_in` ([audio_utils/AudioFrame](https://github.com/introlab/audio_utils/blob/main/msg/AudioFrame.msg)): The sound topic processed.

#### Published Topics
 - `transcript` ([speech_to_text/Transcript](msg/Transcript.msg)): The text said.

#### Services
 - `audio_in/filter_state` ([hbba_lite/SetOnOffFilterState](../../hbba_lite/srv/SetOnOffFilterState.srv)): The HBBA filter state service to enable or disable the processing.
