# audio_analyzer
This folder contains the node to analyze the audio.

## Nodes
### `audio_analyzer_node.py`
This node classifies the audio and extracts a general audio embedding. If the audio contains a voice, it extracts a voice embedding.

#### Parameters
 - `inference_type` (string): Indicates where to run the neural network (cpu, torch_gpu or trt_gpu).
 - `interval` (int): The interval between inferences in sample (low_quality_audio_in).
 - `voice_probability_threshold` (double): The voice probability threshold for the voice embedding.
 - `class_probability_threshold` (double): The class probability threshold for the audio analysis.

#### Subscribed Topics
 - `audio_in` ([audio_utils/AudioFrame](https://github.com/introlab/audio_utils/blob/main/msg/AudioFrame.msg)): The sound topic processed, which must be signed_16, at 16000 Hz and 1 channel.
 - `sst` ([odas_ros/OdasSstArrayStamped](https://github.com/introlab/odas_ros/blob/main/msg/OdasSstArrayStamped.msg)): The sound source tracking information.

#### Published Topics
 - `audio_analysis` ([audio_analyzer/AudioAnalysis](msg/AudioAnalysis.msg)): The audio analysis containing the audio classes, general audio embedding, voice embedding and the sound direction.

#### Services
 - `audio_in/filter_state` ([hbba_lite/SetOnOffFilterState](../../hbba_lite/srv/SetOnOffFilterState.srv)): The HBBA filter state service to enable or disable the processing.
