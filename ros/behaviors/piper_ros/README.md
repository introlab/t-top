# talk

This folder contains the node to generate speech files with [Piper](https://github.com/rhasspy/piper).
The [models](models) folder contains models trained by [Piper](https://github.com/rhasspy/piper). The license of the models is [MIT](models/MODEL_LICENSE).

## Nodes

### `piper_node`

This node generates speech files with [Piper](https://github.com/rhasspy/piper).

#### Parameters

- `use_gpu_if_available` (bool): Indicates whether to use the GPU or not. The default value is false.

#### Services

- `piper/generate_speech_from_text` ([behavior_srvs/GenerateSpeechFromText](../behavior_srvs/srv/GenerateSpeechFromText.srv)):
  The service to generate speech files.
