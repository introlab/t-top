# talk

This folder contains the node to generate speech files with [Piper](https://github.com/rhasspy/piper).
The [models](models) folder contains models trained by [Piper](https://github.com/rhasspy/piper).

## Nodes

### `piper_node`

This node generates speech files with [Piper](https://github.com/rhasspy/piper).

#### Services

- `piper/generate_speech_from_text` ([piper_ros/GenerateSpeechFromText](srv/GenerateSpeechFromText.srv)):
  The service to generate speech files.
