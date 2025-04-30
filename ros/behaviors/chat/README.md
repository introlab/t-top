# chat

This folder contains the node to make T-Top chat (LLM based). The node can be used with a local model or using the OpenAI API (paid service). Performance may vary depending on the model used and the embedded computer performance if ran locally.

## `chat_node.py`

This ROS2 node allows T-Top to communicate with the user using a large language model (LLM). It uses either an OpenAI model or a local model using Ollama. The local Ollama API is using the same chat completion API endpoints as the OpenAI API. Instead of connecting to external servers, it connects to the local ollama server at http://localhost:11434.

To reduce latency, the model is called in streaming mode. The node is designed to output the text to be spoken by the TTS module incrementally. Tools are called dynamically using the service name and the service message. The tools can be configured using a JSON file. Initial context can also be configured with a JSON file.

The `chat_node.py` is designed to be used in conjunction with a speech-to-text (STT) module and a text-to-speech (TTS) module. Listening or speaking is performed using an external HBBA node which controls desires, strategies and filters. A demo application can be found in the [demos/chatbot](../../demos/chatbot/README.md) folder.

### Requirements

- `openai` (Python package): The OpenAI API client.
```bash
# Python package dependencies are handled with rosdep
rosdep install --from-paths src/t-top/ros/behaviors/chat/ --ignore-src -r -y
```
- `ollama` (optional) : Ollama can be installed from [ollama.com](https://ollama.com/). Local models can be downloaded from [ollama.com/models](https://ollama.com/models). [Installation instructions](https://ollama.com/download/linux) are available on the website.
```bash
# Install the ollama CLI, this will automatically setup the Jetson with an embedded server and API.
curl -fsSL https://ollama.com/install.sh | sh
# Download the model
ollama pull <model_name>
# You can test the model with the following command
ollama run <model_name> "Hello, how are you?"
```

> If you plan to use the OpenAI API, you need to set the `OPENAI_API_KEY` environment variable with your OpenAI API key. You can do this by running the following command in your terminal:
```bash
# This variable is already set on the T-Top's Jetson computer
export OPENAI_API_KEY=<your_openai_api_key>
```

### Parameters

- `language` (string) ['fr', 'en']. The language to use for the LLM. The default is 'fr' (French).
- `language_model` (string) default: 'gpt-4o-mini': The language model to use. The models can be found in the [ollama.com/models](https://ollama.com/models) website for local models or on [OpenAI website](https://platform.openai.com/docs/models). If you use ollama, make sure you download the model first using the `ollama pull <model_name>` command.
-  `model_type` (string) ['ollama', 'chatgpt']: The model type to use. The default is 'chatgpt'.
- `enable_tools` (bool) default: True: Enable or disable the tools. The tools are used to call functions from the LLM. The tools can be configured using a JSON file.
- `tools_config` (string) default: `get_package_share_directory("chat") + "/tools/default_tools.json"`: The JSON file containing the tools configuration. The default files should be in the package share directory. The structure of the JSON file must follow OpenAI's function calling format. See the [OpenAI documentation](https://platform.openai.com/docs/guides/gpt/function-calling) for more information.
- `enable_prompts` (bool) default: True: Enable or disable the prompts. The prompts are used to configure the LLM. The prompts can be configured using a JSON file.
- `prompts_config` (string) default: `get_package_share_directory("chat") + "/prompts/default_<language>.json"`: The JSON file containing the prompts configuration. The default files should be in the package share directory.

### Subscribed Topics

- `speech_to_text/transcript` ([perception_msgs/Transcript](../../perceptions/perception_msgs/msg/Transcript.msg)): The text output by the Speech-to-Text (STT) perception module. This is used to trigger the chat API (processing the output of the LLM).
- `talk/done` ([behavior_msgs/Done](../behavior_msgs/msg/Done.msg)): Sent when the Text To Speech (TTS) module is done talking. This is used to trigger the next action to be performed by the chat behavior.

### Published Topics

- `talk/text` ([behavior_msgs/Text](../behavior_msgs/msg/Text.msg)): The text to be spoken by the TTS module. This is used to trigger the TTS module.
- `chat/done` ([behavior_msgs/Done](../behavior_msgs/msg/Done.msg)): Sent when the chat session is completed.

### Services

- (TODO) Clear the chat history. This is used to clear the chat history when the user says "clear" or "reset". This can also be triggered by external nodes.
- (TODO) Add information to the chat history. This can be useful to add more context to the chat session. This could also be done via a topic. 

### Tools (Dynamic remote service calls)

If tools are enabled, external service calls are made using the following service name and using the following service message:
  - `chat/tools/functions/{function_name}` ([behavior_srvs/ChatToolsFunctionCall](../behavior_srvs/srv/ChatToolsFunctionCall.srv)): The service is dynamically called according to the function name defined in the tools configuration file. The service must answer within 5 seconds and return a JSON response to be sent to the LLM.
