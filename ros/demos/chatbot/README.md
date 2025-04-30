 # Chatbot Demo

This demo showcases a simple chatbot using the T-Top framework. The chatbot is designed to respond to user queries and provide information on various topics using the embedded speech-to-text (STT) and text-to-speech (TTS) modules. The demo uses a large language model (LLM) to generate responses and can be configured to use either a local model or the OpenAI API.

# Implementation

The demo is implemented using the [chatbot_node.cpp](src/chatbot_node.cpp) which is a ROS2 node that handles the interaction with the user. We use `hbba_lite` to manage the desires and strategies of the chatbot.

There are three main desires in the demo :
* ChatDesire: This desire is responsible for handling the chat interactions with the user. It uses the LLM to generate responses based on the user's input. LLM is handled in the chat behavior node. It uses the `ChatStrategy` to process the user's input and generate a response.
* NearestFaceFollowingDesire: This desire is responsible for following the nearest face detected by the perception module. It uses the `NearestFaceFollowingStrategy` to achieve this.
* TooCloseReactionDesire: This desire is responsible for reacting when the user is too close to the robot. It uses the `TooCloseReactionStrategy` to achieve this.

The main goal is to have the robot change its state based on the user's input from listening (STT) and speaking (TTS). The robot will listen to the user and respond using the LLM. The robot will also follow the user if they are too close or if they are not facing the robot and send led information to communicate its state. When listening, the robot will have "green" rotating leds and when speaking, the robot will have "red" rotating leds.

Listening --> Chat Behavior --> Speaking --> Listening --> Chat Behavior --> Speaking --> (never ending loop)...

When tools are enabled, the LLM can call external services to perform actions. The tools are defined in a JSON file and can be configured to call any service in the T-Top framework. The tools are called dynamically using the service name and the service message. The tools can be used to perform actions. A demonstration of `volume_up`and `volume_down` function calling is provided in the demo.

`volume_up`: This function increases the volume of the robot. It is called when the user says "volume up" or "increase volume". The service name is `chat/tools/functions/volume_up` and the service message is `behavior_srvs/ChatToolsFunctionCall.srv`. The service must return a JSON response to be sent to the LLM.
`volume_down`: This function decreases the volume of the robot. It is called when the user says "volume down" or "decrease volume". The service name is `chat/tools/functions/volume_down` and the service message is `behavior_srvs/ChatToolsFunctionCall.srv`. The service must return a JSON response to be sent to the LLM.



## Launching the demo
To launch the demo, use the following command:

```bash
# Make sure DISPLAY is set to :0
export DISPLAY=:0
# Source ROS2 and the workspace
source /opt/ros/humble/install/setup.bash
source <your workspace>/install/setup.bash
# Launch the demo
ros2 launch chatbot chatbot.launch.xml
```
