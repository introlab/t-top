#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import requests
import json
import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
import rclpy
import rclpy.callback_groups
import rclpy.node
import rclpy.executors

from std_msgs.msg import Float32
from behavior_msgs.msg import Text, Done, Statistics
from perception_msgs.msg import Transcript
from audio_utils_msgs.msg import AudioFrame

import hbba_lite
import time_utils
import openai

openai.api_key = os.environ.get('OPENAI_API_KEY')

class BaseChatAPI(ABC):
    def __init__(self,  chat_node: rclpy.node.Node, language: str, language_model: str):
        self._chat_node = chat_node
        self.history = list()
        self.language = language
        self.language_model = language_model
        self.load_default_context()

    def add_to_history(self, message: str, role: str, timestamp: datetime):
        """ Add to history to conserve context """
        if len(message) > 0:
            self.history.append({"role": role, "content": message, "datetime": str(timestamp)})

    def reset_history(self):
        """ Reset the history """
        self.history.clear()
        # Reload default context
        self.load_default_context()

    def load_default_context(self):
        """ Load default context """
        if self.language == 'fr':
            self.add_to_history(message="Vous êtes un robot assistant. Vous répondez toujours en français.",
                                        role="system",
                                        timestamp=datetime.now())
        else:
            self.add_to_history(message="You are a robot assistant. You always answer in English.",
                                        role="system",
                                        timestamp=datetime.now())

    def get_request_messages(self) -> list:
        """ Get the messages to send to the server """
        messages = list()
        for message in self.history:
            # Discard timestamp for now
            messages.append({"role": message["role"], "content": message["content"]})

        return messages

    @abstractmethod
    def send_request_and_process_response(self):
        """ Send request to the server and get the response """
        pass


class ChatGPTAPI(BaseChatAPI):
    def __init__(self, chat_node: rclpy.node.Node, language: str, language_model: str):
        super().__init__(chat_node, language, language_model)

        # Available functions
        self._available_functions = {
            "volume_up":{
                "en": "Raising volume",
                "fr": "Je monte le volume",
                "function": self._volume_up
            },
            "volume_down": {
                "en": "Lowering volume",
                "fr": "Je baisse le volume",
                "function": self._volume_down
            }
        }


        self._function_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "volume_up",
                    "description": "Increase the volume",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "volume_down",
                    "description": "Decrease the volume",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]


    def _volume_up(self):
        print('Volume up')

    def _volume_down(self):
        print('Volume down')

    def send_request_and_process_response(self):
        try:
            response = openai.ChatCompletion.create(
                model=self.language_model,
                messages=self.get_request_messages(),
                max_tokens=1600,
                temperature=0.5, # Somewhat creative
                frequency_penalty=0.5, # Avoid repetition
                tools=self._function_schemas,
                tool_choice="auto",
                top_p=0.9, # Avoid repetition
                stream=True # Enable streaming mode
            )

            assistant_message = ""

            # Generator will give partial responses
            for chunk in response:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    # Get delta
                    delta = chunk['choices'][0].get('delta', {})

                    # Process function calls
                    if 'tool_calls' in delta:
                        # print(delta)
                        for tool_call in delta['tool_calls']:
                            function_index = tool_call.get('index', 0)
                            # print('Function index:', function_index)
                            tool_type = tool_call.get('type', None)

                            if tool_type == 'function':
                                if tool_call['function']['name'] in self._available_functions:
                                    function_info = self._available_functions[tool_call['function']['name']]

                                    # Get function description
                                    if self.language == 'fr':
                                        function_description = function_info['fr']
                                    else:
                                        function_description = function_info['en']

                                    self._chat_node.add_pending_message(function_description)

                                    # Call function
                                    function_info['function']()

                    # Process normal messages
                    if 'content' in delta and delta['content'] is not None:
                        content = delta['content']
                        # Add frangement to output message
                        assistant_message += content
                        # Send fragment to be processed
                        self._chat_node.add_pending_message(content)


            # Add full output message to the history
            self.add_to_history(message=assistant_message, role='assistant', timestamp=datetime.now())


        except Exception as e:
            print('Error:', e)
            self._chat_node.add_pending_message(str(e))

class OllamaAPI(BaseChatAPI):
    def __init__(self, chat_node: rclpy.node.Node,  language: str, language_model: str):
        super().__init__(chat_node, language, language_model)
        self._server_url = "http://localhost:11434/api/chat"

    def send_request_and_process_response(self):
        data = {
            "model": self.language_model,
            "messages": self.get_request_messages(),
            "stream": True
        }

        response = requests.post(self._server_url, json=data, stream=True)
        if response.status_code == 200:

            output_message: str = str()

            for line in response.iter_lines():
                if line:
                    json_obj = json.loads(line.decode("utf-8"))

                    if 'message' in json_obj and json_obj['message']['role'] == 'assistant':
                        # Chat node will add the message to the pending messages to send to the talk node
                        self._chat_node.add_pending_message(json_obj['message']['content'])
                        output_message += json_obj['message']['content']

                    if 'done' in json_obj:
                        if json_obj['done']:
                            # Add full output message to the history
                            self.add_to_history(message=output_message,
                                            role='assistant',
                                            timestamp=datetime.now())
                            break


class ChatNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('chat_node')

        self._talking = False
        self._processing = False
        self._pending_messages = list()

        self._executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
        # This will allow to receive callbacks while processing another one
        self._callback_group = rclpy.callback_groups.ReentrantCallbackGroup()

        self._language = self.declare_parameter('language', 'fr').get_parameter_value().string_value
        self._language_model = self.declare_parameter('language_model', 'llama3.2').get_parameter_value().string_value

        # Testing Ollama API
        # self._chat_api = OllamaAPI(self, language=self._language, language_model=self._language_model)
        self._chat_api = ChatGPTAPI(self, language=self._language, language_model='gpt-4o-mini')

        # Subscribers
        self._transcript_sub = self.create_subscription(Transcript,
                                                        'speech_to_text/transcript',
                                                        self._on_transcript_received_cb,
                                                        1,
                                                        callback_group=self._callback_group)

        self._talk_done_sub = self.create_subscription(Done,
                                                       'talk/done',
                                                       self._on_talk_done_cb,
                                                       1,
                                                       callback_group=self._callback_group)

        # Publishers
        self._talk_text_pub = self.create_publisher(Text, 'talk/text', 1)
        self._chat_done_pub = self.create_publisher(Done, 'chat/done', 1)

    def add_pending_message(self, message: str):
        self._pending_messages.append(message)
        self._process_pending_messages()


    def _on_transcript_received_cb(self, msg: Transcript):
        print('Transcript received:', msg.text)

        # Add the transcript to the context history
        self._chat_api.add_to_history(message=msg.text, role='user', timestamp=datetime.now())
        # Process the request
        self._processing = True
        self._chat_api.send_request_and_process_response()
        self._processing = False

    def _on_talk_done_cb(self, msg: Done):
        print('Talk done:', msg.ok)
        self._talking = False
        self._process_pending_messages()

        if not self._processing and len(self._pending_messages) == 0 and not self._talking:
            # Send the output message to the chat node
            chat_msg = Done()
            chat_msg.ok = True
            print('Chat done')
            self._chat_done_pub.publish(chat_msg)


    def _process_pending_messages(self):
        if not self._talking:
            partial_message: str = ""
            for message in self._pending_messages:
                partial_message += message
            self._pending_messages.clear()

            if len(partial_message) == 0:
                return

            talk_msg = Text()

            # Remove all the text between the <think> </think> tags
            partial_message = re.sub(r'<think>.*?</think>', '', partial_message, flags=re.DOTALL)

            # Avoid "*" because TTS will say "Asterisk"
            # TODO find a better replacement
            partial_message= partial_message.replace("*", '...')

            if not self._processing:
                # Send everything, we are done!
                self._talking = True
                talk_msg.text = partial_message
                print('Sending talk message: ', talk_msg.text)
                self._talk_text_pub.publish(talk_msg)
            else:
                # Send the first phrase and buffer the rest
                # Split the message into sentences based on punctuation marks .?!
                sentences = re.findall(r'[^!.?]+[!.?]?', partial_message)  # Match text with optional punctuation

                if len(sentences) > 0:
                    self._talking = True
                    talk_msg.text = sentences[0]
                    print('Sending talk message: ', talk_msg.text)
                    self._talk_text_pub.publish(talk_msg)

                    # Remove the first sentence from the list
                    sentences.pop(0)

                # Re-Add the remaining sentences to the pending messages
                for sentence in sentences:
                    self._pending_messages.append(sentence)

    def run(self):
        self._executor.add_node(self)
        self._executor.spin()


def main():
    rclpy.init()
    chat_node = ChatNode()

    try:
        chat_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        chat_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
