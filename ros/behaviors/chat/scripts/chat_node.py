#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import requests
import json
import re
import time

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


class ChatNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('chat_node')

        self._executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
        self._callback_group = rclpy.callback_groups.ReentrantCallbackGroup()

        # Default Ollama server URL
        self._server_url = "http://localhost:11434/api/chat"
        self._context = list()
        self._output_message = ""
        self._language = self.declare_parameter('language', 'fr').get_parameter_value().string_value
        self._language_model = self.declare_parameter('language_model', 'llama3.2').get_parameter_value().string_value
        # TODO Interface with Ollama server or ChatGPT

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

        self._talking = False
        self._pending_messages = list()

        self._generate_default_context(self._language)

    def _generate_default_context(self, language: str):
        if language == 'fr':
            self._context.append({"role": "system",
                                  "content": "Vous êtes un robot assistant. Vous répondez toujours en français."})
        else:
            self._context.append({"role": "system",
                                  "content": "You are a robot assistant. You always answer in English."})

    def _on_transcript_received_cb(self, msg: Transcript):
        print('Transcript received:', msg.text)

        # Add the transcript to the context
        self._context.append({"role": "user", "content": msg.text})

        data = {
            "model": self._language_model,
            "messages": self._context
            # "stop": ["<think></think>"]
        }

        self._processing = True
        print("Sending request to the server...")

        # Send request with streaming
        response = requests.post(self._server_url, json=data, stream=True)

        output_message = ""

        # Remove all the text between the <think> </think> tags
        def remove_think_tags(text: str):
            return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    json_obj = json.loads(line.decode("utf-8"))
                    if 'message' in json_obj and json_obj['message']['role'] == 'assistant':
                        output_message += json_obj['message']['content']
                        # print('New message:', json_obj['message']['content'])
                        self._pending_messages.append(remove_think_tags(json_obj['message']['content']))

                        # Verify talking status, if not talking, send the buffered messages.
                        if not self._talking:
                            self._process_pending_messages()

                        if 'done' in json_obj:
                            if json_obj['done']:
                                self._context.append({"role": "assistant", "content": output_message})
                                self._processing = False
                                print('Processing done')
                                break

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

            if not self._processing:
                # Send everything, we are done!
                self._talking = True
                talk_msg.text = partial_message.replace("*", '(...)')
                print('Sending talk message: ', talk_msg.text)
                self._talk_text_pub.publish(talk_msg)
            else:
                # Send the first phrase and buffer the rest
                # Split the message into sentences based on punctuation marks .?!
                sentences = re.findall(r'[^!.?]+[!.?]?', partial_message)  # Match text with optional punctuation

                if len(sentences) > 1:
                    self._talking = True
                    # Remove all "*" in the string
                    talk_msg.text = sentences[0].replace("*", '(...)')
                    print('Sending talk message: ', talk_msg.text)
                    self._talk_text_pub.publish(talk_msg)

                    # Remove the first sentence from the list
                    sentences.pop(0)

                # Add the remaining sentences to the pending messages
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
