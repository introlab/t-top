#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


import json
import os
import re

from abc import ABC, abstractmethod
from datetime import datetime
import rclpy
import rclpy.callback_groups
import rclpy.node
import rclpy.executors

from behavior_msgs.msg import Text, Done, ChatToolsFunctionCall
from behavior_srvs.srv import ChatToolsFunctionCall

from perception_msgs.msg import Transcript
from ament_index_python.packages import get_package_share_directory



import openai

class ModelNotFoundError(Exception):
    """ Exception raised when the model is not found """
    def __init__(self, model_name: str):
        super().__init__(f'Model {model_name} not found')
        self.model_name = model_name

    def __str__(self):
        return f'Model {self.model_name} not found'

class ToolsNotFoundError(Exception):
    """ Exception raised when the tool is not found """
    def __init__(self, tools_name: str):
        super().__init__(f'Tool {tools_name} not found')
        self.tools_name = tools_name

    def __str__(self):
        return f'Tool {self.tools_name} not found'

class PromptsNotFoundError(Exception):
    """ Exception raised when the prompts are not found """
    def __init__(self, prompts_name: str):
        super().__init__(f'Prompts {prompts_name} not found')
        self.prompts_name = prompts_name

    def __str__(self):
        return f'Prompts {self.prompts_name} not found'


class BaseChatAPI(ABC):
    def __init__(self,  chat_node: rclpy.node.Node, language: str, language_model: str):
        self._chat_node = chat_node
        self.history = list()
        self.language = language
        self.language_model = language_model
        self._tools_schema = None
        self._prompts = None

    def add_to_history(self, message: str, role: str, timestamp: datetime):
        """ Add to history to conserve context """
        if len(message) > 0:
            self.history.append({"role": role, "content": message, "datetime": str(timestamp)})

    def add_tool_calls_to_history(self, tool_calls: list, timestamp: datetime):
        """ Add tool calls to history """
        self.history.append({"role": "assistant",
                             "tool_calls": tool_calls,
                             "datetime": str(timestamp)})

    def add_tool_call_response_to_history(self, tool_call: dict, result: dict, timestamp: datetime):
        """ Add tool call result to history """
        self.history.append({"role": "tool",
                             "tool_call_id": tool_call['id'],
                             "name": tool_call['function']['name'],
                             "content": json.dumps(result),
                             "datetime": str(timestamp)})

    def reset_history(self):
        """ Reset the history """
        self.history.clear()
        # Reload default context
        if (self._prompts is None):
            self.load_default_context()
        else:
            self._chat_node.get_logger().info('No prompts loaded, not resetting context')

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

    def load_prompts(self, file : str) -> bool:
        """ Load prompts from file """
        with open(file, 'r') as f:
            self._prompts = json.load(f)
            # Should be an array
            if isinstance(self._prompts, list):
                # Add to history
                for prompt in self._prompts:
                    if 'role' in prompt and 'content' in prompt:
                        self.add_to_history(message=prompt['content'],
                                            role=prompt['role'],
                                            timestamp=datetime.now())
                    else:
                        self._chat_node.get_logger().info('Invalid prompt format')
                        return False
            else:
                self._chat_node.get_logger().info('Invalid prompts format')
                return False
            return True
        return False

    def load_tools(self, file : str) -> bool:
        try:
            with open(file, 'r') as f:
                # TODO Validate schema
                self._tools_schema = json.load(f)

                return True
        except Exception as e:
            self._chat_node.get_logger().info(f'Failed to load tools: {e}')
            return False

        return False


    def get_request_messages(self) -> list:
        """ Get the messages to send to the server """
        messages = list()
        for message in self.history:
            # Copy message
            new_message = message.copy()
            # Discard timestamp for now
            new_message.pop('datetime', None)
            messages.append(new_message)

        return messages


    @abstractmethod
    def send_request_and_process_response(self):
        """ Send request to the server and get the response """
        pass


class ChatGPTAPI(BaseChatAPI):
    def __init__(self, chat_node: rclpy.node.Node, language: str, language_model: str):
        super().__init__(chat_node, language, language_model)
        openai.api_key = os.environ.get('OPENAI_API_KEY')

    def create_chat_completion(self):
        return openai.ChatCompletion.create(
            model=self.language_model,
            messages=self.get_request_messages(),
            max_tokens=1600,
            temperature=0.5, # Somewhat creative
            frequency_penalty=0.5, # Avoid repetition
            tools=self._tools_schema,
            tool_choice="auto",
            top_p=0.9, # Avoid repetition
            stream=True # Enable streaming mode
        )

    def send_request_and_process_response(self):
        try:
            response = self.create_chat_completion()

            assistant_message = ""

            # Generator will give partial responses
            for chunk in response:
                # print(chunk)
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    # Get delta
                    delta = chunk['choices'][0].get('delta', {})

                    # Process function calls
                    if 'tool_calls' in delta:
                        for tool_call in delta['tool_calls']:
                            tool_type = tool_call.get('type', None)
                            if tool_type == 'function':

                                if 'name' in tool_call['function']:
                                    # Call function, will be handled by chat_node and published to be processed externally
                                    id = tool_call.get('id', None)
                                    function_name = tool_call['function'].get('name', None)
                                    function_arguments = tool_call['function'].get('arguments', None)
                                    self.add_tool_calls_to_history([tool_call], timestamp=datetime.now())

                                    # Call service with
                                    response : ChatToolsFunctionCall.Response = self._chat_node.call_tools_external_service(id, function_name, function_arguments)

                                    if response is not None:
                                        # TODO Add response to history
                                        # TODO Call back API with response
                                        pass
                                    else:
                                        self._chat_node.get_logger().error(f'Failed to call external service: {function_name}')
                                        self.add_tool_call_response_to_history(tool_call=tool_call,
                                                                               result={"error": "Failed to call external service"},
                                                                               timestamp=datetime.now())

                                    # TODO Call back API with response ?




                    # Process normal messages
                    if 'content' in delta and delta['content'] is not None:
                        content = delta['content']
                        if len(content) == 0:
                            continue

                        # Add frangement to output message
                        assistant_message += content
                        # Send fragment to be processed
                        self._chat_node.add_pending_message(content)


            # Add full output message to the history
            if len(assistant_message) > 0:
                # Add to history
                self.add_to_history(message=assistant_message, role='assistant', timestamp=datetime.now())

        except Exception as e:
            self._chat_node.get_logger().error(f"Error: {e}")
            self._chat_node.add_pending_message(str(e))


class OllamaAPI(ChatGPTAPI):
    def __init__(self, chat_node: rclpy.node.Node, language: str, language_model: str):
        super().__init__(chat_node, language, language_model)
        openai.api_key = 'ollama'
        openai.api_base = "http://localhost:11434/v1"

    def create_chat_completion(self):
        return openai.ChatCompletion.create(
            model=self.language_model,
            messages=self.get_request_messages(),
            tools=self._tools_schema,
            tool_choice="auto",
            top_p=0.9, # Avoid repetition
            stream=True # Enable streaming mode
        )

class ChatNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('chat_node')

        self._talking = False
        self._processing = False
        self._tools_calls = list()
        self._pending_messages = list()
        self._executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
        # This will allow to receive callbacks while processing another one
        self._callback_group = rclpy.callback_groups.ReentrantCallbackGroup()

        self._language = self.declare_parameter('language', 'fr').get_parameter_value().string_value
        #self._language_model = self.declare_parameter('language_model', 'llama3.2').get_parameter_value().string_value
        #self._model_type = self.declare_parameter('model_type', 'ollama').get_parameter_value().string_value

        self._language_model = self.declare_parameter('language_model', 'gpt-4o-mini').get_parameter_value().string_value
        self._model_type = self.declare_parameter('model_type', 'chatgpt').get_parameter_value().string_value

        # Tools
        self._enable_tools = self.declare_parameter('enable_tools', True).get_parameter_value().bool_value
        self._tools_file = self.declare_parameter('tools_config',
                                                  get_package_share_directory('chat') + '/tools/default_tools.json').get_parameter_value().string_value

        # Prompts
        self._enable_prompts = self.declare_parameter('enable_prompts', True).get_parameter_value().bool_value
        self._prompts_file = self.declare_parameter('prompts_config',
                                                    get_package_share_directory('chat') + f'/prompts/default_{self._language}.json').get_parameter_value().string_value


        # Initialize API
        if self._model_type == 'ollama':
            self._chat_api = OllamaAPI(self, language=self._language, language_model=self._language_model)
        elif self._model_type == 'chatgpt':
            self._chat_api = ChatGPTAPI(self, language=self._language, language_model=self._language_model)
        else:
            raise ModelNotFoundError(self._model_type)

        # Load tools
        if self._enable_tools:
            if not self._chat_api.load_tools(self._tools_file):
                self.get_logger().error('Failed to load tools')
                raise ToolsNotFoundError(self._tools_file)
        # Load prompts
        if self._enable_prompts:
            if not self._chat_api.load_prompts(self._prompts_file):
                self.get_logger().error('Failed to load prompts')
                raise PromptsNotFoundError(self._prompts_file)
        else :
            self.get_logger().info('Prompts not enabled')
            self.get_logger().info('Using default prompts')
            self._chat_api.load_default_context()


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
        # self._chat_tools_functions_pub = self.create_publisher(ChatToolsFunctionCall, 'chat/tools/functions', 1)


    def call_tools_external_service(self, id: str, function_name: str, function_arguments: str) -> str:
        """ Call the external service to process the tool function call """
        self.get_logger().info(f'Calling external service: {function_name} with arguments: {function_arguments}')

        # Create client
        client = self.create_client(ChatToolsFunctionCall, f'/chat/tools/functions/{function_name}')

        # Create Request
        request : ChatToolsFunctionCall.Request = ChatToolsFunctionCall.Request()

        # Fill request
        request.id = id
        request.function_name = function_name
        request.function_arguments = function_arguments

        # Wait for service to be available
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Service {client.srv_name} not available')
            return None

        # Call and wait for service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is not None:
            self.get_logger().info(f'Service call result: {future.result()}')
            return future.result()
        else:
            self.get_logger().error(f'Service call failed: {future.exception()}')
            return None
        return None

    def add_pending_message(self, message: str):
        self._pending_messages.append(message)
        self._process_pending_messages()


    def _on_transcript_received_cb(self, msg: Transcript):
        self.get_logger().info(f'Transcript received: {msg.text}')

        if len(msg.text) == 0:
            self.get_logger().error('Empty transcript')

        self._talking = False
        self._processing = False

        if len(msg.text) > 0:
            # Add the transcript to the context history
            self._chat_api.add_to_history(message=msg.text,
                                        role='user', timestamp=datetime.now())

            # Process the request
            self._processing = True
            self.get_logger().info('Processing...')
            self._chat_api.send_request_and_process_response()
            self._processing = False
            self.get_logger().info('Processing done!')

        else:
            self.get_logger().error('Empty transcript')

        # Safety always call _process_pending_messages
        self._process_pending_messages()

    def _on_talk_done_cb(self, msg: Done):
        self.get_logger().info(f'Talk done : {msg.ok}')
        self._talking = False
        self._process_pending_messages()

    def _on_tools_function_call_service_cb(self, request, response):
        self.get_logger().info(f'Tools function call received: {request}')

        # Find the tool call in the list
        for tool_call in self._tools_calls:
            if tool_call.id == request.id:
                # Remove the tool call from the list
                self._tools_calls.remove(tool_call)
                break

        # Add the result to the history
        self._chat_api.add_tool_call_response_to_history(tool_call=tool_call,
                                                         result=request.result,
                                                         timestamp=datetime.now())

        # Send to API

        # Send response to client
        response.ok = True
        response.message = 'Tool call processed'
        return response


    def _process_pending_messages(self):
        if not self._talking:
            partial_message: str = ""
            for message in self._pending_messages:
                partial_message += message
            self._pending_messages.clear()

            # To be done we need to have no pending messages and no tools calls
            if len(partial_message) == 0:
                if len(self._tools_calls) == 0:
                    # We are done
                    chat_msg = Done()
                    chat_msg.ok = True
                    self.get_logger().info('Chat done')
                    self._chat_done_pub.publish(chat_msg)

                return

            talk_msg = Text()

            # Remove all the text between the <think> </think> tags
            partial_message = re.sub(r'<think>.*?</think>', '', partial_message, flags=re.DOTALL)

            # Avoid "*" because TTS will say "Asterisk"
            # TODO find a better replacement
            partial_message= partial_message.replace("*", '-')

            if not self._processing:
                # Send everything, we are done!
                self._talking = True
                talk_msg.text = partial_message
                self.get_logger().info(f'Sending talk message: {talk_msg.text}')
                self._talk_text_pub.publish(talk_msg)
            else:
                # Send the first phrase and buffer the rest
                # Split the message into sentences based on punctuation marks .?!
                # Match text with optional punctuation
                sentences = re.findall(r'[^!.?]+[!.?]?', partial_message)

                # We are talking if we found at least one sentence
                if len(sentences) > 1:
                    self._talking = True
                    talk_msg.text = sentences[0]
                    self.get_logger().info(f'Sending talk message: {talk_msg.text}')
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
    except ModelNotFoundError as e:
        chat_node.get_logger().error(f'{e}')
    finally:
        chat_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
