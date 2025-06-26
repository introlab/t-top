#!/usr/bin/env python3

import json
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from threading import Event
from typing import List, Callable
from functools import reduce

import openai
from openai import OpenAI

import rclpy
import rclpy.callback_groups
import rclpy.executors
import rclpy.node
import rclpy.parameter
from rcl_interfaces.msg import SetParametersResult

from rclpy.qos import QoSProfile
from ament_index_python.packages import get_package_share_directory
from behavior_msgs.msg import Done, Text
from behavior_srvs.srv import ChatToolsFunctionCall
from perception_msgs.msg import ContextInput
import hbba_lite
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall


class ModelNotFoundError(Exception):
    """Exception raised when the model is not found"""

    def __init__(self, model_name: str):
        super().__init__(f"Model {model_name} not found")
        self.model_name = model_name

    def __str__(self):
        return f"Model {self.model_name} not found"


class ToolsNotFoundError(Exception):
    """Exception raised when the tool is not found"""

    def __init__(self, tools_name: str):
        super().__init__(f"Tool {tools_name} not found")
        self.tools_name = tools_name

    def __str__(self):
        return f"Tool {self.tools_name} not found"


class PromptsNotFoundError(Exception):
    """Exception raised when the prompts are not found"""

    def __init__(self, prompts_name: str):
        super().__init__(f"Prompts {prompts_name} not found")
        self.prompts_name = prompts_name

    def __str__(self):
        return f"Prompts {self.prompts_name} not found"


class BaseChatAPI(ABC):
    def __init__(
        self,
        chat_node: "ChatNode",
        language: str,
        language_model: str,
        streaming: bool,
        save_history: bool,
        save_history_path: str,
    ):
        self._chat_node = chat_node
        self.history = list()
        self.history_to_save = list()
        self.language = language
        self.language_model = language_model
        self._tools_schema = None
        self._prompts = None
        self._streaming = streaming
        self._save_history = save_history
        self._save_history_path = save_history_path

    def add_to_history(self, message: str, role: str, timestamp: datetime):
        """Add to history to conserve context"""
        if len(message) > 0:
            message = {"role": role, "content": message, "datetime": str(timestamp)}
            self.history.append(message)
            if self._save_history:
                self.save_history(message)

    def add_tool_calls_to_history(
        self, tool_call: list, timestamp: datetime, function_name: str
    ):
        """Add tool calls to history"""
        message = {
            "role": "assistant",
            "tool_calls": tool_call,
            "datetime": str(timestamp),
        }
        self.history.append(message)
        if self._save_history:
            message = {
                "role": "assistant",
                "tool_calls": tool_call.function.name,
                "datetime": str(timestamp),
            }
            self.save_history(message)

    def add_tool_call_result_to_history(
        self, tool_call: dict, result: dict, timestamp: datetime
    ):
        message = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": json.dumps(result),
            "datetime": str(timestamp),
        }
        self.history.append(message)
        if self._save_history:
            self.save_history(message)

    def reset_history(self):
        """Reset the history"""
        self.history.clear()
        self.history_to_save.clear()
        self._chat_node.get_logger().info("History Reset")

    def save_history(self, message):
        try:
            self.history_to_save.append(message)
            with open(self._save_history_path, "w") as f:
                json.dump(self.history_to_save, f, indent=4)
            self._chat_node.get_logger().info(
                f"History saved to {self._save_history_path}"
            )
        except Exception as e:
            self._chat_node.get_logger().error(f"Failed to save history: {e}")

    def load_default_context(self):
        """Load default context"""
        if self.language == "fr":
            self.add_to_history(
                message="Vous êtes un robot assistant. Vous répondez toujours en français. Puisque vous communiquerez à l'oral, veuillez répondre avec une ponctuation adéquate.",
                role="system",
                timestamp=datetime.now(),
            )
        else:
            self.add_to_history(
                message="You are a robot assistant. You always answer in English. Since you will be communicating orally, please respond with proper punctuation.",
                role="system",
                timestamp=datetime.now(),
            )

    def load_prompts_into_history(self, prompts: list) -> bool:
        """Load prompts from a dict"""
        if isinstance(prompts, list):
            for prompt in prompts:
                if "role" in prompt and "content" in prompt:
                    self.add_to_history(
                        message=prompt["content"],
                        role=prompt["role"],
                        timestamp=datetime.now(),
                    )
                else:
                    self._chat_node.get_logger().info(
                        f"Invalid prompt format : {prompt}"
                    )
                    return False

            return True

        return False

    def load_prompts(self, file: str) -> bool:
        """Load prompts from file"""
        try:
            with open(file, "r") as f:
                # Keep a copy of the prompts
                self._prompts = json.load(f)

            # Load to history
            return self.load_prompts_into_history(self._prompts)
        except Exception as e:
            self._chat_node.get_logger().info(f"Failed to load prompts: {e}")
            self._prompts = None
            return False

    def load_tools(self, file: str) -> bool:
        """Load tools from file"""
        try:
            with open(file, "r") as f:
                # TODO Validate schema
                self._tools_schema = json.load(f)

                return True
        except Exception as e:
            self._chat_node.get_logger().info(f"Failed to load tools: {e}")
            self._tools_schema = None
            return False

        return False

    def get_request_messages(self) -> list:
        """Get the messages to send to the server"""
        messages = list()
        for message in self.history:
            # Copy message
            new_message = message.copy()
            # Discard timestamp for now
            new_message.pop("datetime", None)
            messages.append(new_message)

        return messages

    @abstractmethod
    def send_request_and_process_response(self):
        """Send request to the server and get the response"""
        pass


class ChatGPTAPI(BaseChatAPI):
    def __init__(
        self,
        chat_node: rclpy.node.Node,
        language: str,
        language_model: str,
        streaming: bool,
        save_history: bool,
        save_history_path: str,
    ):
        super().__init__(
            chat_node,
            language,
            language_model,
            streaming,
            save_history,
            save_history_path,
        )
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def create_chat_completion(self):
        return openai.chat.completions.create(
            model=self.language_model,
            messages=self.get_request_messages(),
            max_tokens=1600,
            temperature=0.5,  # Somewhat creative
            frequency_penalty=0.5,  # Avoid repetition
            tools=self._tools_schema,
            tool_choice="auto",
            top_p=0.9,  # Avoid repetition
            stream=self._streaming,  # Enable streaming mode
        )

    def send_request_and_process_response(self):
        # TODO High cyclomatic complexity, should be split in a few functions
        try:
            response = self.create_chat_completion()
            assistant_message = ""
            final_tool_calls = {}
            # Generator will give partial responses
            if self._streaming:
                for chunk in response:
                    if len(chunk.choices) > 0:
                        # Get delta
                        delta = chunk.choices[0].delta
                        # Process tool calls
                        if delta.tool_calls is not None:
                            # Tool calls are sent in chuncks so we need to accumulate them
                            # And process them when we have the full message
                            for tool_call in delta.tool_calls:
                                index = tool_call.index
                                if index not in final_tool_calls:
                                    final_tool_calls[index] = tool_call
                                else:
                                    if (
                                        tool_call.function.arguments
                                        and len(tool_call.function.arguments) > 0
                                    ):
                                        final_tool_calls[
                                            index
                                        ].function.arguments += (
                                            tool_call.function.arguments
                                        )
                        # Process normal messages
                        if delta.content is not None:
                            content = delta.content
                            if len(content) == 0:
                                continue
                            # Add fragement to output message
                            assistant_message += content
                            # Send fragment to be processed
                            self._chat_node.add_pending_message(content)
                            # Add full output message to the history
                if len(assistant_message) > 0:
                    # Add to history
                    self.add_to_history(
                        message=assistant_message,
                        role="assistant",
                        timestamp=datetime.now(),
                    )
            else:
                if len(response.choices) > 0:
                    choice = response.choices[0]

                    if choice.message is not None:
                        assistant_message = choice.message.content
                    # Process tool calls
                    if choice.message.tool_calls is not None:
                        for tool_call in choice.message.tool_calls:
                            final_tool_calls[0] = tool_call
                    if assistant_message:
                        self._chat_node.add_pending_message(assistant_message)
                        # Add to history
                        self.add_to_history(
                            message=assistant_message,
                            role="assistant",
                            timestamp=datetime.now(),
                        )
            # Process final tools calls
            self._chat_node.get_logger().info(f"Reponse: {final_tool_calls.values()}")
            for tool_call in final_tool_calls.values():
                if tool_call.type == "function":
                    # Get Function information
                    id = tool_call.id
                    function_name = tool_call.function.name
                    function_arguments = tool_call.function.arguments
                    # Add to History
                    self.add_tool_calls_to_history(
                        [tool_call],
                        timestamp=datetime.now(),
                        function_name=function_name,
                    )

                    # Call service with
                    response = self._chat_node.call_tools_external_service(
                        id, function_name, function_arguments
                    )
                    if response is not None:
                        try:
                            raw = response.result if hasattr(response, "result") else response
                            parsed = json.loads(raw) if raw else {}
                        except json.JSONDecodeError as e:
                            parsed = {"error": f"Failed to decode JSON: {e}"}

                        self.add_tool_call_result_to_history(
                            tool_call=tool_call,
                            result=parsed,
                            timestamp=datetime.now(),
                        )
                    else:
                        self._chat_node.get_logger().error(
                            f"Failed to call external service: {function_name}"
                        )
                        self.add_tool_call_result_to_history(
                            tool_call=tool_call,
                            result={"error": "Failed to call external service"},
                            timestamp=datetime.now(),
                        )

                    # Process final message recursively
                    self.send_request_and_process_response()

            print("send_request_and_process_response done")
        except Exception as e:
            self._chat_node.get_logger().error(f"Error: {e}")
            self._chat_node.add_pending_message(str(e))

class OllamaAPI(ChatGPTAPI):
    def __init__(
        self,
        chat_node: rclpy.node.Node,
        language: str,
        language_model: str,
        streaming: bool,
        save_history: bool,
        save_history_path: str,
    ):
        super().__init__(
            chat_node,
            language,
            language_model,
            streaming,
            save_history,
            save_history_path,
        )
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        )

    def create_chat_completion(self):
        # TODO Validate if all parameters are supported in the ollama API.
        # Fine-tuned parameters for ollama could be required, this is why
        # create_chat_completion (identical to the ChatGPTAPI class for now)
        # is overloaded here.
        return self.client.chat.completions.create(
            model=self.language_model,
            messages=self.get_request_messages(),
            max_tokens=1600,
            temperature=0.5,  # Somewhat creative
            frequency_penalty=0.5,  # Avoid repetition
            tools=self._tools_schema,
            tool_choice="auto",
            top_p=0.9,  # Avoid repetition
            stream=self._streaming,  # Enable streaming mode
        )


class ChatNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("chat_node")

        self._talking = False
        self._processing = False
        self._pending_messages = list()
        self._executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
        # This will allow to receive callbacks while processing another one
        self._subscriber_callback_group = (
            rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        )
        # self._subscriber_callback_group_transcript = (
        #    rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        # )
        self._service_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        self.partial_message_transformations: List[Callable[[str], str]] = []
        self.partial_message_transformations.append(ChatNode._remove_think_tags)
        self.partial_message_transformations.append(
            ChatNode._replace_enumeration_characters
        )

        self.revive_counter = 0

        self._language = (
            self.declare_parameter("language", "fr").get_parameter_value().string_value
        )

        self._language_model = (
            self.declare_parameter("language_model", "gpt-4o-mini")
            .get_parameter_value()
            .string_value
        )
        self._model_type = (
            self.declare_parameter("model_type", "chatgpt")
            .get_parameter_value()
            .string_value
        )

        # Tools
        self._enable_tools = (
            self.declare_parameter("enable_tools", True)
            .get_parameter_value()
            .bool_value
        )
        self._tools_file = (
            self.declare_parameter(
                "tools_config",
                get_package_share_directory("chat") + "/tools/default_tools.json",
            )
            .get_parameter_value()
            .string_value
        )

        # Prompts
        self._enable_prompts = (
            self.declare_parameter("enable_prompts", True)
            .get_parameter_value()
            .bool_value
        )
        self._prompts_file = (
            self.declare_parameter(
                "prompts_config",
                get_package_share_directory("chat")
                + f"/prompts/default_{self._language}.json",
            )
            .get_parameter_value()
            .string_value
        )
        self._context = (
            self.declare_parameter(
                "context",
                "",
            )
            .get_parameter_value()
            .string_value
        )
        self._user_name = (
            self.declare_parameter(
                "user_name",
                "general",
            )
            .get_parameter_value()
            .string_value
        )
        self._streaming = (
            self.declare_parameter("streaming", False).get_parameter_value().bool_value
        )
        self._save_history = (
            self.declare_parameter("save_history", True)
            .get_parameter_value()
            .bool_value
        )
        self._save_history_path = os.path.expanduser(
            # Path is temporary, it will be changed to send to opentera
            self.declare_parameter(
                "save_history_path",
                f"~/.ros/chat_history/{self._user_name}_chat_history.json",
            )
            .get_parameter_value()
            .string_value
        )
        # Initialize API
        if self._model_type == "ollama":
            self._chat_api = OllamaAPI(
                self,
                language=self._language,
                language_model=self._language_model,
                streaming=self._streaming,
                save_history=self._save_history,
                save_history_path=self._save_history_path,
            )
        elif self._model_type == "chatgpt":
            self._chat_api = ChatGPTAPI(
                self,
                language=self._language,
                language_model=self._language_model,
                streaming=self._streaming,
                save_history=self._save_history,
                save_history_path=self._save_history_path,
            )
        else:
            raise ModelNotFoundError(
                f"Model not found : {self._model_type}. Available models are ollama and chatgpt."
            )

        # Load tools
        if self._enable_tools:
            if not self._chat_api.load_tools(self._tools_file):
                self.get_logger().error("Failed to load tools")
                raise ToolsNotFoundError(self._tools_file)
        # Load prompts
        if self._enable_prompts:
            if not self._chat_api.load_prompts(self._prompts_file):
                self.get_logger().error("Failed to load prompts")
                raise PromptsNotFoundError(self._prompts_file)
        else:
            self.get_logger().info("Prompts not enabled")
            self.get_logger().info("Using default prompts")
            self._chat_api.load_default_context()

        # Subscribers
        self._context_input_sub = hbba_lite.OnOffHbbaSubscriber(
            self,
            ContextInput,
            "chat/context_input",
            self._on_context_input_received_cb,
            qos_profile=QoSProfile(history=1, depth=1),
            state_service_name="chat/context_input/filter_state",
        )

        self._context_input_sub.on_filter_state_changed(
            self._on_context_input_filter_state_cb
        )

        self._talk_done_sub = self.create_subscription(
            Done,
            "talk/done",
            self._on_talk_done_cb,
            1,
            callback_group=self._subscriber_callback_group,
        )

        # Publishers
        self._talk_text_pub = self.create_publisher(Text, "talk/text", 1)
        self._chat_done_pub = self.create_publisher(Done, "chat/done", 1)

        self.add_on_set_parameters_callback(self.parameter_callback)

        # Print parameters summary
        self.get_logger().info(f"Language: {self._language}")
        self.get_logger().info(f"Language model: {self._language_model}")
        self.get_logger().info(f"Model type: {self._model_type}")
        self.get_logger().info(f"Enable tools: {self._enable_tools}")
        self.get_logger().info(f"Tools file: {self._tools_file}")
        self.get_logger().info(f"Enable prompts: {self._enable_prompts}")
        self.get_logger().info(f"Prompts file: {self._prompts_file}")
        self.get_logger().info(f"Streaming: {self._streaming}")
        self.get_logger().info(f"Save history: {self._save_history}")
        self.get_logger().info(f"Save history path: {self._save_history_path}")
        self.get_logger().info("Chat node initialized")

    def parameter_callback(self, params):
        for param in params:
            if param.name == "user_name":
                self._user_name = param.value
                self.get_logger().info(
                    f"Received an update to parameter user_name: {param.value}"
                )
            if param.name == "context":
                self.get_logger().info(
                    f"Received an update to parameter context: {param.value}"
                )
                self._chat_api.reset_history()
                self._chat_api.load_prompts(self._prompts_file)
                self.change_save_path()
                self._chat_api.add_to_history(
                    message=param.value,
                    role="system",
                    timestamp=datetime.now(),
                )

        return SetParametersResult(successful=True)

    def call_tools_external_service(
        self, id: str, function_name: str, function_arguments: str
    ) -> ChatToolsFunctionCall.Response:
        """Call the external service to process the tool function call"""
        self.get_logger().info(
            f"Calling external service: {function_name} with arguments: {function_arguments}"
        )

        # Create client
        # Service name is dynamic according to function_name
        client = self.create_client(
            ChatToolsFunctionCall,
            f"/chat/tools/functions/{function_name}",
            callback_group=self._service_callback_group,
        )

        # Create Request
        request = ChatToolsFunctionCall.Request()

        # Fill request
        request.id = id
        request.function_name = function_name
        request.function_arguments = function_arguments

        # Wait for service to be available
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"Service {client.srv_name} not available")
            return None

        # Call and wait for service
        future = client.call_async(request)

        # WARNING : This is a workaround to wait for the service to be done
        # It is not possible to use rclpy.spin_until_future_complete because
        # it will block the executor and the node will not be able to process
        # other callbacks even when using a ReentrantCallbackGroup and a MultiThreadedExecutor.
        # For this to work, the service must be called in a separate thread.
        event = Event()

        def service_done_cb(future):
            event.set()

        future.add_done_callback(service_done_cb)
        event.wait(timeout=10.0)

        if future.done():
            try:
                response_msg = future.result()
            except Exception as exc:
                self.get_logger().error(f"Service call raised: {exc}")
                return None

            self.get_logger().info(f"Service call result: {response_msg}")
            return response_msg
        else:
            self.get_logger().error("Service call timed-out or failed")
            return None


    def add_pending_message(self, message: str):
        self._pending_messages.append(message)
        self._process_pending_messages()

    def _on_context_input_filter_state_cb(
        self, previous_is_filtering_all_messages, new_is_filtering_all_messages
    ):
        self.get_logger().info(
            f"Transcript filter state changed: {new_is_filtering_all_messages} from {previous_is_filtering_all_messages}"
        )

    def _on_context_input_received_cb(self, msg: ContextInput):
        self._talking = False
        self._processing = False

        if len(msg.transcript.text) > 0:
            self.get_logger().info(f"Transcript received: {msg.transcript.text}")
            # Add the transcript to the context history
            self._chat_api.add_to_history(
                message=msg.transcript.text, role="user", timestamp=datetime.now()
            )
            # Process the request
            self._processing = True
            self.get_logger().info("Processing...")
            self._chat_api.send_request_and_process_response()
            self._processing = False
            self.get_logger().info("Processing done!")
            self.revive_counter = 0

        elif (
            len(msg.objects) > 0
            and len(msg.transcript.text) == 0
            and self.revive_counter < 2
            and msg.revive_conversation
        ):
            self.get_logger().info("Reviving with objects")
            self._chat_api.add_to_history(
                message=self._revive_conversation_msg(),
                role="system",
                timestamp=datetime.now(),
            )
            # Process the request
            self._processing = True
            self.get_logger().info("Processing...")
            self._chat_api.send_request_and_process_response()
            self._processing = False
            self.get_logger().info("Processing done!")
            self.revive_counter += 1

        else:
            self.get_logger().error("Empty transcript and not reviving conversation.")

        # Safety always call _process_pending_messages
        self._process_pending_messages()

    def _on_talk_done_cb(self, msg: Done):
        self.get_logger().info(f"Talk done : {msg.ok}")
        self._talking = False
        self._process_pending_messages()

    @staticmethod
    def _remove_think_tags(partial_message: str) -> str:
        # TODO better handling of <think></think> tags over multiple partial messages.
        return re.sub(r"<think>.*</think>", "", partial_message, flags=re.DOTALL)

    @staticmethod
    def _replace_enumeration_characters(partial_message: str) -> str:
        # Avoid "*" because TTS will say "Asterisk"
        # TODO find a better replacement
        return partial_message.replace("*", "-")

    def _revive_conversation_msg(self) -> str:
        if self._language == "fr":
            revive_msg = (
                "La conversation est arrêtée, essaie de relancer la conversation en utilisant "
                "les objets dans ton champ de vision et le contexte de la conversation."
            )
        else:
            revive_msg = (
                "The conversation has stopped. Try to restart it by using the objects in your field of view "
                "and the context of the conversation."
            )
        return revive_msg

    def _process_pending_messages(self):
        if not self._talking:
            partial_message: str = ""
            for message in self._pending_messages:
                partial_message += message
            self._pending_messages.clear()

            # To be done we need to have no pending messages
            if len(partial_message) == 0:
                # We are done
                chat_msg = Done()
                chat_msg.ok = True
                self.get_logger().info("Chat done")
                self._chat_done_pub.publish(chat_msg)
                return

            talk_msg = Text()

            # Apply transformations to partial messages (cleanup mostly)
            partial_message = reduce(
                lambda msg, f: f(msg),
                self.partial_message_transformations,
                partial_message,
            )

            if not self._processing:
                # Send everything, we are done!
                self._talking = True
                talk_msg.text = partial_message
                self.get_logger().info(f"Sending talk message: {talk_msg.text}")
                self._talk_text_pub.publish(talk_msg)
            else:
                # Send the first phrase and buffer the rest
                # Split the message into sentences based on punctuation marks .?!
                # Match text with optional punctuation
                sentences = re.findall(r"[^!.?]+[!.?]?", partial_message)

                # We are talking if we found at least one sentence
                if len(sentences) > 1:
                    self._talking = True
                    talk_msg.text = sentences[0]
                    self.get_logger().info(f"Sending talk message: {talk_msg.text}")
                    self._talk_text_pub.publish(talk_msg)
                    # Remove the first sentence from the list
                    sentences.pop(0)

                # Re-Add the remaining sentences to the pending messages
                for sentence in sentences:
                    self._pending_messages.append(sentence)

    def change_save_path(self):
        self._save_history_path = os.path.expanduser(
            f"~/.ros/chat_history/{self._user_name}_chat_history.json"
        )
        self._chat_api._save_history_path = os.path.expanduser(
            f"~/.ros/chat_history/{self._user_name}_chat_history.json"
        )
        self.get_logger().info(
            f"Save history path changed to: {self._chat_api._save_history_path}"
        )

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
        chat_node.get_logger().error(f"{e}")
    finally:
        chat_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    print(openai.__version__)
    main()
