#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import rclpy
import rclpy.node
import rclpy.executors

from std_msgs.msg import Float32
from behavior_msgs.msg import Text, Done, Statistics
from audio_utils_msgs.msg import AudioFrame

import hbba_lite
import time_utils


class ChatNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('chat_node')

    def run(self):
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
        executor.add_node(self)
        executor.spin()


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
