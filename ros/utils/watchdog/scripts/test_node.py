#!/usr/bin/env python3

import rclpy
import rclpy.node

from std_msgs.msg import Empty


class TestNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('test_node')

        self._startup_delay_ns = self.declare_parameter('startup_delay_s', 1.5).get_parameter_value().double_value * 1e9
        self._freeze_delay_ns = self.declare_parameter('freeze_delay_s', 10.0).get_parameter_value().double_value * 1e9
        self._message_frequency = self.declare_parameter('message_frequency', 10.0).get_parameter_value().double_value

        self._pub = self.create_publisher(Empty, 'topic', 1)

        self._timer = self.create_timer(1 / self._message_frequency, self._timer_callback)

        self._startup_time = self.get_clock().now()

    def _timer_callback(self):
        startup_duration = self.get_clock().now() - self._startup_time
        if startup_duration.nanoseconds > self._startup_delay_ns and startup_duration.nanoseconds < self._freeze_delay_ns:
            self._pub.publish(Empty())

    def run(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    test_node = TestNode()

    try:
        test_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        test_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
