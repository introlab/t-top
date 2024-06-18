#!/usr/bin/env python3

import rclpy
import rclpy.node

from std_msgs.msg import Empty

from t_top import MovementCommands, HEAD_ZERO_Z


INACTIVE_SLEEP_DURATION = 0.1


class ExploreNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('explore_node')

        self._simulation = self.declare_parameter('simulation', False).get_parameter_value().bool_value
        self._explore_frequency = self.declare_parameter('explore_frequency', 0.00833333333).get_parameter_value().double_value
        self._torso_speed = self.declare_parameter('torso_speed_rad_sec', 0.5).get_parameter_value().double_value
        self._head_speed = self.declare_parameter('head_speed_rad_sec', 0.5).get_parameter_value().double_value

        self._timer = self.create_timer(1 / self._explore_frequency, self._timer_callback)

        self._movement_commands = MovementCommands(self, self._simulation, namespace='explore')
        self._done_pub = self.create_publisher(Empty, 'explore/done', 5)

    def _timer_callback(self):
        if self._movement_commands.is_filtering_all_messages:
            return

        self._movement_commands.move_torso(0, should_wait=True, speed_rad_sec=self._torso_speed)
        self._movement_commands.move_head([0, 0, HEAD_ZERO_Z, 0, -0.3, 0], should_wait=True, speed_rad_sec=self._head_speed)
        self._movement_commands.move_torso(1.57, should_wait=False, speed_rad_sec=self._torso_speed)
        self._movement_commands.move_torso(3.14, should_wait=True, speed_rad_sec=self._torso_speed)
        self._movement_commands.move_head([0, 0, HEAD_ZERO_Z, 0, 0.15, 0], should_wait=True, speed_rad_sec=self._head_speed)
        self._movement_commands.move_torso(1.57, should_wait=False, speed_rad_sec=self._torso_speed)
        self._movement_commands.move_torso(0, should_wait=False, speed_rad_sec=self._torso_speed)
        self._movement_commands.move_torso(-1.57, should_wait=False, speed_rad_sec=self._torso_speed)
        self._movement_commands.move_torso(-3.14, should_wait=True, speed_rad_sec=self._torso_speed)
        self._movement_commands.move_head([0, 0, HEAD_ZERO_Z, 0, -0.3, 0], should_wait=True, speed_rad_sec=self._head_speed)
        self._movement_commands.move_torso(-1.57, should_wait=False, speed_rad_sec=self._torso_speed)
        self._movement_commands.move_torso(0, should_wait=True, speed_rad_sec=self._torso_speed)
        self._movement_commands.move_head([0, 0, HEAD_ZERO_Z, 0, 0, 0], should_wait=True, speed_rad_sec=self._head_speed)

        self._done_pub.publish(Empty())

    def run(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    explore_node = ExploreNode()

    try:
        explore_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        explore_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
