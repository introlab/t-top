#!/usr/bin/env python3

import rclpy

from std_msgs.msg import Float32
from daemon_ros_client.msg import MotorStatus

from dance.lib_pose_dance_node import PoseDanceNode


class TorsoDanceNode(PoseDanceNode):
    def __init__(self):
        super().__init__('torso_dance_node')

        self._current_torso_orientation = 0.0

        self._torso_offset = 0.0
        self._torso_orientation_pub = self.create_publisher(Float32, 'dance/set_torso_orientation', 5)

        self._motor_status_sub = self.create_subscription(MotorStatus, 'daemon/motor_status', self._motor_status_cb, 1)

    def _hbba_filter_state_cb(self, previous_is_filtering_all_messages, new_is_filtering_all_messages):
        if previous_is_filtering_all_messages and not new_is_filtering_all_messages:
            self._torso_offset = self._current_torso_orientation

    def _motor_status_cb(self, msg):
        self._current_torso_orientation = msg.torso_orientation

    def _send_pose(self, pose):
        if len(pose) == 1:
            pose_msg = Float32()
            pose_msg.data = float(pose[0] + self._torso_offset)

            self._torso_orientation_pub.publish(pose_msg)
        else:
            self.get_logger().error('Invalid pose')


def main():
    rclpy.init()
    torso_dance_node = TorsoDanceNode()

    try:
        torso_dance_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        torso_dance_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
