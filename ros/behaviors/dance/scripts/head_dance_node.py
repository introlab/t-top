#!/usr/bin/env python3
import rclpy

from geometry_msgs.msg import PoseStamped

from t_top import HEAD_ZERO_Z
from dance.lib_pose_dance_node import PoseDanceNode


class HeadDanceNode(PoseDanceNode):
    def __init__(self):
        super().__init__('head_dance_node')
        self._head_pose_pub = self.create_publisher(PoseStamped, 'dance/set_head_pose', 5)

    def _send_pose(self, pose):
        """ Called with self._lock locked """
        if len(pose) == 7:
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = 'stewart_base'

            pose_msg.pose.position.x = float(pose[0])
            pose_msg.pose.position.y = float(pose[1])
            pose_msg.pose.position.z = float(HEAD_ZERO_Z + pose[2])

            pose_msg.pose.orientation.x = float(pose[3])
            pose_msg.pose.orientation.y = float(pose[4])
            pose_msg.pose.orientation.z = float(pose[5])
            pose_msg.pose.orientation.w = float(pose[6])

            self._head_pose_pub.publish(pose_msg)
        else:
            self.get_logger().error('Invalid pose')


def main():
    rclpy.init()
    head_dance_node = HeadDanceNode()

    try:
        head_dance_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        head_dance_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
