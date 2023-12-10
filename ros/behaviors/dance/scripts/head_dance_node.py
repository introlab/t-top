#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped

from t_top import HEAD_ZERO_Z
from dance.lib_pose_dance_node import PoseDanceNode


class HeadDanceNode(PoseDanceNode):
    def __init__(self):
        self._head_pose_pub = rospy.Publisher('dance/set_head_pose', PoseStamped, queue_size=5)
        super(HeadDanceNode, self).__init__()

    def _send_pose(self, pose):
        """ Called with self._lock locked """
        if len(pose) == 7:
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = 'stewart_base'

            pose_msg.pose.position.x = pose[0]
            pose_msg.pose.position.y = pose[1]
            pose_msg.pose.position.z = HEAD_ZERO_Z + pose[2]

            pose_msg.pose.orientation.x = pose[3]
            pose_msg.pose.orientation.y = pose[4]
            pose_msg.pose.orientation.z = pose[5]
            pose_msg.pose.orientation.w = pose[6]

            self._head_pose_pub.publish(pose_msg)
        else:
            rospy.logerr('Invalid pose')


def main():
    rospy.init_node('head_dance_node')
    head_dance_node = HeadDanceNode()
    head_dance_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
