#!/usr/bin/env python3

import threading

import rospy
from std_msgs.msg import Float32
from daemon_ros_client.msg import MotorStatus

from dance.lib_pose_dance_node import PoseDanceNode


P_CHANGE_MOVEMENT = 0.25


class TorsoDanceNode(PoseDanceNode):
    def __init__(self):
        self._current_torso_orientation_lock = threading.Lock()
        self._current_torso_orientation = 0.0

        self._torso_offset = 0.0
        self._torso_orientation_pub = rospy.Publisher('dance/set_torso_orientation', Float32, queue_size=5)

        super(TorsoDanceNode, self).__init__()

        self._motor_status_sub = rospy.Subscriber('daemon/motor_status', MotorStatus, self._motor_status_cb, queue_size=1)

    def _hbba_filter_state_cb(self, previous_is_filtering_all_messages, new_is_filtering_all_messages):
        if previous_is_filtering_all_messages and not new_is_filtering_all_messages:
            with self._lock, self._current_torso_orientation_lock:
                self._torso_offset = self._current_torso_orientation

    def _motor_status_cb(self, msg):
        with self._current_torso_orientation_lock:
            self._current_torso_orientation = msg.torso_orientation

    def _send_pose(self, pose):
        """ Called with self._lock locked """
        if len(pose) == 1:
            pose_msg = Float32()
            pose_msg.data = pose[0] + self._torso_offset

            self._torso_orientation_pub.publish(pose_msg)
        else:
            rospy.logerr('Invalid pose')


def main():
    rospy.init_node('torso_dance_node')
    torso_dance_node = TorsoDanceNode()
    torso_dance_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
