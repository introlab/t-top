#!/usr/bin/env python3

import rospy
from std_msgs.msg import Empty

from t_top import MovementCommands, HEAD_ZERO_Z


INACTIVE_SLEEP_DURATION = 0.1


class ExploreNode:
    def __init__(self):
        self._simulation = rospy.get_param('~simulation')
        self._rate = rospy.Rate(rospy.get_param('~explore_frequency'))
        self._torso_speed = rospy.get_param('~torso_speed_rad_sec')
        self._head_speed = rospy.get_param('~head_speed_rad_sec')

        self._movement_commands = MovementCommands(self._simulation, namespace='explore')
        self._done_pub = rospy.Publisher('explore/done', Empty, queue_size=5)

    def run(self):
        while not rospy.is_shutdown():
            if self._movement_commands.is_filtering_all_messages:
                rospy.sleep(INACTIVE_SLEEP_DURATION)
                continue

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

            self._rate.sleep()


def main():
    rospy.init_node('explore_node')
    explore_node = ExploreNode()
    explore_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
