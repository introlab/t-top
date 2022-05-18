#!/usr/bin/env python3

import math

import rospy
from hbba_lite.srv import SetOnOffFilterState

from t_top import MovementCommands


PAUSE_DURATION_S = 5.0


class EgoNoiseReductionTestNode:
    def __init__(self):
        self._movement_commands = MovementCommands()

    def run(self):
        self._enable_on_off_filter('ego_noise_reduction/filter_state')
        self._enable_on_off_filter('pose/filter_state')

        while not rospy.is_shutdown():
            rospy.sleep(PAUSE_DURATION_S)
            self._movement_commands.move_torso(math.pi / 2, should_wait=True)
            self._movement_commands.move_torso(-math.pi / 2, should_wait=True)

            self._movement_commands.move_yes(speed_rad_sec=1.0)
            self._movement_commands.move_no(speed_rad_sec=0.5)
            self._movement_commands.move_maybe(speed_rad_sec=1.5)

    def _enable_on_off_filter(self, name):
        rospy.wait_for_service(name)
        filter_state = rospy.ServiceProxy(name, SetOnOffFilterState)
        filter_state(is_filtering_all_messages=False)


def main():
    rospy.init_node('ego_noise_reduction_test_node')
    capture_voice_node = EgoNoiseReductionTestNode()
    capture_voice_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
