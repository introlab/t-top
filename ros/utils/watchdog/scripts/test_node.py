#!/usr/bin/env python3

import rospy

from std_msgs.msg import Empty


class TestNode:
    def __init__(self):
        self._startup_delay_s = rospy.get_param('~startup_delay_s', 1.5)
        self._freeze_delay_s = rospy.get_param('~freeze_delay_s', 10.0)
        self._message_frequency = rospy.get_param('~message_frequency', 10.0)

        self._pub = rospy.Publisher('topic', Empty, queue_size=1)
        self._rate = rospy.Rate(self._message_frequency)

    def run(self):
        startup_time_s = rospy.get_time()

        while not rospy.is_shutdown():
            startup_duration = rospy.get_time() - startup_time_s
            if startup_duration > self._startup_delay_s and startup_duration < self._freeze_delay_s:
                self._pub.publish(Empty())

            self._rate.sleep()


def main():
    rospy.init_node('test_node')
    test_node = TestNode()
    test_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
