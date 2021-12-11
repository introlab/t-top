#!/usr/bin/env python3
import rospy

from std_msgs.msg import Int8

import hbba_lite


def callback(data):
    rospy.loginfo('Data received : {}'.format(data.data))


def main():
    rospy.init_node('test_throttling_hbba_subscriber')
    sub = hbba_lite.ThrottlingHbbaSubscriber('int_topic', Int8, callback)
    rospy.spin()
    sub.unregister()


if __name__ == '__main__':
    main()
