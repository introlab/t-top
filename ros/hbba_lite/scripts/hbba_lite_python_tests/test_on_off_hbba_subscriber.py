#!/usr/bin/env python3
import rospy

from std_msgs.msg import Int8

import hbba_lite


def callback(data):
    rospy.loginfo('Data received : {}'.format(data.data))


def main():
    rospy.init_node('test_on_off_hbba_subscriber')
    sub = hbba_lite.OnOffHbbaSubscriber('int_topic', Int8, callback)
    rospy.spin()
    sub.unregister()


if __name__ == '__main__':
    main()
