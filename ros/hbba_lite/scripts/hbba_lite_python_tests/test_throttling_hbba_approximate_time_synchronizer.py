#!/usr/bin/env python3
import rospy
import message_filters

from hbba_lite.msg import Int32Stamped

import hbba_lite


def callback(data1, data2):
    rospy.loginfo('Data received : {} {}'.format(data1.data, data2.data))


def main():
    rospy.init_node('test_throttling_hbba_approximate_time_synchronizer')
    sub1 = message_filters.Subscriber('int_topic_1', Int32Stamped)
    sub2 = message_filters.Subscriber('int_topic_2', Int32Stamped)
    _ = hbba_lite.ThrottlingHbbaApproximateTimeSynchronizer([sub1, sub2], 10, 0.1, callback, 'int_topics/filter_state')
    rospy.spin()


if __name__ == '__main__':
    main()
