#!/usr/bin/env python3
import rospy

from std_msgs.msg import Int8

import hbba_lite


def main():
    rospy.init_node('test_on_off_hbba_publisher')
    pub = hbba_lite.OnOffHbbaPublisher('int_topic', Int8, queue_size=10)

    rate = rospy.Rate(1) # 1hz
    i = 0
    while not rospy.is_shutdown():
        pub.publish(Int8(i))
        i += 1
        rate.sleep()


if __name__ == '__main__':
    main()
