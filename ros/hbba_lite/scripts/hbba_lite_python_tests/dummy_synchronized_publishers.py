#!/usr/bin/env python3
import rospy

from std_msgs.msg import Header
from hbba_lite.msg import Int32Stamped

import hbba_lite


def main():
    rospy.init_node('test_on_off_hbba_publisher')
    pub1 = rospy.Publisher('int_topic_1', Int32Stamped, queue_size=10)
    pub2 = rospy.Publisher('int_topic_2', Int32Stamped, queue_size=10)

    rate = rospy.Rate(1) # 1hz
    i = 0
    while not rospy.is_shutdown():
        header = Header()
        header.seq = i
        header.stamp = rospy.Time.now()

        msg1 = Int32Stamped()
        msg1.header = header
        msg1.data = 2 * i

        msg2 = Int32Stamped()
        msg2.header = header
        msg2.data = 4 * i

        pub1.publish(msg1)
        pub2.publish(msg2)

        i += 1
        rate.sleep()


if __name__ == '__main__':
    main()
