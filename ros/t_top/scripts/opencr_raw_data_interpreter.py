#!/usr/bin/env python3

import rospy
import tf

from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class OpenCrRawDataInterpreter:
    def __init__(self):
        self._rate = rospy.Rate(15)

        self._base_footprint_torso_base_delta_z = rospy.get_param('~base_link_torso_base_delta_z', 0.0)

        self._tf_broadcaster = tf.TransformBroadcaster()

        self._tf_listener = tf.TransformListener()

        self._imu_pub = rospy.Publisher('opencr_imu/data_raw', Imu, queue_size=5)
        self._mag_pub = rospy.Publisher('opencr_imu/mag', MagneticField, queue_size=5)

        self._odom_pub = rospy.Publisher('opencr/odom', Odometry, queue_size=5)

        self._raw_imu_sub = rospy.Subscriber('opencr/raw_imu',
            Float32MultiArray, self._raw_imu_cb, queue_size=5)
        self._current_torso_orientation_sub = rospy.Subscriber('opencr/current_torso_orientation',
            Float32, self._current_torso_orientation_cb, queue_size=5)
        self._current_head_pose_sub = rospy.Subscriber('opencr/current_head_pose',
            PoseStamped, self._current_head_pose_cb, queue_size=5)

    def _raw_imu_cb(self, raw_imu):
        now = rospy.Time.now()

        imu_msg = Imu()
        imu_msg.header.stamp = now
        imu_msg.header.frame_id = 'opencr_imu'
        imu_msg.linear_acceleration.x = raw_imu.data[0]
        imu_msg.linear_acceleration.y = raw_imu.data[1]
        imu_msg.linear_acceleration.z = raw_imu.data[2]
        imu_msg.angular_velocity.x = raw_imu.data[3]
        imu_msg.angular_velocity.y = raw_imu.data[4]
        imu_msg.angular_velocity.z = raw_imu.data[5]
        self._imu_pub.publish(imu_msg)

        magnetic_field = MagneticField()
        magnetic_field.header.stamp = now
        magnetic_field.header.frame_id = 'opencr_imu'
        magnetic_field.magnetic_field.x = raw_imu.data[6]
        magnetic_field.magnetic_field.y = raw_imu.data[7]
        magnetic_field.magnetic_field.z = raw_imu.data[8]
        self._imu_pub.publish(imu_msg)

    def _current_torso_orientation_cb(self, orientation):
        now = rospy.Time.now()
        translation = (0, 0, self._base_footprint_torso_base_delta_z)
        quaternion = tf.transformations.quaternion_from_euler(0, 0, orientation.data)
        self._tf_broadcaster.sendTransform(translation, quaternion, now, 'torso_base', 'base_link')

    def _current_head_pose_cb(self, pose):
        now = rospy.Time.now()
        translation = (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
        quaternion = (pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w)
        self._tf_broadcaster.sendTransform(translation, quaternion, now, 'head', pose.header.frame_id)

    def _send_odom(self, trans, ori):
        odom_msg = Odometry()
        odom_msg.header.frame_id = 'base_link'
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.child_frame_id = 'camera_depth_optical_frame'

        odom_msg.pose.pose.position.x = trans[0]
        odom_msg.pose.pose.position.y = trans[1]
        odom_msg.pose.pose.position.z = trans[2]

        odom_msg.pose.pose.orientation.x = ori[0]
        odom_msg.pose.pose.orientation.y = ori[1]
        odom_msg.pose.pose.orientation.z = ori[2]
        odom_msg.pose.pose.orientation.w = ori[3]

        # From stewart_simulator
        odom_msg.pose.covariance = [1.29593696e-07,  -3.02490686e-09, -2.34185217e-09,  3.43909522e-08, -1.76583062e-07,  5.85076793e-09,
                                    -3.02490686e-09,  1.25348464e-07,  4.17541942e-09,  1.95763784e-07, -1.30688450e-08,  3.89092298e-08,
                                    -2.34185217e-09,  4.17541942e-09,  8.27253806e-08,  4.62811502e-10,  3.30862971e-09, -1.41326028e-08,
                                    3.43909522e-08,   1.95763784e-07,  4.62811502e-10,  1.14239749e-05,  3.26955028e-08,  2.41604671e-07,
                                    -1.76583062e-07, -1.30688450e-08,  3.30862971e-09,  3.26955028e-08,  1.07156371e-05, -2.98094285e-07,
                                    5.85076793e-09,   3.89092298e-08, -1.41326028e-08,  2.41604671e-07, -2.98094285e-07,  1.20712896e-05]

        self._odom_pub.publish(odom_msg)

    def run(self):
        while not rospy.is_shutdown():
            self._rate.sleep()

            try:
                (trans, ori) = self._tf_listener.lookupTransform('/camera_depth_optical_frame', '/base_link', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

            self._send_odom(trans, ori)


def main():
    rospy.init_node('opencr_raw_data_interpreter')
    opencr_raw_data_interpreter = OpenCrRawDataInterpreter()
    opencr_raw_data_interpreter.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
