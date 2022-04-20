#!/usr/bin/env python3

import rospy
import tf

from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from geometry_msgs.msg import PoseStamped


class OpenCrRawDataInterpreter:
    def __init__(self):
        self._base_footprint_torso_base_delta_z = rospy.get_param('~base_link_torso_base_delta_z', 0.0)

        self._tf_broadcaster = tf.TransformBroadcaster()

        self._imu_pub = rospy.Publisher('opencr_imu/data_raw', Imu, queue_size=5)
        self._mag_pub = rospy.Publisher('opencr_imu/mag', MagneticField, queue_size=5)

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

        mag_msg = MagneticField()
        mag_msg.header.stamp = now
        mag_msg.header.frame_id = 'opencr_imu'
        mag_msg.magnetic_field.x = raw_imu.data[6]
        mag_msg.magnetic_field.y = raw_imu.data[7]
        mag_msg.magnetic_field.z = raw_imu.data[8]
        self._mag_pub.publish(mag_msg)

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

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('opencr_raw_data_interpreter')
    opencr_raw_data_interpreter = OpenCrRawDataInterpreter()
    opencr_raw_data_interpreter.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
