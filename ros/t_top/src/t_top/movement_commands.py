#!/usr/bin/env python3

import rospy
import tf
import numpy as np
import math

from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped

import hbba_lite


HEAD_ZERO_Z = 0.16893650962222248


def vector_to_angles(vector):
    vector = [vector.x, vector.y, vector.z]
    unit_vector = vector / np.linalg.norm(vector)
    x = unit_vector[0]
    y = unit_vector[1]
    z = unit_vector[2]
    yaw = np.arctan2(y,x)
    pitch = -np.arctan2(z,x)
    return yaw, pitch


class MovementCommands:
    def __init__(self, simulation=False):
        self._maxFreq = 30
        self._minTime = 1 / self._maxFreq
        self._read_torso = 0
        self._read_head = 6 * [0]
        self._min_speed_rad_sec = 0.001
        self._min_speed_meters_sec = 0.0001
        self._np_head_tolerances = np.array(3 * [0.02] + 3 * [0.1])

        if simulation:
            self._np_head_tolerances = np.array(3 * [0.07] + 3 * [0.1])

        self._hbba_filter_state = hbba_lite.OnOffHbbaFilterState('pose/filter_state')
        self._torso_orientation_pub = rospy.Publisher('opencr/torso_orientation', Float32, queue_size=5)
        self._head_pose_pub = rospy.Publisher('opencr/head_pose', PoseStamped, queue_size=5)

        self._torso_orientation_sub = rospy.Subscriber('opencr/current_torso_orientation', Float32, self._read_torso_cb, queue_size=5)
        self._head_pose_sub = rospy.Subscriber('opencr/current_head_pose', PoseStamped, self._read_head_cb, queue_size=5)

    def _read_torso_cb(self, msg):
        self._read_torso = math.fmod(msg.data, 2 * math.pi)

    def _read_head_cb(self, msg):
        self._read_head[0] = msg.pose.position.x
        self._read_head[1] = msg.pose.position.y
        self._read_head[2] = msg.pose.position.z

        angles = tf.transformations.euler_from_quaternion([msg.pose.orientation.x,
                                                           msg.pose.orientation.y,
                                                           msg.pose.orientation.z,
                                                           msg.pose.orientation.w])
        self._read_head[3] = angles[0]
        self._read_head[4] = angles[1]
        self._read_head[5] = angles[2]

    def move_torso(self, pose, should_wait=False, speed=1.0e10):
        if self._hbba_filter_state.is_filtering_all_messages:
            return False

        if speed < self._min_speed_rad_sec:
            speed = self._min_speed_rad_sec

        steps_size = speed * self._minTime

        pose = math.fmod(pose, 2 * math.pi)
        distance = pose - self._read_torso
        if distance < -math.pi:
            distance = 2 * math.pi + distance
        elif distance > math.pi:
            distance = -(2 * math.pi - distance)

        if distance < 0:
            steps_size = -steps_size

        if abs(distance) > abs(steps_size):
            steps_number = int(abs(distance / steps_size))

            for i in range(1, steps_number):
                if self._hbba_filter_state.is_filtering_all_messages:
                    return False
                steps_cumulative_size = i * steps_size
                offset = distance - steps_cumulative_size
                self._torso_orientation_pub.publish(pose - offset)
                rospy.sleep(self._minTime)
        else:
            if self._hbba_filter_state.is_filtering_all_messages:
                return False
            self._torso_orientation_pub.publish(pose)

        if should_wait:
            while abs(pose - self._read_torso) > 0.1:
                if self._hbba_filter_state.is_filtering_all_messages:
                    return False
                self._torso_orientation_pub.publish(pose)
                rospy.sleep(self._minTime)

        return True

    def _get_head_steps_size_array(self, speed):
        steps_size = speed * self._minTime
        return np.array(3 * [steps_size])

    def _get_steps_size_dir(self, np_distances, np_steps_size):
        for i in range(0, np_distances.size):
            if np_distances[i] < 0:
                np_steps_size[i] = -np_steps_size[i]

        return np_steps_size

    def _compute_head_steps_size_array(self, speed_meters_sec, speed_rad_sec, np_distances):
        np_steps_size_trans = self._get_head_steps_size_array(speed_meters_sec)
        np_steps_size_rot = self._get_head_steps_size_array(speed_rad_sec)
        np_steps_size = np.append(np_steps_size_trans, np_steps_size_rot)

        return self._get_steps_size_dir(np_distances, np_steps_size)

    def _compute_head_steps_number_array(self, np_distances, np_steps_size):
        np_steps_number = np.zeros(6)
        for i in range(0, 6):
            if np.abs(np_distances[i]) > np.abs(np_steps_size[i]):
                np_steps_number[i] = int(abs(np_distances[i] / np_steps_size[i]))
            else:
                np_steps_number[i] = 1

        return np_steps_number

    def _head_msg(self, pose):
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = 'stewart_base'

        pose_msg.pose.position.x = pose[0]
        pose_msg.pose.position.y = pose[1]
        pose_msg.pose.position.z = pose[2]

        q = tf.transformations.quaternion_from_euler(pose[3], pose[4], pose[5])

        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]

        self._head_pose_pub.publish(pose_msg)

    def move_head(self, pose, should_wait=False, speed_meters_sec=1.0e10, speed_rad_sec=1.0e10):
        if self._hbba_filter_state.is_filtering_all_messages:
                return False

        if speed_meters_sec < self._min_speed_meters_sec:
            speed_meters_sec = self._min_speed_meters_sec

        if speed_rad_sec < self._min_speed_rad_sec:
            speed_rad_sec = self._min_speed_rad_sec

        np_pose = np.array(pose)

        np_distances = np_pose - np.array(self._read_head)

        np_steps_size = self._compute_head_steps_size_array(speed_meters_sec, speed_rad_sec, np_distances)

        np_steps_number = self._compute_head_steps_number_array(np_distances, np_steps_size)

        if int(np.amax(np_steps_number)) > 1:
            for i in range(1, int(np.amax(np_steps_number))):
                if self._hbba_filter_state.is_filtering_all_messages:
                    return False

                np_steps_cumulative_size = i * np_steps_size
                np_offsets = np_distances - np_steps_cumulative_size

                for j in range(np_steps_number.size):
                    if np_steps_number[j] <= i:
                        np_offsets[j] = 0

                self._head_msg(np_pose - np_offsets)
                rospy.sleep(self._minTime)
        else:
            if self._hbba_filter_state.is_filtering_all_messages:
                return False
            self._head_msg(pose)

        if should_wait:
            while not (np.abs(np_pose - np.array(self._read_head)) < self._np_head_tolerances).all():
                if self._hbba_filter_state.is_filtering_all_messages:
                    return False
                self._head_msg(pose)
                rospy.sleep(self._minTime)

        return True

    def move_yes(self):
        for i in range(0, 5):
            decay = 0.03 * i
            if not self.move_head([0, 0, HEAD_ZERO_Z, 0, -0.3 + decay, 0], True, speed_rad_sec=1.0):
                return False
            if not self.move_head([0, 0, HEAD_ZERO_Z, 0, 0.15 - decay, 0], True, speed_rad_sec=1.0):
                return False

        if not self.move_head([0, 0, HEAD_ZERO_Z, 0, 0, 0], True, speed_rad_sec=1.0):
            return False
        return True

    def move_no(self):
        for i in range(0, 3):
            if not self.move_head([0, 0, HEAD_ZERO_Z, 0, 0, 0.25], True, speed_rad_sec=1.0):
                return False
            if not self.move_head([0, 0, HEAD_ZERO_Z, 0, 0, -0.25], True, speed_rad_sec=1.0):
                return False

        if not self.move_head([0, 0, HEAD_ZERO_Z, 0, 0, 0], True, speed_rad_sec=1.0):
            return False
        return True

    def move_maybe(self):
        if not self.move_head([-0.01, 0, HEAD_ZERO_Z, 0.1, -0.2, 0], True, speed_rad_sec=0.5):
            return False
        rospy.sleep(1.2)
        if not self.move_head([-0.02, 0, HEAD_ZERO_Z, -0.2, -0.3, 0], True, speed_rad_sec=0.5):
            return False
        rospy.sleep(2.0)
        if not self.move_head([0, 0, HEAD_ZERO_Z, 0, 0, 0], True, speed_rad_sec=0.5):
            return False
        return True
