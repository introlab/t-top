#!/usr/bin/env python3

import threading
import math
import time

import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np

from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped
from daemon_ros_client.msg import MotorStatus

import hbba_lite


HEAD_ZERO_Z = 0.17029453895440808

HEAD_POSE_X_INDEX = 0
HEAD_POSE_Y_INDEX = 1
HEAD_POSE_Z_INDEX = 2
HEAD_POSE_ROLL_INDEX = 3
HEAD_POSE_PITCH_INDEX = 4
HEAD_POSE_YAW_INDEX = 5


def vector_to_angles(vector):
    vector = [vector.x, vector.y, vector.z]
    unit_vector = vector / np.linalg.norm(vector)
    x = unit_vector[0]
    y = unit_vector[1]
    z = unit_vector[2]

    yaw = np.arctan2(y, x)
    pitch = -np.arctan2(z, x)
    return yaw, pitch


def fmod_radian(v):
    return math.fmod(math.fmod(v, 2 * math.pi) + 2 * math.pi, 2 * math.pi)


def abs_diff_torso_angle(a, b):
    d = abs(a - b)
    if d > math.pi:
        d = 2 * math.pi - d
    return d


class MovementCommands:
    def __init__(self, simulation=False, namespace='daemon'):
        self._read_torso_lock = threading.Lock()
        self._read_head_lock = threading.Lock()

        self._maxFreq = 30
        self._minTime = 1 / self._maxFreq
        self._read_torso = 0
        self._read_head = 6 * [0]
        self._min_speed_rad_sec = 0.001
        self._min_speed_meters_sec = 0.0001
        self._np_head_tolerances = np.array(3 * [0.004] + 3 * [0.1])

        if simulation:
            self._np_head_tolerances = np.array(3 * [0.07] + 3 * [0.1])

        self._hbba_filter_state = hbba_lite.OnOffHbbaFilterState(
            'pose/filter_state')
        self._torso_orientation_pub = rospy.Publisher(
            f'{namespace}/set_torso_orientation', Float32, queue_size=5)
        self._head_pose_pub = rospy.Publisher(
            f'{namespace}/set_head_pose', PoseStamped, queue_size=5)

        self._motor_status_sub = rospy.Subscriber('daemon/motor_status', MotorStatus, self._motor_status_cb, queue_size=1)

    @property
    def is_filtering_all_messages(self):
        return self._hbba_filter_state._is_filtering_all_messages

    @property
    def current_torso_pose(self):
        with self._read_torso_lock:
            return self._read_torso

    @property
    def current_head_pose(self):
        with self._read_head_lock:
            return self._read_head

    @property
    def frequency(self):
        return self._maxFreq

    @property
    def sleep_time(self):
        return self._minTime

    def _motor_status_cb(self, msg):
        with self._read_torso_lock:
            self._read_torso = fmod_radian(msg.torso_orientation)

        head_angles = euler_from_quaternion([msg.head_pose.orientation.x,
                                             msg.head_pose.orientation.y,
                                             msg.head_pose.orientation.z,
                                             msg.head_pose.orientation.w])

        with self._read_head_lock:
            self._read_head[0] = msg.head_pose.position.x
            self._read_head[1] = msg.head_pose.position.y
            self._read_head[2] = msg.head_pose.position.z
            self._read_head[3] = head_angles[0]
            self._read_head[4] = head_angles[1]
            self._read_head[5] = head_angles[2]

    # Should be called in a loop
    def move_torso_speed(self, speed_rad_sec_torso, should_sleep=True):
        if self._hbba_filter_state.is_filtering_all_messages:
            return False

        if abs(speed_rad_sec_torso) < self._min_speed_rad_sec:
            speed_rad_sec_torso = np.sign(
                speed_rad_sec_torso) * self._min_speed_rad_sec

        steps_size_torso = speed_rad_sec_torso * self._minTime
        torso_pose = self.current_torso_pose + steps_size_torso

        if abs(steps_size_torso) > 0:
            self._torso_orientation_pub.publish(torso_pose)

            if should_sleep:
                rospy.sleep(self._minTime)

    # Should be called in a loop
    def move_head_speed(self, speeds_head, should_sleep=True, current_head_pose=None):
        if self._hbba_filter_state.is_filtering_all_messages:
            return False

        for i, speed in enumerate(speeds_head[0:3]):
            if abs(speed) < self._min_speed_meters_sec:
                speeds_head[i] = np.sign(
                    speed) * self._min_speed_meters_sec
        for i, speed in enumerate(speeds_head[3:]):
            if abs(speed) < self._min_speed_rad_sec:
                speeds_head[i] = np.sign(speed) * self._min_speed_rad_sec

        # Little boost against gravity so that speed is closer to the desired speed
        if speeds_head[4] < 0:
            speeds_head[4] += np.sign(speeds_head[4]) * 0.5

        steps_size_head = np.array(speeds_head) * self._minTime

        if current_head_pose is None:
            head_pose = np.array(self.current_head_pose) + steps_size_head
        else:
            head_pose = np.array(current_head_pose) + steps_size_head

        if (np.abs(steps_size_head) > 0).any():
            self._head_msg(head_pose)

            if should_sleep:
                rospy.sleep(self._minTime)

    def move_torso(self, pose, should_wait=False, speed_rad_sec=1.0e10, stop_cb=None, timeout=float('inf')):
        if self._hbba_filter_state.is_filtering_all_messages or (stop_cb and stop_cb()):
            return False

        if speed_rad_sec < self._min_speed_rad_sec:
            speed_rad_sec = self._min_speed_rad_sec

        steps_size = speed_rad_sec * self._minTime

        pose = fmod_radian(pose)
        distance = pose - self.current_torso_pose
        if distance < -math.pi:
            distance = 2 * math.pi + distance
        elif distance > math.pi:
            distance = -(2 * math.pi - distance)

        if distance < 0:
            steps_size = -steps_size

        start_time = time.time()
        if abs(distance) > abs(steps_size):
            steps_number = int(abs(distance / steps_size))

            for i in range(1, steps_number):
                if self._hbba_filter_state.is_filtering_all_messages or (stop_cb and stop_cb()):
                    return False
                steps_cumulative_size = i * steps_size
                offset = distance - steps_cumulative_size
                self._torso_orientation_pub.publish(pose - offset)
                rospy.sleep(self._minTime)
            self._torso_orientation_pub.publish(pose)
        else:
            if self._hbba_filter_state.is_filtering_all_messages or (stop_cb and stop_cb()):
                return False
            self._torso_orientation_pub.publish(pose)

        if should_wait:
            while abs_diff_torso_angle(pose, self.current_torso_pose) > 0.1:
                if self._hbba_filter_state.is_filtering_all_messages or (stop_cb and stop_cb()):
                    return False
                if (time.time() - start_time) > timeout:
                    raise TimeoutError()

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
                np_steps_number[i] = int(
                    abs(np_distances[i] / np_steps_size[i]))
            else:
                np_steps_number[i] = 1

        return np_steps_number

    def _head_msg(self, pose):
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = 'stewart_base'

        pose_msg.pose.position.x = pose[0]
        pose_msg.pose.position.y = pose[1]
        pose_msg.pose.position.z = pose[2]

        q = quaternion_from_euler(pose[3], pose[4], pose[5])

        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]

        self._head_pose_pub.publish(pose_msg)

    def move_head(self, pose, should_wait=False,
                  speed_meters_sec=1.0e10, speed_rad_sec=1.0e10,
                  stop_cb=None, timeout=float('inf')):
        """
        pose[0]: x
        pose[1]: y
        pose[2]: z
        pose[3]: roll (x)
        pose[4]: pitch (y)
        pose[5]: yaw (z)
        """
        if self._hbba_filter_state.is_filtering_all_messages or (stop_cb and stop_cb()):
            return False

        if speed_meters_sec < self._min_speed_meters_sec:
            speed_meters_sec = self._min_speed_meters_sec

        if speed_rad_sec < self._min_speed_rad_sec:
            speed_rad_sec = self._min_speed_rad_sec

        np_pose = np.array(pose)

        np_distances = np_pose - np.array(self.current_head_pose)

        np_steps_size = self._compute_head_steps_size_array(
            speed_meters_sec, speed_rad_sec, np_distances)

        np_steps_number = self._compute_head_steps_number_array(
            np_distances, np_steps_size)

        start_time = time.time()
        if int(np.amax(np_steps_number)) > 1:
            for i in range(1, int(np.amax(np_steps_number))):
                if self._hbba_filter_state.is_filtering_all_messages or (stop_cb and stop_cb()):
                    return False

                np_steps_cumulative_size = i * np_steps_size
                np_offsets = np_distances - np_steps_cumulative_size

                for j in range(np_steps_number.size):
                    if np_steps_number[j] <= i:
                        np_offsets[j] = 0

                self._head_msg(np_pose - np_offsets)
                rospy.sleep(self._minTime)
            self._head_msg(pose)
        else:
            if self._hbba_filter_state.is_filtering_all_messages or (stop_cb and stop_cb()):
                return False
            self._head_msg(pose)

        if should_wait:
            while not (np.abs(np_pose - np.array(self.current_head_pose)) < self._np_head_tolerances).all():
                if self._hbba_filter_state.is_filtering_all_messages or (stop_cb and stop_cb()):
                    return False
                if (time.time() - start_time) > timeout:
                    raise TimeoutError()

                self._head_msg(pose)
                rospy.sleep(self._minTime)

        return True

    def move_yes(self, count=3, speed_rad_sec=0.5, timeout=float('inf')):
        timeout /= 2 * count

        for i in range(count):
            if not self.move_head([0, 0, HEAD_ZERO_Z, 0, 0.25, 0], should_wait=True,
                                  speed_rad_sec=speed_rad_sec, timeout=timeout):
                return False
            if not self.move_head([0, 0, HEAD_ZERO_Z, 0, -0.25, 0], should_wait=True,
                                  speed_rad_sec=speed_rad_sec, timeout=timeout):
                return False

        if not self.move_head([0, 0, HEAD_ZERO_Z, 0, 0, 0], should_wait=True,
                              speed_rad_sec=speed_rad_sec, timeout=timeout):
            return False
        return True

    def move_no(self, count=3, speed_rad_sec=0.5, timeout=float('inf')):
        timeout /= 2 * count

        for i in range(count):
            if not self.move_head([0, 0, HEAD_ZERO_Z, 0, 0, 0.25], should_wait=True,
                                  speed_rad_sec=speed_rad_sec, timeout=timeout):
                return False
            if not self.move_head([0, 0, HEAD_ZERO_Z, 0, 0, -0.25], should_wait=True,
                                  speed_rad_sec=speed_rad_sec, timeout=timeout):
                return False

        if not self.move_head([0, 0, HEAD_ZERO_Z, 0, 0, 0], should_wait=True,
                              speed_rad_sec=speed_rad_sec, timeout=timeout):
            return False
        return True

    def move_maybe(self, count=3, speed_rad_sec=0.5, timeout=float('inf')):
        timeout /= 2 * count

        for i in range(count):
            if not self.move_head([0, 0, HEAD_ZERO_Z, 0.25, 0, 0],should_wait= True,
                                  speed_rad_sec=speed_rad_sec, timeout=timeout):
                return False
            if not self.move_head([0, 0, HEAD_ZERO_Z, -0.25, 0, 0], should_wait=True,
                                  speed_rad_sec=speed_rad_sec, timeout=timeout):
                return False

        if not self.move_head([0, 0, HEAD_ZERO_Z, 0, 0, 0], should_wait=True,
                              speed_rad_sec=speed_rad_sec, timeout=timeout):
            return False
        return True

    def move_head_to_thinking(self, speed_rad_sec=0.5, timeout=float('inf')):
        self.move_head([0, 0, HEAD_ZERO_Z, 0.25, 0, 0], should_wait=True, speed_rad_sec=speed_rad_sec, timeout=timeout)

    def move_head_to_sad(self, speed_rad_sec=0.5, timeout=float('inf')):
        self.move_head([0, 0, HEAD_ZERO_Z, 0, 0.25, 0], should_wait=True, speed_rad_sec=speed_rad_sec, timeout=timeout)

    def move_head_to_origin(self, should_wait=True, speed_meters_sec=1.0e10,
                            speed_rad_sec=1.0e10, timeout=float('inf')):
        self.move_head([0, 0, HEAD_ZERO_Z, 0, 0, 0], should_wait, speed_meters_sec,
                       speed_rad_sec, timeout=timeout)

    def move_torso_to_origin(self, should_wait=True, timeout=float('inf')):
        self.move_torso(0, should_wait, timeout=timeout)
