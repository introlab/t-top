#!/usr/bin/env python3

import threading
import json
import random

import rospy
from std_msgs.msg import Float32
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped

import hbba_lite
from t_top import HEAD_ZERO_Z


P_CHANGE_MOVEMENT = 0.25


class DanceNode:
    def __init__(self):
        self._lock = threading.Lock()

        with open(rospy.get_param('~movement_file'), 'r') as f:
            self._movements = json.load(f)

        self._current_movement = self._movements[list(self._movements.keys())[0]]
        self._current_movement_pose_index = 0
        self._last_pose_time = rospy.Time.now()
        self._bpm = 120
        self._peak_delay_pose_count = 0


        self._rate = rospy.Rate(100)

        self._head_pose_pub = rospy.Publisher('opencr/head_pose', PoseStamped, queue_size=5)
        self._torso_orientation_pub = rospy.Publisher('opencr/torso_orientation', Float32, queue_size=5)

        self._hbba_filter_state = hbba_lite.OnOffHbbaFilterState('pose/filter_state')
        self._bpm_sub = rospy.Subscriber('bpm', Float32, self._bpm_cb, queue_size=1)
        self._beat_sub = rospy.Subscriber('beat', Bool, self._beat_cb, queue_size=1)

    def _bpm_cb(self, msg):
        if msg.data > 0:
            with self._lock:
                self._bpm = msg.data

    def _beat_cb(self, msg):
        if msg.data:
            with self._lock:
                self._peak_delay_pose_count = self._find_peak_delay_pose_count()

    def _find_peak_delay_pose_count(self):
        relative_index = float(self._current_movement_pose_index) / (len(self._current_movement['poses']) - 1)
        absolute_difference_function = lambda list_value : abs(list_value - relative_index)
        closest_value = min(self._current_movement['peak_relative_index'], key=absolute_difference_function)

        return (closest_value - relative_index) * (len(self._current_movement['poses']) - 1)

    def run(self):
        while not rospy.is_shutdown():
            with self._lock:
                if not self._hbba_filter_state.is_filtering_all_messages:
                    pose_duration = self._get_pose_duration()
                    if (rospy.Time.now() - self._last_pose_time) > pose_duration:
                        self._send_pose(self._current_movement['poses'][self._current_movement_pose_index])
                        self._current_movement_pose_index += 1
                        self._last_pose_time = rospy.Time.now()

                        if self._current_movement_pose_index == len(self._current_movement['poses']):
                            self._current_movement_pose_index = 0
                            self._update_movement()
                else:
                    self._current_movement_pose_index = 0
                    self._last_pose_time = rospy.Time.now()

            self._rate.sleep()

    def _update_movement(self):
        if random.random() < P_CHANGE_MOVEMENT:
            movement_names = list(self._movements.keys())
            index = random.randrange(0, len(movement_names))
            self._current_movement = self._movements[movement_names[index]]

    def _get_pose_duration(self):
        beat_duration = self._current_movement['beat_duration']
        pose_count = len(self._current_movement['poses']) + self._peak_delay_pose_count

        return rospy.Duration.from_sec(float(beat_duration) / pose_count / self._bpm * 60.0)

    def _send_pose(self, pose):
        if len(pose) == 7:
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = 'stewart_base'

            pose_msg.pose.position.x = pose[0]
            pose_msg.pose.position.y = pose[1]
            pose_msg.pose.position.z = HEAD_ZERO_Z + pose[2]

            pose_msg.pose.orientation.x = pose[3]
            pose_msg.pose.orientation.y = pose[4]
            pose_msg.pose.orientation.z = pose[5]
            pose_msg.pose.orientation.w = pose[6]

            self._head_pose_pub.publish(pose_msg)
        elif len(pose) == 1:
            pose_msg = Float32()
            pose_msg.data = pose[0]

            self._torso_orientation_pub.publish(pose_msg)
        else:
            rospy.logerr('Invalid pose')


def main():
    rospy.init_node('dance_node')
    dance_node = DanceNode()
    dance_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
