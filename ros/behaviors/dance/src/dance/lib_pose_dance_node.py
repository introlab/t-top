#!/usr/bin/env python3

from abc import ABC, abstractmethod
import threading
import json
import random

import rospy
from std_msgs.msg import Float32
from std_msgs.msg import Bool

import hbba_lite


P_CHANGE_MOVEMENT = 0.25


class PoseDanceNode(ABC):
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

        self._hbba_filter_state = hbba_lite.OnOffHbbaFilterState('pose/filter_state')
        self._hbba_filter_state.on_changed(self._hbba_filter_state_cb)

        self._bpm_sub = rospy.Subscriber('bpm', Float32, self._bpm_cb, queue_size=1)
        self._beat_sub = rospy.Subscriber('beat', Bool, self._beat_cb, queue_size=1)

    def _hbba_filter_state_cb(self, previous_is_filtering_all_messages, new_is_filtering_all_messages):
        pass

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

    @abstractmethod
    def _send_pose(self, pose):
        """ Called with self._lock locked """
        pass


def main():
    rospy.init_node('dance_node')
    dance_node = DanceNode()
    dance_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
