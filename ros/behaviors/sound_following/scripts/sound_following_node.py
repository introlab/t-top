#!/usr/bin/env python3

import math
import threading

import rospy
from odas_ros.msg import OdasSstArrayStamped

from t_top import MovementCommands, vector_to_angles, HEAD_ZERO_Z, HEAD_POSE_PITCH_INDEX


class SoundFollowingNode:
    def __init__(self):
        self._simulation = rospy.get_param('~simulation')
        self._rate = rospy.Rate(rospy.get_param('~control_frequency'))
        self._control_alpha = rospy.get_param('~control_alpha')
        self._head_enabled = rospy.get_param('~head_enabled')
        self._min_head_pitch = rospy.get_param('~min_head_pitch_rad')
        self._max_head_pitch = rospy.get_param('~max_head_pitch_rad')

        self._target_lock = threading.Lock()
        self._target_torso_yaw = None
        self._target_head_pitch = None

        self._movement_commands = MovementCommands(self._simulation)
        self._sst_sub = rospy.Subscriber('sst', OdasSstArrayStamped, self._sst_cb, queue_size=10)

    def _sst_cb(self, sst):
        if len(sst.sources) > 1:
            rospy.logerr('Invalid sst (len(sst.sources)={})'.format(len(sst.sources)))
            return

        if self._movement_commands.is_filtering_all_messages or len(sst.sources) == 0:
            yaw, pitch = None, None
        else:
            yaw, pitch = vector_to_angles(sst.sources[0])

        with self._target_lock:
            self._target_torso_yaw = yaw
            self._target_head_pitch = None if pitch is None else max(self._min_head_pitch, min(pitch, self._max_head_pitch))

    def run(self):
        while not rospy.is_shutdown():
            self._rate.sleep()
            if self._movement_commands.is_filtering_all_messages:
                continue

            self._update_torso()
            if self._head_enabled:
                self._update_head()

    def _update_torso(self):
        with self._target_lock:
            target_torso_yaw = self._target_torso_yaw
        if target_torso_yaw is None:
            return

        distance = target_torso_yaw - self._movement_commands.current_torso_pose
        if distance < -math.pi:
            distance = 2 * math.pi + distance
        elif distance > math.pi:
            distance = -(2 * math.pi - distance)

        pose = self._movement_commands.current_torso_pose + self._control_alpha * distance
        self._movement_commands.move_torso(pose)

    def _update_head(self):
        with self._target_lock:
            target_head_pitch = self._target_head_pitch
        if target_head_pitch is None:
            return

        current_pitch = self._movement_commands.current_head_pose[HEAD_POSE_PITCH_INDEX]
        pitch = self._control_alpha * target_head_pitch + (1 - self._control_alpha) * current_pitch
        self._movement_commands.move_head([0, 0, HEAD_ZERO_Z, 0, pitch, 0])


def main():
    rospy.init_node('sound_following_node')
    sound_following_node = SoundFollowingNode()
    sound_following_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
