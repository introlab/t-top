import math
import threading

import rospy

from t_top import MovementCommands, HEAD_ZERO_Z, HEAD_POSE_PITCH_INDEX


TARGET_HEAD_IMAGE_Y = 0.5


class FaceFollowingNode:
    def __init__(self, namespace):
        self._simulation = rospy.get_param('~simulation')
        self._rate = rospy.Rate(rospy.get_param('~control_frequency'))
        self._torso_control_alpha = rospy.get_param('~torso_control_alpha')
        self._head_control_p_gain = rospy.get_param('~head_control_p_gain')
        self._head_enabled = rospy.get_param('~head_enabled')
        self._min_head_pitch = rospy.get_param('~min_head_pitch_rad')
        self._max_head_pitch = rospy.get_param('~max_head_pitch_rad')

        self._target_lock = threading.Lock()
        self._target_torso_yaw = None
        self._current_head_image_y = None

        self._movement_commands = MovementCommands(self._simulation, namespace)

    def _update(self, yaw, head_image_y):
        with self._target_lock:
            if yaw is None or math.isfinite(yaw):
                self._target_torso_yaw = yaw
            if head_image_y is None or math.isfinite(head_image_y):
                self._current_head_image_y = head_image_y

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

        pose = self._movement_commands.current_torso_pose + self._torso_control_alpha * distance
        self._movement_commands.move_torso(pose)

    def _update_head(self):
        with self._target_lock:
            current_head_image_y = self._current_head_image_y
        if current_head_image_y is None:
            return

        current_pitch = self._movement_commands.current_head_pose[HEAD_POSE_PITCH_INDEX]
        pitch = current_pitch + self._head_control_p_gain * (current_head_image_y - TARGET_HEAD_IMAGE_Y)
        pitch = max(self._min_head_pitch, min(pitch, self._max_head_pitch))
        self._movement_commands.move_head([0, 0, HEAD_ZERO_Z, 0, pitch, 0])
