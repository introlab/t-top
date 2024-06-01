import math

import rclpy
import rclpy.node

from t_top import MovementCommands, HEAD_ZERO_Z, HEAD_POSE_PITCH_INDEX


TARGET_HEAD_IMAGE_Y = 0.5


class FaceFollowingNode(rclpy.node.Node):
    def __init__(self, node_name, namespace):
        super().__init__(node_name)

        self._simulation = self.declare_parameter('simulation', False).get_parameter_value().bool_value
        self._control_frequency = self.declare_parameter('control_frequency', 30.0).get_parameter_value().double_value
        self._torso_control_alpha = self.declare_parameter('torso_control_alpha', 0.2).get_parameter_value().double_value
        self._head_control_p_gain = self.declare_parameter('head_control_p_gain', 0.175).get_parameter_value().double_value
        self._head_enabled = self.declare_parameter('head_enabled', True).get_parameter_value().bool_value
        self._min_head_pitch = self.declare_parameter('min_head_pitch_rad', -0.35).get_parameter_value().double_value
        self._max_head_pitch = self.declare_parameter('max_head_pitch_rad', 0.35).get_parameter_value().double_value

        self._target_torso_yaw = None
        self._current_head_image_y = None

        self._movement_commands = MovementCommands(self, self._simulation, namespace)

        self._timer = self.create_timer(1 / self._control_frequency, self._timer_callback)

    def _update(self, yaw, head_image_y):
        if yaw is None or math.isfinite(yaw):
            self._target_torso_yaw = yaw
        if head_image_y is None or math.isfinite(head_image_y):
            self._current_head_image_y = head_image_y

    def _timer_callback(self):
        if self._movement_commands.is_filtering_all_messages:
            return

        self._update_torso()
        if self._head_enabled:
            self._update_head()

    def run(self):
        rclpy.spin(self)

    def _update_torso(self):
        if self._target_torso_yaw is None:
            return

        distance = self._target_torso_yaw - self._movement_commands.current_torso_pose
        if distance < -math.pi:
            distance = 2 * math.pi + distance
        elif distance > math.pi:
            distance = -(2 * math.pi - distance)

        pose = self._movement_commands.current_torso_pose + self._torso_control_alpha * distance
        self._movement_commands.move_torso(pose)

    def _update_head(self):
        if self._current_head_image_y is None:
            return

        current_pitch = self._movement_commands.current_head_pose[HEAD_POSE_PITCH_INDEX]
        pitch = current_pitch + self._head_control_p_gain * (self._current_head_image_y - TARGET_HEAD_IMAGE_Y)
        pitch = max(self._min_head_pitch, min(pitch, self._max_head_pitch))
        self._movement_commands.move_head([0, 0, HEAD_ZERO_Z, 0, pitch, 0])
