#!/usr/bin/env python3

import math

import rclpy
import rclpy.node

from odas_ros_msgs.msg import OdasSstArrayStamped

from t_top import MovementCommands, vector_to_angles, HEAD_ZERO_Z, HEAD_POSE_PITCH_INDEX


TARGET_TOLERANCE = 0.02


class SoundFollowingNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('sound_following_node')

        self._simulation = self.declare_parameter('simulation', False).get_parameter_value().bool_value
        self._control_frequency = self.declare_parameter('control_frequency', 30.0).get_parameter_value().double_value
        self._torso_control_alpha = self.declare_parameter('torso_control_alpha', 0.2).get_parameter_value().double_value
        self._head_control_alpha = self.declare_parameter('head_control_alpha', 0.2).get_parameter_value().double_value
        self._head_enabled = self.declare_parameter('head_enabled', False).get_parameter_value().bool_value
        self._min_head_pitch = self.declare_parameter('min_head_pitch_rad', -0.35).get_parameter_value().double_value
        self._max_head_pitch = self.declare_parameter('max_head_pitch_rad', 0.35).get_parameter_value().double_value
        self._min_activity = self.declare_parameter('min_activity', 0.1).get_parameter_value().double_value
        self._min_valid_source_pitch = self.declare_parameter('min_valid_source_pitch_rad', -1.4).get_parameter_value().double_value
        self._max_valid_source_pitch = self.declare_parameter('max_valid_source_pitch_rad', 1.4).get_parameter_value().double_value
        self._direction_frame_id = self.declare_parameter('direction_frame_id', 'odas').get_parameter_value().string_value

        self._target_torso_yaw = None
        self._target_head_pitch = None

        self._movement_commands = MovementCommands(self, self._simulation, namespace='sound_following')
        self._sst_sub = self.create_subscription(OdasSstArrayStamped, 'sst', self._sst_cb, 10)

        self._timer = self.create_timer(1 / self._control_frequency, self._timer_callback)

    def _sst_cb(self, sst):
        if self._movement_commands.is_filtering_all_messages:
            return
        if len(sst.sources) > 1:
            self.get_logger().error(f'Invalid sst (len(sst.sources)={len(sst.sources)})')
            return
        if sst.header.frame_id != self._direction_frame_id:
            self.get_logger().error(f'Invalid direction frame id ({sst.header.frame_id} != {self._direction_frame_id})')
            return
        if len(sst.sources) == 0 or sst.sources[0].activity < self._min_activity:
            return

        yaw, pitch = vector_to_angles(sst.sources[0])
        if pitch < self._min_valid_source_pitch or pitch > self._max_valid_source_pitch:
            return

        self._target_torso_yaw = yaw
        self._target_head_pitch = None if pitch is None else max(self._min_head_pitch, min(pitch, self._max_head_pitch))

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
        if abs(distance) < TARGET_TOLERANCE:
            return

        if distance < -math.pi:
            distance = 2 * math.pi + distance
        elif distance > math.pi:
            distance = -(2 * math.pi - distance)

        pose = self._movement_commands.current_torso_pose + self._torso_control_alpha * distance
        self._movement_commands.move_torso(pose)

    def _update_head(self):
        if self._target_head_pitch is None:
            return

        current_pitch = self._movement_commands.current_head_pose[HEAD_POSE_PITCH_INDEX]
        if abs(self._target_head_pitch - current_pitch) < TARGET_TOLERANCE:
            return

        pitch = self._head_control_alpha * self._target_head_pitch + (1 - self._head_control_alpha) * current_pitch
        self._movement_commands.move_head([0, 0, HEAD_ZERO_Z, 0, pitch, 0])


def main():
    rclpy.init()
    sound_following_node = SoundFollowingNode()

    try:
        sound_following_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        sound_following_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
