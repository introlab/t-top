#!/usr/bin/env python3

import math
import threading

import numpy as np


import rclpy
import rclpy.node
import rclpy.executors

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs

from geometry_msgs.msg import PointStamped
from perception_msgs.msg import VideoAnalysis
from std_msgs.msg import String
from opentera_webrtc_ros_msgs.msg import PeerData

from t_top import MovementCommands, vector_to_angles, HEAD_ZERO_Z, HEAD_POSE_ROLL_INDEX, HEAD_POSE_PITCH_INDEX


PERSON_POSE_NOSE_INDEX = 0


class NearestFaceFollowingOrientationReplicationNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('nearest_face_following_orientation_replication_node')
        self._simulation = self.declare_parameter('simulation', False).get_parameter_value().bool_value
        self._control_frequency = self.declare_parameter('control_frequency', 30.0).get_parameter_value().double_value
        self._torso_control_alpha = self.declare_parameter('torso_control_alpha', 0.2).get_parameter_value().double_value
        self._head_control_alpha = self.declare_parameter('head_control_alpha', 0.1).get_parameter_value().double_value
        self._head_control_pitch_up_alpha_gain = self.declare_parameter('head_control_pitch_up_alpha_gain', 2.0).get_parameter_value().double_value
        self._min_head_roll = self.declare_parameter('min_head_roll_rad', -0.3).get_parameter_value().double_value
        self._max_head_roll = self.declare_parameter('max_head_roll_rad', 0.3).get_parameter_value().double_value
        self._min_head_pitch = self.declare_parameter('min_head_pitch_rad', -0.15).get_parameter_value().double_value
        self._max_head_pitch = self.declare_parameter('max_head_pitch_rad', 0.3).get_parameter_value().double_value
        self._nose_confidence_threshold = self.declare_parameter('nose_confidence_threshold', 0.4).get_parameter_value().double_value

        self._target_lock = threading.Lock()
        self._target_torso_yaw = None
        self._target_head_roll = 0.0
        self._target_head_pitch = 0.0

        self._movement_commands = MovementCommands(self, self._simulation, 'other')

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._video_analysis_sub = self.create_subscription(VideoAnalysis, 'video_analysis', self._video_analysis_cb, 1)
        self._face_orientation_sub = self.create_subscription(String, 'face_orientation', self._face_orientation_cb, 1)
        self._face_orientation_peer_data_sub = self.create_subscription(PeerData, 'face_orientation_peer_data', self._face_orientation_cb, 1)

        self._timer = self.create_timer(1 / self._control_frequency, self._timer_callback)

    def _video_analysis_cb(self, msg):
        if self._movement_commands.is_filtering_all_messages:
            return
        if not msg.contains_3d_positions:
            self.get_logger().error('The video analysis must contain 3d positions.')
            return

        try:
            face_position = self._find_nearest_face_position(msg.objects, msg.header)
            if face_position is None:
                return

            target_torso_yaw, _ = vector_to_angles(face_position)
            if math.isfinite(target_torso_yaw):
                with self._target_lock:
                    self._target_torso_yaw = target_torso_yaw
        except TransformException as e:
            self.get_logger().error(f'Could not transform: {e}')

    def _find_nearest_face_position(self, objects, header):
        nose_points_3d = []
        for i, o in enumerate(objects):
            if (len(o.person_pose_3d) > 0 and
                len(o.person_pose_confidence) > 0 and
                o.person_pose_confidence[PERSON_POSE_NOSE_INDEX] > self._nose_confidence_threshold):
                nose_points_3d.append((i, o.person_pose_3d[PERSON_POSE_NOSE_INDEX]))

        if len(nose_points_3d) == 0:
            return None
        else:
            nose_point_index = min(nose_points_3d, key=lambda p: p[1].x ** 2 + p[1].y ** 2 + p[1].z ** 2)[0]
            nose_point_3d = objects[nose_point_index].person_pose_3d[PERSON_POSE_NOSE_INDEX]

            self._transform_point(nose_point_3d, header)

            return nose_point_3d

    def _transform_point(self, point, header):
        temp_in_point = PointStamped()
        temp_in_point.header = header
        temp_in_point.point.x = point.x
        temp_in_point.point.y = point.y
        temp_in_point.point.z = point.z

        transform = self._tf_buffer.lookup_transform('base_link', header.frame_id, rclpy.time.Time.from_msg(header.stamp))
        base_link_point = tf2_geometry_msgs.do_transform_point(temp_in_point, transform)

        point.x = base_link_point.point.x
        point.y = base_link_point.point.y
        point.z = base_link_point.point.y

    def _face_orientation_cb(self, msg):
        fields = msg.data.split(',')
        if len(fields) != 2:
            self.get_logger().error('Invalid face orientation message')
            return

        roll = np.clip(float(fields[0]), self._min_head_roll, self._max_head_roll)
        pitch = np.clip(float(fields[1]), self._min_head_pitch, self._max_head_pitch)

        with self._target_lock:
            self._target_head_roll = roll
            self._target_head_pitch = pitch

    def _timer_callback(self):
        if self._movement_commands.is_filtering_all_messages:
            return

        self._update_torso()
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
            target_head_roll = self._target_head_roll
            target_head_pitch = self._target_head_pitch

        current_roll = self._movement_commands.current_head_pose[HEAD_POSE_ROLL_INDEX]
        current_pitch = self._movement_commands.current_head_pose[HEAD_POSE_PITCH_INDEX]

        roll = current_roll * (1.0 - self._head_control_alpha) + target_head_roll * self._head_control_alpha

        pitch_alpha = self._head_control_alpha
        if target_head_pitch < current_pitch:
            pitch_alpha *= self._head_control_pitch_up_alpha_gain

        pitch = current_pitch * (1.0 - pitch_alpha) + target_head_pitch * pitch_alpha

        self._movement_commands.move_head([0, 0, HEAD_ZERO_Z, roll, pitch, 0])

    def run(self):
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
        executor.add_node(self)
        executor.spin()

def main():
    rclpy.init()
    nearest_face_following_orientation_replication_node = NearestFaceFollowingOrientationReplicationNode()

    try:
        nearest_face_following_orientation_replication_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        nearest_face_following_orientation_replication_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
