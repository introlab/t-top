#!/usr/bin/env python3

import math

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
import rclpy.node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs

from geometry_msgs.msg import PointStamped, Point, Vector3
from perception_msgs.msg import VideoAnalysis
from std_msgs.msg import String

from t_top import vector_to_angles


PERSON_POSE_NOSE_INDEX = 0
PERSON_POSE_LEFT_EYE_INDEX = 1
PERSON_POSE_RIGHT_EYE_INDEX = 2


class NearestFaceOrientationNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('nearest_face_orientation_node')

        self._nose_confidence_threshold = self.declare_parameter('nose_confidence_threshold', 0.4).get_parameter_value().double_value
        self._pitch_offset_rad = self.declare_parameter('pitch_offset_rad', -0.8).get_parameter_value().double_value
        self._filter_alpha = self.declare_parameter('filter_alpha', 0.65).get_parameter_value().double_value
        self._roll_dead_zone = self.declare_parameter('roll_dead_zone', 0.05).get_parameter_value().double_value

        self._roll = 0.0
        self._pitch = 0.0

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._face_orientation_pub = self.create_publisher(String, 'face_orientation', 1)
        self._video_analysis_sub = self.create_subscription(VideoAnalysis, 'video_analysis', self._video_analysis_cb, 1)

    def _video_analysis_cb(self, msg):
        if not msg.contains_3d_positions:
            self.get_logger().error('The video analysis must contain 3d positions.')
            return

        try:
            self._update_nearest_face_orientation(msg.objects, msg.header)
            msg = String()
            msg.data = f'{self._roll},{self._pitch}'
            self._face_orientation_pub.publish(msg)
        except TransformException as e:
            self.get_logger().error(f'Could not transform: {e}')

    def _update_nearest_face_orientation(self, objects, header):
        poses = []
        for i, o in enumerate(objects):
            if (len(o.person_pose_3d) > 0 and
                len(o.person_pose_confidence) > 0 and
                o.person_pose_confidence[PERSON_POSE_NOSE_INDEX] > self._nose_confidence_threshold and
                o.person_pose_confidence[PERSON_POSE_LEFT_EYE_INDEX] > self._nose_confidence_threshold and
                o.person_pose_confidence[PERSON_POSE_RIGHT_EYE_INDEX] > self._nose_confidence_threshold):
                poses.append((i, o.person_pose_3d))

        if len(poses) > 0:
            pose_index = min(poses, key=lambda p: (p[1][PERSON_POSE_NOSE_INDEX].x ** 2 +
                                                   p[1][PERSON_POSE_NOSE_INDEX].y ** 2 +
                                                   p[1][PERSON_POSE_NOSE_INDEX].z ** 2))[0]
            self._update_face_orientation(objects[pose_index].person_pose_3d, header)

    def _update_face_orientation(self, person_pose_3d, header):
            nose_point_3d = person_pose_3d[PERSON_POSE_NOSE_INDEX]
            left_eye_point_3d = person_pose_3d[PERSON_POSE_LEFT_EYE_INDEX]
            right_eye_point_3d = person_pose_3d[PERSON_POSE_RIGHT_EYE_INDEX]

            self._transform_point(nose_point_3d, header)
            self._transform_point(left_eye_point_3d, header)
            self._transform_point(right_eye_point_3d, header)

            left_to_right_eye_vector = self._vector_from_to(left_eye_point_3d, right_eye_point_3d)
            yaw = self._vector_to_yaw(left_to_right_eye_vector)
            if not math.isfinite(yaw):
                return

            self._rotate_point_yaw(nose_point_3d, yaw)
            self._rotate_point_yaw(left_eye_point_3d, yaw)
            self._rotate_point_yaw(right_eye_point_3d, yaw)

            eye_middle_point_3d = self._point_middle(left_eye_point_3d, right_eye_point_3d)
            nose_to_eye_middle_vector = self._vector_from_to(nose_point_3d, eye_middle_point_3d)

            roll, pitch = self._vector_to_roll_pitch(nose_to_eye_middle_vector)
            if math.isfinite(roll):
                if abs(self._roll_dead_zone) < self._roll_dead_zone:
                    roll = 0.0
                self._roll = self._roll * (1.0 - self._filter_alpha) + roll * self._filter_alpha
            if math.isfinite(pitch):
                pitch += self._pitch_offset_rad
                self._pitch = self._pitch * (1.0 - self._filter_alpha) + pitch * self._filter_alpha

    def _transform_point(self, point, header):
        temp_in_point = PointStamped()
        temp_in_point.header = header
        temp_in_point.point.x = point.x
        temp_in_point.point.y = point.y
        temp_in_point.point.z = point.z

        transform = self._tf_buffer.lookup_transform('stewart_base', header.frame_id, rclpy.time.Time.from_msg(header.stamp))
        stewart_base_point = tf2_geometry_msgs.do_transform_point(temp_in_point, transform)

        point.x = stewart_base_point.point.x
        point.y = stewart_base_point.point.y
        point.z = stewart_base_point.point.z

    def _vector_from_to(self, from_p, to_p):
        v = Vector3()
        v.x = to_p.x - from_p.x
        v.y = to_p.y - from_p.y
        v.z = to_p.z - from_p.z
        return v

    def _vector_to_yaw(self, vector):
        vector = np.array([vector.x, vector.y, vector.z])
        vector /= np.linalg.norm(vector)

        return np.arctan2(vector[0], vector[1])

    def _rotate_point_yaw(self, point, yaw):
        p = np.array([point.x, point.y, point.z])
        r = R.from_euler('z', yaw)
        p = r.apply(p)
        point.x = p[0]
        point.y = p[1]
        point.z = p[2]

    def _point_middle(self, p1, p2):
        m = Point()
        m.x = (p1.x + p2.x) / 2.0
        m.y = (p1.y + p2.y) / 2.0
        m.z = (p1.z + p2.z) / 2.0
        return m

    def _vector_to_roll_pitch(self, vector):
        vector = np.array([vector.x, vector.y, vector.z])
        vector /= np.linalg.norm(vector)

        roll = np.arctan2(vector[1], vector[2])
        pitch = np.arctan2(vector[2], vector[0])

        return roll, pitch

    def run(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    nearest_face_orientation_node = NearestFaceOrientationNode()

    try:
        nearest_face_orientation_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        nearest_face_orientation_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
