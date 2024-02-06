#!/usr/bin/env python3

import math

import numpy as np
from scipy.spatial.transform import Rotation as R

import rospy
import tf
from geometry_msgs.msg import PointStamped, Point, Vector3
from video_analyzer.msg import VideoAnalysis
from std_msgs.msg import String

from t_top import vector_to_angles


PERSON_POSE_NOSE_INDEX = 0
PERSON_POSE_LEFT_EYE_INDEX = 1
PERSON_POSE_RIGHT_EYE_INDEX = 2


class NearestFaceOrientationNode:
    def __init__(self):
        self._nose_confidence_threshold = rospy.get_param('~nose_confidence_threshold')
        self._pitch_offset_rad = rospy.get_param('~pitch_offset_rad')
        self._filter_alpha = rospy.get_param('~filter_alpha')
        self._roll_dead_zone = rospy.get_param('~roll_dead_zone')

        self._roll = 0.0
        self._pitch = 0.0

        self._tf_listener = tf.TransformListener()

        self._face_orientation_pub = rospy.Publisher('face_orientation', String, queue_size=1)
        self._video_analysis_sub = rospy.Subscriber('video_analysis', VideoAnalysis, self._video_analysis_cb, queue_size=1)

    def _video_analysis_cb(self, msg):
        if not msg.contains_3d_positions:
            rospy.logerr('The video analysis must contain 3d positions.')
            return

        self._update_nearest_face_orientation(msg.objects, msg.header)
        msg = String()
        msg.data = f'{self._roll},{self._pitch}'
        self._face_orientation_pub.publish(msg)

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

        stewart_base_point = self._tf_listener.transformPoint('/stewart_base', temp_in_point)

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
        rospy.spin()


def main():
    rospy.init_node('nearest_face_orientation_node')
    nearest_face_orientation_node = NearestFaceOrientationNode()
    nearest_face_orientation_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
