#!/usr/bin/env python3

import math
import threading

import numpy as np

import rospy
import tf
from geometry_msgs.msg import PointStamped
from video_analyzer.msg import VideoAnalysis
from std_msgs.msg import String
from opentera_webrtc_ros_msgs.msg import PeerData

from t_top import MovementCommands, vector_to_angles, HEAD_ZERO_Z, HEAD_POSE_ROLL_INDEX, HEAD_POSE_PITCH_INDEX


PERSON_POSE_NOSE_INDEX = 0


class NearestFaceFollowingOrientationReplicationNode:
    def __init__(self):
        self._simulation = rospy.get_param('~simulation')
        self._rate = rospy.Rate(rospy.get_param('~control_frequency'))
        self._torso_control_alpha = rospy.get_param('~torso_control_alpha')
        self._head_control_alpha = rospy.get_param('~head_control_alpha')
        self._head_control_pitch_up_alpha_gain = rospy.get_param('~head_control_pitch_up_alpha_gain')
        self._min_head_roll = rospy.get_param('~min_head_roll_rad')
        self._max_head_roll = rospy.get_param('~max_head_roll_rad')
        self._min_head_pitch = rospy.get_param('~min_head_pitch_rad')
        self._max_head_pitch = rospy.get_param('~max_head_pitch_rad')

        self._nose_confidence_threshold = rospy.get_param('~nose_confidence_threshold')

        self._target_lock = threading.Lock()
        self._target_torso_yaw = None
        self._target_head_roll = 0.0
        self._target_head_pitch = 0.0

        self._movement_commands = MovementCommands(self._simulation, 'other')

        self._tf_listener = tf.TransformListener()
        self._video_analysis_sub = rospy.Subscriber('video_analysis', VideoAnalysis, self._video_analysis_cb, queue_size=1)
        self._face_orientation_sub = rospy.Subscriber('face_orientation', String, self._face_orientation_cb, queue_size=1)
        self._face_orientation_peer_data_sub = rospy.Subscriber('face_orientation_peer_data', PeerData, self._face_orientation_cb, queue_size=1)

    def _video_analysis_cb(self, msg):
        if self._movement_commands.is_filtering_all_messages:
            return
        if not msg.contains_3d_positions:
            rospy.logerr('The video analysis must contain 3d positions.')
            return

        face_position = self._find_nearest_face_position(msg.objects, msg.header)
        if face_position is None:
            return

        target_torso_yaw, _ = vector_to_angles(face_position)
        if math.isfinite(target_torso_yaw):
            with self._target_lock:
                self._target_torso_yaw = target_torso_yaw

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

        base_link_point = self._tf_listener.transformPoint('/base_link', temp_in_point)

        point.x = base_link_point.point.x
        point.y = base_link_point.point.y
        point.z = base_link_point.point.y

    def _face_orientation_cb(self, msg):
        fields = msg.data.split(',')
        if len(fields) != 2:
            rospy.logerr('Invalid face orientation message')
            return

        roll = np.clip(float(fields[0]), self._min_head_roll, self._max_head_roll)
        pitch = np.clip(float(fields[1]), self._min_head_pitch, self._max_head_pitch)

        with self._target_lock:
            self._target_head_roll = roll
            self._target_head_pitch = pitch

    def run(self):
        while not rospy.is_shutdown():
            self._rate.sleep()
            if self._movement_commands.is_filtering_all_messages:
                continue

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


def main():
    rospy.init_node('nearest_face_following_orientation_replication_node')
    nearest_face_following_orientation_replication_node = NearestFaceFollowingOrientationReplicationNode()
    nearest_face_following_orientation_replication_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
