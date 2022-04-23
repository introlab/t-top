#!/usr/bin/env python3

import math
import threading

import rospy
import tf
from geometry_msgs.msg import PointStamped
from video_analyzer.msg import VideoAnalysis

from t_top import MovementCommands, vector_to_angles, HEAD_ZERO_Z, HEAD_POSE_PITCH_INDEX


PERSON_POSE_NOSE_INDEX = 0
TARGET_HEAD_IMAGE_Y = 0.5


class FaceFollowingNode:
    def __init__(self):
        self._simulation = rospy.get_param('~simulation')
        self._rate = rospy.Rate(rospy.get_param('~control_frequency'))
        self._torso_control_alpha = rospy.get_param('~torso_control_alpha')
        self._head_control_p_gain = rospy.get_param('~head_control_p_gain')
        self._nose_confidence_threshold = rospy.get_param('~nose_confidence_threshold')
        self._head_enabled = rospy.get_param('~head_enabled')
        self._min_head_pitch = rospy.get_param('~min_head_pitch_rad')
        self._max_head_pitch = rospy.get_param('~max_head_pitch_rad')

        self._target_lock = threading.Lock()
        self._target_torso_yaw = None
        self._current_head_image_y = None

        self._tf_listener = tf.TransformListener()

        self._movement_commands = MovementCommands(self._simulation)
        self._video_analysis_sub = rospy.Subscriber('video_analysis', VideoAnalysis, self._video_analysis_cb, queue_size=1)

    def _video_analysis_cb(self, msg):
        yaw, head_image_y = self._find_nearest_face_yaw_head_image_y(msg.objects, msg.header)
        with self._target_lock:
            self._target_torso_yaw = yaw
            self._current_head_image_y = head_image_y

    def _find_nearest_face_yaw_head_image_y(self, objects, header):
        nose_points_3d = []
        for i, o in enumerate(objects):
            if len(o.person_pose_2d) > 0 and len(o.person_pose_3d) > 0 and len(o.person_pose_confidence) > 0 \
                    and o.person_pose_confidence[PERSON_POSE_NOSE_INDEX] > self._nose_confidence_threshold:
                nose_points_3d.append((i, o.person_pose_3d[PERSON_POSE_NOSE_INDEX]))

        if len(nose_points_3d) == 0:
            return None, None
        else:
            nose_point_index = min(nose_points_3d, key=lambda p: p[1].x ** 2 + p[1].y ** 2 + p[1].z ** 2)[0]
            nose_point_2d = objects[nose_point_index].person_pose_2d[PERSON_POSE_NOSE_INDEX]
            nose_point_3d = objects[nose_point_index].person_pose_3d[PERSON_POSE_NOSE_INDEX]

            self._transform_point(nose_point_3d, header)
            yaw, _ = vector_to_angles(nose_point_3d)

            return yaw, nose_point_2d.y

    def _transform_point(self, point, header):
        temp_in_point = PointStamped()
        temp_in_point.header = header
        temp_in_point.point.x = point.x
        temp_in_point.point.y = point.y
        temp_in_point.point.z = point.z

        base_link_point = self._tf_listener.transformPoint('/base_link', temp_in_point)
        stewart_base_point = self._tf_listener.transformPoint('/stewart_base', temp_in_point)

        point.x = base_link_point.point.x
        point.y = base_link_point.point.y
        point.z = stewart_base_point.point.z - HEAD_ZERO_Z

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


def main():
    rospy.init_node('face_following_node')
    face_following_node = FaceFollowingNode()
    face_following_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
