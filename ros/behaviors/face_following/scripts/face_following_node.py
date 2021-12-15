#!/usr/bin/env python3

import threading

from librosa.core import pitch

import rospy
import tf
from geometry_msgs.msg import PointStamped
from video_analyzer.msg import VideoAnalysis

from t_top import MovementCommands, vector_to_angles, HEAD_ZERO_Z, HEAD_POSE_PITCH_INDEX


PERSON_POSE_NOSE_INDEX = 0


class FaceFollowingNode:
    def __init__(self):
        self._simulation = rospy.get_param('~simulation')
        self._rate = rospy.Rate(rospy.get_param('~control_frequency'))
        self._control_alpha = rospy.get_param('~control_alpha')
        self._nose_confidence_threshold = rospy.get_param('~nose_confidence_threshold')
        self._head_enabled = rospy.get_param('~head_enabled')

        self._target_lock = threading.Lock()
        self._target_torso_yaw = None
        self._target_head_pitch = None

        self._tf_listener = tf.TransformListener()

        self._movement_commands = MovementCommands(self._simulation)
        self._video_analysis_sub = rospy.Subscriber("video_analysis", VideoAnalysis, self._video_analysis_cb, queue_size=1)

    def _video_analysis_cb(self, msg):
        if self._movement_commands.is_filtering_all_messages:
            return

        yaw, pitch = self._find_nearest_face_yaw_pitch(msg.objects, msg.header)
        print(yaw, pitch)
        with self._target_lock:
            self._target_torso_yaw = yaw
            self._target_head_pitch = pitch

    def _find_nearest_face_yaw_pitch(self, objects, header):
        nose_points = []
        for object in objects:
            if len(object.person_pose) > 0 and len(object.person_pose_confidence) > 0 \
                    and object.person_pose_confidence[PERSON_POSE_NOSE_INDEX] > self._nose_confidence_threshold:
                nose_points.append(object.person_pose[PERSON_POSE_NOSE_INDEX])

        if len(nose_points) == 0:
            return None, None
        else:
            nose_point = min(nose_points, key=lambda p: p.x ** 2 + p.y ** 2 + p.z ** 2)

            self._transform_point(nose_point, header)
            return vector_to_angles(nose_point)

    def _transform_point(self, point, header):
        temp_in_point = PointStamped()
        temp_in_point.header = header
        temp_in_point.point.x = point.x
        temp_in_point.point.y = point.y
        temp_in_point.point.z = point.z

        base_link_point = self._tf_listener.transformPoint("/base_link", temp_in_point)
        stewart_base_point = self._tf_listener.transformPoint("/stewart_base", temp_in_point)

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

        pose = self._control_alpha * target_torso_yaw + (1 - self._control_alpha) * self._movement_commands.current_torso_pose
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
    rospy.init_node('face_following_node')
    face_following_node = FaceFollowingNode()
    face_following_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
