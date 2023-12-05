#!/usr/bin/env python3

import rospy
import tf
from geometry_msgs.msg import PointStamped
from video_analyzer.msg import VideoAnalysis

from t_top import vector_to_angles, HEAD_ZERO_Z

from face_following.lib_face_following_node import FaceFollowingNode


PERSON_POSE_NOSE_INDEX = 0


class NearestFaceFollowingNode(FaceFollowingNode):
    def __init__(self):
        super().__init__(namespace='nearest_face_following')
        self._nose_confidence_threshold = rospy.get_param('~nose_confidence_threshold')

        self._tf_listener = tf.TransformListener()
        self._video_analysis_sub = rospy.Subscriber('video_analysis', VideoAnalysis, self._video_analysis_cb, queue_size=1)

    def _video_analysis_cb(self, msg):
        if self._movement_commands.is_filtering_all_messages:
            return
        if not msg.contains_3d_positions:
            rospy.logerr('The video analysis must contain 3d positions.')
            return

        yaw, head_image_y = self._find_nearest_face_yaw_head_image_y(msg.objects, msg.header)
        self._update(yaw, head_image_y)

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


def main():
    rospy.init_node('nearest_face_following_node')
    face_following_node = NearestFaceFollowingNode()
    face_following_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
