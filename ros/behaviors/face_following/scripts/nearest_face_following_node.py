#!/usr/bin/env python3

import rclpy

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs

from geometry_msgs.msg import PointStamped
from perception_msgs.msg import VideoAnalysis

from t_top import vector_to_angles

from face_following.lib_face_following_node import FaceFollowingNode


PERSON_POSE_NOSE_INDEX = 0


class NearestFaceFollowingNode(FaceFollowingNode):
    def __init__(self):
        super().__init__(node_name='nearest_face_following_node', namespace='nearest_face_following')
        self._nose_confidence_threshold = self.declare_parameter('nose_confidence_threshold', 0.4).get_parameter_value().double_value

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._video_analysis_sub = self.create_subscription(VideoAnalysis, 'video_analysis', self._video_analysis_cb, 1)

    def _video_analysis_cb(self, msg):
        if self._movement_commands.is_filtering_all_messages:
            return
        if not msg.contains_3d_positions:
            self.get_logger().error('The video analysis must contain 3d positions.')
            return

        try:
            yaw, head_image_y = self._find_nearest_face_yaw_head_image_y(msg.objects, msg.header)
            self._update(yaw, head_image_y)
        except TransformException as ex:
                self.get_logger().warn(f'Could not transform: {ex}')

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

        transform = self._tf_buffer.lookup_transform('base_link', header.frame_id, rclpy.time.Time.from_msg(header.stamp))
        base_link_point = tf2_geometry_msgs.do_transform_point(temp_in_point, transform)

        point.x = base_link_point.point.x
        point.y = base_link_point.point.y
        point.z = base_link_point.point.z


def main():
    rclpy.init()
    nearest_face_following_node = NearestFaceFollowingNode()

    try:
        nearest_face_following_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        nearest_face_following_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
