#!/usr/bin/env python3

import rclpy

from std_msgs.msg import String
from perception_msgs.msg import PersonNames

from t_top import vector_to_angles

from face_following.lib_face_following_node import FaceFollowingNode


class SpecificFaceFollowingNode(FaceFollowingNode):
    def __init__(self):
        super().__init__(node_name='specific_face_following_node', namespace='specific_face_following')
        self._direction_frame_id = self.declare_parameter('direction_frame_id', 'odas').get_parameter_value().string_value

        self._target_name = None

        self._target_name_sub = self.create_subscription(String, 'target_name', self._target_name_cb, 1)
        self._person_names_sub = self.create_subscription(PersonNames, 'person_names', self._person_names_cb, 1)

    def _target_name_cb(self, msg):
        self._target_name = msg.data

    def _person_names_cb(self, msg):
        if self._movement_commands.is_filtering_all_messages:
            return

        for person_name in msg.names:
            if person_name.detection_type != 'voice' and person_name.name == self._target_name:
                if person_name.frame_id != self._direction_frame_id:
                    self.get_logger().error(f'Invalid direction frame id ({person_name.frame_id} != {self._direction_frame_id})')
                    return
                if len(person_name.direction) == 0 or len(person_name.position_2d) == 0:
                    self.get_logger().error(f'The direction and position_2d must not be empty.')
                    return

                yaw, _ = vector_to_angles(person_name.direction[0])
                head_image_y = person_name.position_2d[0].y
                self._update(yaw, head_image_y)
                break


def main():
    rclpy.init()
    specific_face_following_node = SpecificFaceFollowingNode()

    try:
        specific_face_following_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        specific_face_following_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
