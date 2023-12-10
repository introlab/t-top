#!/usr/bin/env python3

import threading

import rospy
from std_msgs.msg import String
from person_identification.msg import PersonNames

from t_top import vector_to_angles

from face_following.lib_face_following_node import FaceFollowingNode


class SpecificFaceFollowingNode(FaceFollowingNode):
    def __init__(self):
        super().__init__(namespace='specific_face_following')
        self._direction_frame_id = rospy.get_param('~direction_frame_id')

        self._target_name_lock = threading.Lock()
        self._target_name = None

        self._target_name_sub = rospy.Subscriber('target_name', String, self._target_name_cb, queue_size=1)
        self._person_names_sub = rospy.Subscriber('person_names', PersonNames, self._person_names_cb, queue_size=1)

    def _target_name_cb(self, msg):
        with self._target_name_lock:
            self._target_name = msg.data

    def _person_names_cb(self, msg):
        if self._movement_commands.is_filtering_all_messages:
            return

        with self._target_name_lock:
            target_name = self._target_name

        for person_name in msg.names:
            if person_name.detection_type != 'voice' and person_name.name == target_name:
                if person_name.frame_id != self._direction_frame_id:
                    rospy.logerr(f'Invalid direction frame id ({person_name.frame_id} != {self._direction_frame_id})')
                    return
                if len(person_name.direction) == 0 or len(person_name.position_2d) == 0:
                    rospy.logerr(f'The direction and position_2d must not be empty.')
                    return

                yaw, _ = vector_to_angles(person_name.direction[0])
                head_image_y = person_name.position_2d[0].y
                self._update(yaw, head_image_y)
                break


def main():
    rospy.init_node('specific_face_following_node')
    face_following_node = SpecificFaceFollowingNode()
    face_following_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
