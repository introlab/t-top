#!/usr/bin/env python3

import threading

import rospy
from gesture.msg import GestureName, Done

from t_top import MovementCommands


class GestureNode:
    def __init__(self):
        self._simulation = rospy.get_param('~simulation')

        self._gesture_lock = threading.Lock()
        self._movement_commands = MovementCommands(self._simulation)

        self._done_pub = rospy.Publisher('gesture/done', Done, queue_size=5)
        self._gesture_sub = rospy.Subscriber('gesture/name', GestureName, self._on_gesture_cb)

    def _on_gesture_cb(self, msg):
        with self._gesture_lock:
            if self._movement_commands.is_filtering_all_messages:
                return

            if msg.name == 'yes':
                self._movement_commands.move_yes(count=1)
            elif msg.name == 'no':
                self._movement_commands.move_no(count=1)
            elif msg.name == 'maybe':
                self._movement_commands.move_maybe(count=1)
            elif msg.name == 'origin_all':
                self._movement_commands.move_head_to_origin()
                self._movement_commands.move_torso_to_origin()
            elif msg.name == 'origin_head':
                self._movement_commands.move_head_to_origin()
            elif msg.name == 'origin_torso':
                self._movement_commands.move_torso_to_origin()
            else:
                rospy.logerr(f'Invalid gesture name ({msg.name})')

            self._done_pub.publish(Done(id=msg.id))

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('gesture_node')
    gesture_node = GestureNode()
    gesture_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
