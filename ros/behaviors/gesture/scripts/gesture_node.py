#!/usr/bin/env python3

import threading

import rospy
from gesture.msg import GestureName, Done

from t_top import MovementCommands


MOVE_YES_TIMEOUT = 10
MOVE_NO_TIMEOUT = 10
MOVE_MAYBE_TIMEOUT = 10
MOVE_HEAD_TO_ORIGIN_TIMEOUT = 5
MOVE_TORSO_TO_ORIGIN_TIMEOUT = 30
MOVE_THINKING_TIMEOUT = 5
MOVE_SAD_TIMEOUT = 5


class GestureNode:
    def __init__(self):
        self._simulation = rospy.get_param('~simulation')

        self._gesture_lock = threading.Lock()
        self._movement_commands = MovementCommands(self._simulation, namespace='gesture')

        self._done_pub = rospy.Publisher('gesture/done', Done, queue_size=5)
        self._gesture_sub = rospy.Subscriber('gesture/name', GestureName, self._on_gesture_cb)

    def _on_gesture_cb(self, msg):
        with self._gesture_lock:
            if self._movement_commands.is_filtering_all_messages:
                return

            try:
                ok = self._execute_gesture(msg.name)
            except TimeoutError:
                rospy.logerr(f'The {msg.name} gesture has timed out.')
                ok = False

            self._done_pub.publish(Done(id=msg.id, ok=ok))

    def _execute_gesture(self, name):
        if name == 'yes':
            self._movement_commands.move_yes(count=1, timeout=MOVE_YES_TIMEOUT)
        elif name == 'no':
            self._movement_commands.move_no(count=1, timeout=MOVE_NO_TIMEOUT)
        elif name == 'maybe':
            self._movement_commands.move_maybe(count=1, timeout=MOVE_MAYBE_TIMEOUT)
        elif name == 'origin_all':
            self._movement_commands.move_head_to_origin(should_wait=True, timeout=MOVE_HEAD_TO_ORIGIN_TIMEOUT)
            self._movement_commands.move_torso_to_origin(should_wait=True, timeout=MOVE_TORSO_TO_ORIGIN_TIMEOUT)
        elif name == 'origin_head':
            self._movement_commands.move_head_to_origin(should_wait=True, timeout=MOVE_HEAD_TO_ORIGIN_TIMEOUT)
        elif name == 'slow_origin_head':
            self._movement_commands.move_head_to_origin(should_wait=True, speed_rad_sec=0.5, timeout=MOVE_HEAD_TO_ORIGIN_TIMEOUT)
        elif name == 'origin_torso':
            self._movement_commands.move_torso_to_origin(should_wait=True, timeout=MOVE_TORSO_TO_ORIGIN_TIMEOUT)
        elif name == 'thinking':
            self._movement_commands.move_head_to_thinking(timeout=MOVE_THINKING_TIMEOUT)
        elif name == 'sad':
            self._movement_commands.move_head_to_sad(timeout=MOVE_SAD_TIMEOUT)
        else:
            rospy.logerr(f'Invalid gesture name ({name})')
            return False

        return True

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
