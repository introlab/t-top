#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from contextlib import contextmanager

from t_top import MovementCommands, HEAD_ZERO_Z
from opentera_webrtc_ros_msgs.srv import SetString, SetStringRequest, SetStringResponse


def twist_is_null(twist: Twist) -> bool:
    return twist.angular.z == 0 and twist.linear.x == 0


def twist_up_down_sign(twist: Twist) -> float:
    return 0 if twist.linear.x == 0 else 1 if twist.linear.x > 0 else -1


def twist_left_right_sign(twist: Twist) -> float:
    return 0 if twist.angular.z == 0 else 1 if twist.angular.z > 0 else -1


class TeleopAngles:
    def __init__(self, torso_angle: float = 0, head_angle: float = 0) -> None:
        self.torso_angle = torso_angle
        self.head_angle = head_angle


class TeleopState:
    def __init__(self) -> None:
        self.current_speed = TeleopAngles()
        self.is_moving = False


class TeleoperationNode:
    def __init__(self):
        self._simulation = rospy.get_param('~simulation', False)

        self._movement_commands = MovementCommands(self._simulation, namespace='teleoperation')
        self._state = TeleopState()

        self._twist_sub = rospy.Subscriber(
            'teleoperation/cmd_vel', Twist, self._on_twist_cb, queue_size=1)

        self._timer = rospy.Timer(rospy.Duration(
            self._movement_commands.sleep_time), self._on_timer_cb)

        self._origin_serv = rospy.Service(
            "teleop_do_action", SetString, self._do_action_cb)

    def _on_twist_cb(self, msg):
        if self._movement_commands.is_filtering_all_messages:
            self._state.is_moving = False
            return

        if twist_is_null(msg):
            self._stop_moving()
        elif not self._state.is_moving:
            self._start_moving(msg)
        else:
            self._move(msg)

    def _on_timer_cb(self, _):
        if self._state.is_moving:
            self._movement_commands.move_torso_speed(
                speed_rad_sec_torso=self._state.current_speed.torso_angle, should_sleep=False)

            self._movement_commands.move_head_speed(
                speeds_head=[
                    0,
                    0,
                    0,
                    0,
                    self._state.current_speed.head_angle,
                    0
                ],
                should_sleep=False,
                current_head_pose=[0, 0, HEAD_ZERO_Z, 0,
                                   self._movement_commands.current_head_pose[4], 0]
            )

    def _stop_moving(self) -> None:
        self._state.is_moving = False

    def _start_moving(self, msg: Twist) -> None:
        self._state.is_moving = True
        self._move(msg)

    def _move(self, msg: Twist) -> None:
        self._state.current_speed = TeleopAngles(
            torso_angle=msg.angular.z/0.15, head_angle=-1.1*msg.linear.x/0.15)

    def _do_action_cb(self, req: SetStringRequest) -> SetStringResponse:
        jump_table = {
            "goto_origin": self._goto_origin,
            "do_yes": self._do_yes,
            "do_no": self._do_no,
            "do_maybe": self._do_maybe,
            "goto_origin_head": self._goto_origin_head,
            "goto_origin_torso": self._goto_origin_torso,
        }

        if req.data in jump_table:
            if self._movement_commands.is_filtering_all_messages:
                return SetStringResponse(success=False, message="HBBA filter is active")

            jump_table[req.data]()
            return SetStringResponse(success=True, message="")

        else:
            return SetStringResponse(success=False, message=f"'{req.data}' not in {list(jump_table.keys())}")

    @contextmanager
    def _pause_moving(self):
        state = self._state.is_moving
        self._state.is_moving = False

        yield state

        self._state.is_moving = state

    def _goto_origin(self) -> None:
        with self._pause_moving():
            self._movement_commands.move_head_to_origin(False)
            self._movement_commands.move_torso_to_origin(False)

    def _do_yes(self) -> None:
        with self._pause_moving():
            self._movement_commands.move_yes(count=1)

    def _do_no(self) -> None:
        with self._pause_moving():
            self._movement_commands.move_no(count=1)

    def _do_maybe(self) -> None:
        with self._pause_moving():
            self._movement_commands.move_maybe(count=1)

    def _goto_origin_head(self) -> None:
        with self._pause_moving():
            self._movement_commands.move_head_to_origin(False)

    def _goto_origin_torso(self) -> None:
        with self._pause_moving():
            self._movement_commands.move_torso_to_origin(False)

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('teleoperation_node')
    teleop_node = TeleoperationNode()
    teleop_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
