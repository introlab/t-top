import json
import math

import rospy

from daemon_ros_client.msg import LedColors
from led_animations.msg import Animation, Done

import hbba_lite

from led_animations.lib_led_animations import LedAnimation


NONE_LED_COLORS = LedColors()
for led_color in NONE_LED_COLORS.colors:
    led_color.red = 0
    led_color.green = 0
    led_color.blue = 0


class LedAnimationsNode:
    def __init__(self):
        self._period_s = rospy.get_param('~period_s', 0.0333)

        self._timer = None
        self._timer_msg_id = -1
        self._timer_start_time_s = 0.0
        self._timer_duration_s = 0.0
        self._timer_animation = None

        self._led_colors_pub = hbba_lite.OnOffHbbaPublisher('led_animations/set_led_colors', LedColors, queue_size=1,
                                                            state_service_name='set_led_colors/filter_state')
        self._led_colors_pub.on_filter_state_changing(self._hbba_filter_state_cb)

        self._done_pub = rospy.Publisher('led_animations/done', Done, queue_size=5)
        self._led_emotion_sub = rospy.Subscriber('led_animations/animation', Animation, self._animation_cb, queue_size=1)

    def _hbba_filter_state_cb(self, publish_forced, previous_is_filtering_all_messages, new_is_filtering_all_messages):
        if not previous_is_filtering_all_messages and new_is_filtering_all_messages:
            self._stop_timer()
            publish_forced(NONE_LED_COLORS)

    def _animation_cb(self, msg):
        if self._led_colors_pub.is_filtering_all_messages:
            return

        try:
            animation = LedAnimation.from_name(msg.name, self._period_s, msg.speed, msg.colors)
        except Exception as e:
            rospy.logerr(f'Unable to instantiate the LED animation ({e})')
            self._done_pub.publish(Done(id=msg.id, ok=False))

        self._start_timer(msg.id, animation, msg.duration_s)

    def _stop_timer(self):
        if self._timer is not None:
            self._timer.shutdown()
            self._timer = None
            self._timer_msg_id = -1
            self._timer_start_time_s = 0.0
            self._timer_duration_s = 0.0
            self._timer_animation = None

    def _start_timer(self, id, animation, duration_s):
        self._stop_timer()
        self._timer_msg_id = id
        self._timer_start_time_s = rospy.get_time()
        self._timer_duration_s = duration_s
        self._timer_animation = animation
        self._timer = rospy.Timer(rospy.Duration(self._period_s), self._timer_cb)

    def _timer_cb(self, timer_event):
        if rospy.get_time() - self._timer_start_time_s > self._timer_duration_s:
            self._led_colors_pub.publish(NONE_LED_COLORS)
            self._done_pub.publish(Done(id=self._timer_msg_id, ok=True))
            self._stop_timer()
        else:
            self._led_colors_pub.publish(self._timer_animation.update())

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('led_animations_node')
    led_animations_node = LedAnimationsNode()
    led_animations_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
