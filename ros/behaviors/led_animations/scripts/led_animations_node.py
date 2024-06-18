#!/usr/bin/env python3

import rclpy
import rclpy.node
from rclpy.duration import Duration

from daemon_ros_client.msg import LedColors
from behavior_msgs.msg import LedAnimation as LedAnimationMsg, Done

import hbba_lite

from led_animations.lib_led_animations import LedAnimation


NONE_LED_COLORS = LedColors()
for led_color in NONE_LED_COLORS.colors:
    led_color.red = 0
    led_color.green = 0
    led_color.blue = 0


class LedAnimationsNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('led_animations_node')

        self._period_s = self.declare_parameter('period_s', 0.0333).get_parameter_value().double_value

        self._timer = None
        self._timer_msg_id = -1
        self._timer_start_time = None
        self._timer_duration = None
        self._timer_animation = None

        self._led_colors_pub = hbba_lite.OnOffHbbaPublisher(self, LedColors, 'led_animations/set_led_colors', 1,
                                                            state_service_name='set_led_colors/filter_state')
        self._led_colors_pub.on_filter_state_changing(self._hbba_filter_state_cb)

        self._done_pub = self.create_publisher(Done, 'led_animations/done', 5)
        self._led_emotion_sub = self.create_subscription(LedAnimationMsg, 'led_animations/animation', self._animation_cb, 1)

    def _hbba_filter_state_cb(self, publish_forced, previous_is_filtering_all_messages, new_is_filtering_all_messages):
        if not previous_is_filtering_all_messages and new_is_filtering_all_messages:
            self._stop_timer()
            publish_forced(NONE_LED_COLORS)

    def _animation_cb(self, msg):
        if self._led_colors_pub.is_filtering_all_messages:
            return

        try:
            animation = LedAnimation.from_name(msg.name, self._period_s, msg.speed, msg.colors)
            self._start_timer(msg.id, animation, msg.duration_s)
        except Exception as e:
            self.get_logger().error(f'Unable to instantiate the LED animation ({e})')
            self._done_pub.publish(Done(id=msg.id, ok=False))


    def _stop_timer(self):
        if self._timer is not None:
            self.destroy_timer(self._timer)
            self._timer = None
            self._timer_msg_id = -1
            self._timer_start_time = None
            self._timer_duration = None
            self._timer_animation = None

    def _start_timer(self, id, animation, duration_s):
        self._stop_timer()
        self._timer_msg_id = id
        self._timer_start_time = self.get_clock().now()
        self._timer_duration = Duration(seconds=duration_s)
        self._timer_animation = animation
        self._timer = self.create_timer(self._period_s, self._timer_cb)

    def _timer_cb(self):
        if self.get_clock().now() - self._timer_start_time > self._timer_duration:
            self._led_colors_pub.publish(NONE_LED_COLORS)
            self._done_pub.publish(Done(id=self._timer_msg_id, ok=True))
            self._stop_timer()
        else:
            self._led_colors_pub.publish(self._timer_animation.update())

    def run(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    led_animations_node = LedAnimationsNode()

    try:
        led_animations_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        led_animations_node.destroy_node()
        
    rclpy.shutdown()


if __name__ == '__main__':
    main()
