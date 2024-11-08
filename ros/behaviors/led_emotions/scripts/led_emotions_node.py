#!/usr/bin/env python3

import json
import math

import rclpy
import rclpy.node

from std_msgs.msg import String
from daemon_ros_client.msg import LedColors

import hbba_lite


NONE_LED_COLORS = LedColors()
for led_color in NONE_LED_COLORS.colors:
    led_color.red = 0
    led_color.green = 0
    led_color.blue = 0


class LedPattern:
    def __init__(self, red, green, blue, period_s, ratio):
        self.red = red
        self.green = green
        self.blue = blue
        self.period_s = period_s
        self.ratio = ratio


class LedEmotionsNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('led_emotions_node')

        self._period_s = self.declare_parameter('period_s', 0.0333).get_parameter_value().double_value
        led_patterns_file = self.declare_parameter('led_patterns_file', '').get_parameter_value().string_value

        with open(led_patterns_file, 'r') as f:
            self._led_patterns_by_emotion_name = self._load_led_patterns(json.load(f))

        self._timer = None
        self._timer_time_s = 0
        self._timer_pattern = None

        self._led_colors_pub = hbba_lite.OnOffHbbaPublisher(self, LedColors, 'led_emotions/set_led_colors', 1,
                                                            state_service_name='set_led_colors/filter_state')
        self._led_colors_pub.on_filter_state_changing(self._hbba_filter_state_cb)

        self._led_emotion_sub = self.create_subscription(String, 'led_emotions/name', self._emotion_cb, 1)


    def _load_led_patterns(self, json_dict):
        led_patterns_by_emotion_name = {}
        for name, values in json_dict.items():
            led_patterns_by_emotion_name[name] = LedPattern(
                values["color"]["red"], values["color"]["green"], values["color"]["blue"],
                values["period_s"], values["ratio"]
            )

        return led_patterns_by_emotion_name

    def _hbba_filter_state_cb(self, publish_forced, previous_is_filtering_all_messages, new_is_filtering_all_messages):
        if not previous_is_filtering_all_messages and new_is_filtering_all_messages:
            self._stop_timer()
            publish_forced(NONE_LED_COLORS)

    def _emotion_cb(self, msg):
        if self._led_colors_pub.is_filtering_all_messages:
            return

        if msg.data not in self._led_patterns_by_emotion_name:
            self.get_logger().error(f'Invalid emotion name ({msg.data})')
        else:
            self._start_timer(self._led_patterns_by_emotion_name[msg.data])

    def _stop_timer(self):
        if self._timer is not None:
            self.destroy_timer(self._timer)
            self._timer = None

    def _start_timer(self, pattern):
        self._stop_timer()
        self._timer_time_s = 0
        self._timer_pattern = pattern
        self._timer = self.create_timer(self._period_s, self._timer_cb)

    def _timer_cb(self):
        self._timer_time_s += self._period_s
        if self._timer_time_s > self._timer_pattern.period_s:
            self._timer_time_s -= self._timer_pattern.period_s

        brightness = self._get_brightness(self._timer_time_s, self._timer_pattern.period_s, self._timer_pattern.ratio)
        red = int(self._timer_pattern.red * brightness)
        green = int(self._timer_pattern.green * brightness)
        blue = int(self._timer_pattern.blue * brightness)

        msg = LedColors()
        for led_color in msg.colors:
            led_color.red = red
            led_color.green = green
            led_color.blue = blue

        self._led_colors_pub.publish(msg)

    def _get_brightness(self, t, T, ratio):
        if t < 0.0 or t > T:
            raise ValueError('t must be between 0 and T')
        elif 0 <= t and t <= (ratio * T / 2):
            return -0.5 * math.cos(2 * math.pi * t / (ratio * T)) + 0.5
        elif (ratio * T / 2) < t and t <= (T / 2):
            return 1
        elif (T / 2) < t and t <= (T / 2 * (1 + ratio)):
            return 0.5 * math.cos(2 * math.pi / (ratio * T) * (t - T / 2)) + 0.5
        else:
            return 0

    def run(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    led_emotions_node = LedEmotionsNode()

    try:
        led_emotions_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        led_emotions_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
