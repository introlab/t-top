import json
import math

import rospy

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


class LedEmotionsNode:
    def __init__(self):
        self._period_s = rospy.get_param('~period_s', 0.0333)
        with open(rospy.get_param('~led_patterns_file'), 'r') as f:
            self._led_patterns_by_emotion_name = self._load_led_patterns(json.load(f))

        self._timer = None
        self._timer_time_s = 0
        self._timer_pattern = None

        self._led_colors_pub = hbba_lite.OnOffHbbaPublisher('led_emotions/set_led_colors', LedColors, queue_size=1,
                                                            state_service_name='set_led_colors/filter_state')
        self._led_colors_pub.on_filter_state_changing(self._hbba_filter_state_cb)

        self._led_emotion_sub = rospy.Subscriber('led_emotions/name', String, self._emotion_cb, queue_size=1)


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
            rospy.logerr(f'Invalid emotion name ({msg.data})')
        self._start_timer(self._led_patterns_by_emotion_name[msg.data])

    def _stop_timer(self):
        if self._timer is not None:
            self._timer.shutdown()
            self._timer = None

    def _start_timer(self, pattern):
        self._stop_timer()
        self._timer_time_s = 0
        self._timer_pattern = pattern
        self._timer = rospy.Timer(rospy.Duration(self._period_s), self._timer_cb)

    def _timer_cb(self, timer_event):
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
        rospy.spin()


def main():
    rospy.init_node('led_emotions_node')
    led_emotions_node = LedEmotionsNode()
    led_emotions_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
