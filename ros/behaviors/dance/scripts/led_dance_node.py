import json
import random

import rospy

from std_msgs.msg import Bool
from daemon_ros_client.msg import LedColors

import hbba_lite


class LedDanceNode:
    def __init__(self):
        with open(rospy.get_param('~led_colors_file'), 'r') as f:
            self._none_led_colors, self._dance_led_colors = self._load_led_colors(json.load(f))

        self._led_colors_pub = hbba_lite.OnOffHbbaPublisher('dance/set_led_colors', LedColors, queue_size=1,
                                                            state_service_name='set_led_colors/filter_state')
        self._led_colors_pub.on_filter_state_changing(self._hbba_filter_state_cb)

        self._beat_sub = rospy.Subscriber('beat', Bool, self._beat_cb, queue_size=1)

    def _load_led_colors(self, json_dict):
        none_led_colors = LedColors()
        for led_color in none_led_colors.colors:
            led_color.red = 0
            led_color.green = 0
            led_color.blue = 0

        dance_led_colors = []

        for name, values in json_dict.items():
            led_colors = LedColors()
            for led_color, json_color in zip(led_colors.colors, values):
                led_color.red = json_color['red']
                led_color.green = json_color['green']
                led_color.blue = json_color['blue']

            dance_led_colors.append(led_colors)

        return none_led_colors, dance_led_colors

    def _hbba_filter_state_cb(self, publish_forced, previous_is_filtering_all_messages, new_is_filtering_all_messages):
        if not previous_is_filtering_all_messages and new_is_filtering_all_messages:
            publish_forced(self._none_led_colors)

    def _beat_cb(self, msg):
        if msg.data and not self._led_colors_pub.is_filtering_all_messages:
            self._led_colors_pub.publish(random.choice(self._dance_led_colors))

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('led_dance_node')
    led_dance_node = LedDanceNode()
    led_dance_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
