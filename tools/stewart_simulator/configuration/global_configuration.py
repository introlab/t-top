import json
import math

from configuration.bottom_configuration import BottomConfiguration
from configuration.top_configuration import TopConfiguration
from configuration.ui_configuration import UiConfiguration


class GlobalConfiguration:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            json_configuration = json.load(f)
            yaw_rotation = math.radians(float(json_configuration['yawRotation']))

            self.top_configuration = TopConfiguration(json_configuration['top'], yaw_rotation)
            self.bottom_configuration = BottomConfiguration(json_configuration['bottom'], yaw_rotation)

            self.rod_length = float(json_configuration['rodLength'])
            self.servo_angle_min = math.radians(float(json_configuration['servoAngleRange']['min']))
            self.servo_angle_max = math.radians(float(json_configuration['servoAngleRange']['max']))

            self.ui = UiConfiguration(json_configuration['ui'])

            self._verify_anchor_servo_alignment()

    def _verify_anchor_servo_alignment(self):
        distances = self._get_anchor_servo_distances()
        min_distances = self._get_anchor_servo_min_distances()

        if any(not math.isclose(distances[i], min_distances[i]) for i in range(len(self.top_configuration.anchors))):
            raise ValueError('The anchors and the servos are not in the right order.')

    def _get_anchor_servo_distances(self):
        distances = []

        for i in range(len(self.top_configuration.anchors)):
            distances.append(self._get_anchor_servo_distance(self.top_configuration.anchors[i],
                                                             self.bottom_configuration.servos[i]))

        return distances

    def _get_anchor_servo_min_distances(self):
        min_distances = []

        for i in range(len(self.top_configuration.anchors)):
            min_distances.append(math.inf)
            for j in range(len(self.bottom_configuration.servos)):
                distance = self._get_anchor_servo_distance(self.top_configuration.anchors[i],
                                                           self.bottom_configuration.servos[j])
                min_distances[i] = min(min_distances[i], distance)

        return min_distances

    def _get_anchor_servo_distance(self, anchor, servo):
        return math.sqrt((anchor.x - servo.x) ** 2 + (anchor.y - servo.y) ** 2)
