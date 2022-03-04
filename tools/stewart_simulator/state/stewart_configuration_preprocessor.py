import math

import numpy as np

from state import Shape


class StewartConfigurationPreprocessor:
    def __init__(self, configuration):
        self.bottom_linear_actuator_anchors = self._generate_bottom_linear_actuator_anchors(
            configuration.bottom_configuration.servos,
            configuration.bottom_configuration.horn_offset,
            configuration.bottom_configuration.ball_joint_offset)
        self.bottom_horn_orientation_angles = [self._generate_bottom_horn_orientation_angle(servo)
                                               for servo in configuration.bottom_configuration.servos]

    def _generate_bottom_linear_actuator_anchors(self, servos_configuration, horn_offset, ball_joint_offset):
        offset = horn_offset + ball_joint_offset
        x = [servo.x + servo.orientation_x * offset for servo in servos_configuration]
        y = [servo.y + servo.orientation_y * offset for servo in servos_configuration]
        z = np.zeros(len(x))

        return Shape(x, y, z)

    def _generate_bottom_horn_orientation_angle(self, servo_configuration):
        horn_orientation = np.cross([servo_configuration.orientation_x, servo_configuration.orientation_y, 0],
                                    [0, 0, 1 if servo_configuration.is_horn_orientation_reversed else -1])
        horn_orientation *= 1.0 / np.linalg.norm(horn_orientation)

        return math.atan2(horn_orientation[1], horn_orientation[0])
