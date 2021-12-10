import math

from configuration import get_rotated_x, get_rotated_y
from configuration.shape import parse_shape


class BottomConfiguration:
    def __init__(self, json_configuration, yaw_rotation):
        self.horn_length = float(json_configuration['hornLength'])
        self.horn_offset = float(json_configuration['hornOffset'])
        self.ball_joint_offset = float(json_configuration['ballJointOffset'])

        self.shapes = [parse_shape(shape_configuration, yaw_rotation) for shape_configuration in
                       json_configuration['shapes']]
        self.servos = [Servo(servo_configuration, yaw_rotation) for servo_configuration in json_configuration['servos']]

        if len(self.servos) != 6:
            raise ValueError('The bottom configuration must have 6 servos.')

        radius = self.servos[0].get_radius()
        if any(not math.isclose(radius, anchor.get_radius()) for anchor in self.servos):
            raise ValueError('All servos must have the same radius.')


class Servo:
    def __init__(self, json_configuration, yaw_rotation):
        x = float(json_configuration['x'])
        y = float(json_configuration['y'])
        orientation_x = float(json_configuration['orientation']['x'])
        orientation_y = float(json_configuration['orientation']['y'])

        self.x = get_rotated_x(x, y, yaw_rotation)
        self.y = get_rotated_y(x, y, yaw_rotation)

        self.orientation_x = get_rotated_x(orientation_x, orientation_y, yaw_rotation)
        self.orientation_y = get_rotated_y(orientation_x, orientation_y, yaw_rotation)

        orientation_length = math.sqrt(self.orientation_x ** 2 + self.orientation_y ** 2)
        self.orientation_x /= orientation_length
        self.orientation_y /= orientation_length

        self.is_horn_orientation_reversed = json_configuration['isHornOrientationReversed']

    def get_radius(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)
