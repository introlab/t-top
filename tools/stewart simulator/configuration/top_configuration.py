import math

from configuration import get_rotated_x, get_rotated_y
from configuration.shape import parse_shape


class TopConfiguration:
    def __init__(self, json_configuration, yaw_rotation):
        self.ball_joint_offset = float(json_configuration['ballJointOffset'])

        self.shapes = [parse_shape(shape_configuration, yaw_rotation) for shape_configuration in
                       json_configuration['shapes']]
        self.anchors = [Anchor(anchor_configuration, yaw_rotation) for anchor_configuration in
                        json_configuration['anchors']]

        if len(self.anchors) != 6:
            raise ValueError('The top configuration must have 6 anchors.')

        radius = self.anchors[0].get_radius()
        if any(not math.isclose(radius, anchor.get_radius()) for anchor in self.anchors):
            raise ValueError('All anchors must have the same radius.')


class Anchor:
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

    def get_radius(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)
