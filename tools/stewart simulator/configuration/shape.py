from enum import Enum

from configuration import get_rotated_x, get_rotated_y


def parse_shape(json_configuration, yaw_rotation):
    if json_configuration['type'] == 'circle':
        return Circle(json_configuration['center']['x'], json_configuration['center']['y'],
                      json_configuration['radius'], yaw_rotation)
    elif json_configuration['type'] == 'polygon':
        return Polygon([p['x'] for p in json_configuration['points']], [p['y'] for p in json_configuration['points']],
                       yaw_rotation)
    else:
        raise ValueError('Invalid shape type (' + json_configuration['type'] + ')')


class ShapeType(Enum):
    CIRCLE = 1
    POLYGON = 2


class Shape:
    def __init__(self, type):
        self.type = type


class Circle(Shape):
    def __init__(self, x, y, radius, yaw_rotation):
        super().__init__(ShapeType.CIRCLE)
        self.x = get_rotated_x(x, y, yaw_rotation)
        self.y = get_rotated_y(x, y, yaw_rotation)
        self.radius = radius


class Polygon(Shape):
    def __init__(self, x, y, yaw_rotation):
        super().__init__(ShapeType.POLYGON)
        self.x = x
        self.y = y

        for i in range(len(x)):
            x = self.x[i]
            y = self.y[i]
            self.x[i] = get_rotated_x(x, y, yaw_rotation)
            self.y[i] = get_rotated_y(x, y, yaw_rotation)
