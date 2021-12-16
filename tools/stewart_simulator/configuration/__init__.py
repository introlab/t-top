import math


def get_rotated_x(x, y, angle):
    return math.cos(angle) * x - math.sin(angle) * y


def get_rotated_y(x, y, angle):
    return math.sin(angle) * x + math.cos(angle) * y
