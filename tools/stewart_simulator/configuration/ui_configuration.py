import math


class UiConfiguration:
    def __init__(self, json_configuration):
        self.relative_position_range_min = json_configuration['relativePositionRange']['min']
        self.relative_position_range_max = json_configuration['relativePositionRange']['max']

        self.orientation_angle_range_min = math.radians(json_configuration['orientationAngleRange']['min'])
        self.orientation_angle_range_max = math.radians(json_configuration['orientationAngleRange']['max'])

        self.position_step = json_configuration['positionStep']
        self.position_decimals = json_configuration['positionDecimals']
