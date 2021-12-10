import math

from state import Shape


class StewartInverseKinematics:
    def __init__(self, configuration, initial_top_anchors, bottom_linear_actuator_anchors,
                 bottom_horn_orientation_angles):
        self._configuration = configuration
        self._initial_top_anchors = initial_top_anchors
        self._bottom_linear_actuator_anchors = bottom_linear_actuator_anchors
        self._bottom_horn_orientation_angles = bottom_horn_orientation_angles

    def calculate_servo_angles(self, position, orientation):
        effective_lengths = self._get_effective_length_from_top_anchors(position, orientation)
        bottom_horn_orientation_angles = self._bottom_horn_orientation_angles
        servo_angles = []
        for i in range(len(effective_lengths.x)):
            is_reversed = self._configuration.bottom_configuration.servos[i].is_horn_orientation_reversed
            servo_angles.append(self._get_servo_angle(effective_lengths.x[i],
                                                      effective_lengths.y[i],
                                                      effective_lengths.z[i],
                                                      self._configuration.rod_length,
                                                      self._configuration.bottom_configuration.horn_length,
                                                      bottom_horn_orientation_angles[i],
                                                      is_reversed))

        self._verify_servo_angle(servo_angles)
        return servo_angles

    def _get_effective_length_from_top_anchors(self, position, orientation):
        top_anchors = self._initial_top_anchors.rotate(orientation).translate(position)
        x = top_anchors.x - self._bottom_linear_actuator_anchors.x
        y = top_anchors.y - self._bottom_linear_actuator_anchors.y
        z = top_anchors.z - self._bottom_linear_actuator_anchors.z

        return Shape(x, y, z)

    def _get_servo_angle(self, lx, ly, lz,
                         rod_length, horn_length, horn_orientation_angle,
                         is_reversed):
        ek = 2 * horn_length * lz
        fk = 2 * horn_length * (math.cos(horn_orientation_angle) * lx + math.sin(horn_orientation_angle) * ly)
        gk = lx ** 2 + ly ** 2 + lz ** 2 - rod_length ** 2 + horn_length ** 2

        servo_angle = math.asin(gk / math.sqrt(ek ** 2 + fk ** 2)) - math.atan2(fk, ek)

        if is_reversed:
            servo_angle = -servo_angle

        return servo_angle

    def _verify_servo_angle(self, servo_angles):
        min = self._configuration.servo_angle_min
        max = self._configuration.servo_angle_max
        if any(servo_angle < min or servo_angle > max for servo_angle in servo_angles):
            raise ValueError('Invalid servo angles')
