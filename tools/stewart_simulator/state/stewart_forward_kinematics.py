import math

import numpy as np
from numba import jit
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

from state import Shape

POSITION_BOUND_STEP = 0.001
EULER_ANGLE_BOUND_STEP = 0.01
EULER_ANGLE_SEQUENCE = 'XYZ'

BOUND_SCALE = 2


class StewartForwardKinematics:
    def __init__(self, configuration,
                 initial_top_anchors,
                 bottom_linear_actuator_anchors,
                 bottom_horn_orientation_angles,
                 inverse_kinematics):
        self._configuration = configuration
        self._initial_top_anchors = initial_top_anchors
        self._bottom_linear_actuator_anchors = bottom_linear_actuator_anchors
        self._bottom_horn_orientation_angles = bottom_horn_orientation_angles

        self._x0 = self._find_x0()
        self._set_bounds(inverse_kinematics)

    def _find_x0(self):
        position, orientation = self.calculate_pose(np.zeros(len(self._configuration.bottom_configuration.servos)),
                                                    x0=np.array([0.0, 0.0, 1, 0.0, 0.0, 0.0]),
                                                    bounds=([-np.inf, -np.inf, 0, -math.pi, -math.pi, -math.pi],
                                                            [np.inf, np.inf, np.inf, math.pi, math.pi, math.pi]))
        euler_angles = orientation.as_euler(EULER_ANGLE_SEQUENCE)
        return np.array([position[0], position[1], position[2], euler_angles[0], euler_angles[1], euler_angles[2]])

    def _set_bounds(self, inverse_kinematics):
        self._position_x_bounds = _calculate_bounds(inverse_kinematics, self._x0, 0, POSITION_BOUND_STEP)
        self._position_y_bounds = _calculate_bounds(inverse_kinematics, self._x0, 1, POSITION_BOUND_STEP)
        self._position_z_bounds = _calculate_bounds(inverse_kinematics, self._x0, 2, POSITION_BOUND_STEP)
        self._euler_angle_x_bounds = _calculate_bounds(inverse_kinematics, self._x0, 3, EULER_ANGLE_BOUND_STEP)
        self._euler_angle_y_bounds = _calculate_bounds(inverse_kinematics, self._x0, 4, EULER_ANGLE_BOUND_STEP)
        self._euler_angle_z_bounds = _calculate_bounds(inverse_kinematics, self._x0, 5, EULER_ANGLE_BOUND_STEP)

    def get_x0(self):
        return self._x0

    def get_bounds(self):
        return ([self._position_x_bounds[0], self._position_y_bounds[0], self._position_z_bounds[0],
                 self._euler_angle_x_bounds[0], self._euler_angle_y_bounds[0], self._euler_angle_z_bounds[0]],
                [self._position_x_bounds[1], self._position_y_bounds[1], self._position_z_bounds[1],
                 self._euler_angle_x_bounds[1], self._euler_angle_y_bounds[1], self._euler_angle_z_bounds[1]])

    def calculate_pose(self, servo_angles, bounds=None, x0=None):
        if bounds is None:
            bounds = self.get_bounds()
        if x0 is None:
            x0 = self._x0

        bottom_anchors = self._calculate_bottom_anchors(servo_angles)

        def func(x):
            errors = np.zeros(len(servo_angles))
            for i in range(len(errors)):
                errors[i] = _calculate_rod_length_error(bottom_anchors.x[i],
                                                        bottom_anchors.y[i],
                                                        bottom_anchors.z[i],
                                                        self._initial_top_anchors.x[i],
                                                        self._initial_top_anchors.y[i],
                                                        self._initial_top_anchors.z[i],
                                                        tx=x[0],
                                                        ty=x[1],
                                                        tz=x[2],
                                                        rx=x[3],
                                                        ry=x[4],
                                                        rz=x[5],
                                                        d=self._configuration.rod_length)
            return errors

        def jac(x):
            jacobian = np.zeros((len(servo_angles), len(x)))
            for i in range(len(servo_angles)):
                jacobian[i, :] = _calculate_rod_length_error_grad(bottom_anchors.x[i],
                                                                      bottom_anchors.y[i],
                                                                      bottom_anchors.z[i],
                                                                      self._initial_top_anchors.x[i],
                                                                      self._initial_top_anchors.y[i],
                                                                      self._initial_top_anchors.z[i],
                                                                      tx=x[0],
                                                                      ty=x[1],
                                                                      tz=x[2],
                                                                      rx=x[3],
                                                                      ry=x[4],
                                                                      rz=x[5])

            return jacobian

        result = least_squares(func, x0=x0, jac=jac, bounds=bounds, method='trf')
        return _x_to_position(result.x), _x_to_orientation(result.x)

    def _calculate_bottom_anchors(self, servo_angles):
        h = self._configuration.bottom_configuration.horn_length

        x = np.zeros(len(servo_angles))
        y = np.zeros(len(servo_angles))
        z = np.zeros(len(servo_angles))

        for i in range(len(servo_angles)):
            servo_angle = servo_angles[i]
            if self._configuration.bottom_configuration.servos[i].is_horn_orientation_reversed:
                servo_angle = -servo_angle

            x[i] = self._bottom_linear_actuator_anchors.x[i] + \
                   h * math.cos(servo_angle) * math.cos(self._bottom_horn_orientation_angles[i])
            y[i] = self._bottom_linear_actuator_anchors.y[i] + \
                   h * math.cos(servo_angle) * math.sin(self._bottom_horn_orientation_angles[i])
            z[i] = self._bottom_linear_actuator_anchors.z[i] + h * math.sin(servo_angle)

        return Shape(x, y, z)


def _calculate_bounds(inverse_kinematics, x0, index, step):
    min_step = -step
    max_step = step
    if max_step < min_step:
        min_step, max_step = max_step, min_step

    min_bound = _calculate_one_bound(inverse_kinematics, x0.copy(), index, min_step)
    max_bound = _calculate_one_bound(inverse_kinematics, x0.copy(), index, max_step)

    return min_bound, max_bound


def _calculate_one_bound(inverse_kinematics, x, index, step):
    offset = x[index]
    while True:
        try:
            x[index] += step
            inverse_kinematics.calculate_servo_angles(_x_to_position(x), _x_to_orientation(x))
        except ValueError:
            return (x[index] - offset) * BOUND_SCALE + offset


def _x_to_position(x):
    return x[:3]


def _x_to_orientation(x):
    return Rotation.from_euler(EULER_ANGLE_SEQUENCE, x[3:])


@jit(nopython=True)
def _calculate_rod_length_error(bax, bay, baz, itax, itay, itaz, tx, ty, tz, rx, ry, rz, d):
    """
    See forward_kinematics_jacobian.py
    """
    sin_rx = math.sin(rx)
    cos_rx = math.cos(rx)
    sin_ry = math.sin(ry)
    cos_ry = math.cos(ry)
    sin_rz = math.sin(rz)
    cos_rz = math.cos(rz)

    cos_ry_cos_rz = cos_ry * cos_rz
    sin_rz_cos_ry = sin_rz * cos_ry
    sin_rx_sin_ry_cos_rz = sin_rx * sin_ry * cos_rz
    sin_rz_cos_rx = sin_rz * cos_rx
    sin_rx_sin_ry_sin_rz = sin_rx * sin_ry * sin_rz
    cos_rx_cos_rz = cos_rx * cos_rz
    sin_rx_cos_ry = sin_rx * cos_ry
    sin_rx_sin_rz = sin_rx * sin_rz
    sin_ry_cos_rx_cos_rz = sin_ry * cos_rx_cos_rz
    sin_rx_cos_rz = sin_rx * cos_rz
    sin_ry_sin_rz_cos_rx = sin_ry * sin_rz_cos_rx
    cos_rx_cos_ry = cos_rx * cos_ry

    error = -d + math.sqrt((-bax + itax * cos_ry_cos_rz - itay * sin_rz_cos_ry + itaz * sin_ry + tx) ** 2 +
                           (-bay + itax * (sin_rx_sin_ry_cos_rz + sin_rz_cos_rx) - itay * (
                                       sin_rx_sin_ry_sin_rz - cos_rx_cos_rz) - itaz * sin_rx_cos_ry + ty) ** 2 +
                           (-baz + itax * (sin_rx_sin_rz - sin_ry_cos_rx_cos_rz) + itay * (
                                       sin_rx_cos_rz + sin_ry_sin_rz_cos_rx) + itaz * cos_rx_cos_ry + tz) ** 2)

    return error


@jit(nopython=True)
def _calculate_rod_length_error_grad(bax, bay, baz, itax, itay, itaz, tx, ty, tz, rx, ry, rz):
    """
    See forward_kinematics_jacobian.py
    """
    sin_rx = math.sin(rx)
    cos_rx = math.cos(rx)
    sin_ry = math.sin(ry)
    cos_ry = math.cos(ry)
    sin_rz = math.sin(rz)
    cos_rz = math.cos(rz)

    cos_ry_cos_rz = cos_ry * cos_rz
    sin_rz_cos_ry = sin_rz * cos_ry
    sin_rx_sin_ry_cos_rz = sin_rx * sin_ry * cos_rz
    sin_rz_cos_rx = sin_rz * cos_rx
    sin_rx_sin_ry_sin_rz = sin_rx * sin_ry * sin_rz
    cos_rx_cos_rz = cos_rx * cos_rz
    sin_rx_cos_ry = sin_rx * cos_ry
    sin_rx_sin_rz = sin_rx * sin_rz
    sin_rx_cos_ry_cos_rz = sin_rx * cos_ry_cos_rz
    sin_ry_cos_rx_cos_rz = sin_ry * cos_rx_cos_rz
    sin_rx_cos_rz = sin_rx * cos_rz
    sin_rx_sin_rz_cos_ry = sin_rx * sin_rz_cos_ry
    sin_ry_sin_rz_cos_rx = sin_ry * sin_rz_cos_rx
    cos_rx_cos_ry = cos_rx * cos_ry
    sin_rz_cos_rx_cos_ry = sin_rz_cos_rx * cos_ry
    cos_rx_cos_ry_cos_rz = cos_rx_cos_ry * cos_rz
    sin_ry_cos_rz = sin_ry * cos_rz
    sin_ry_sin_rz = sin_ry * sin_rz
    sin_ry_cos_rx = sin_ry * cos_rx
    sin_rx_sin_ry = sin_rx * sin_ry

    denominator = math.sqrt((-bax + itax * cos_ry_cos_rz - itay * sin_rz_cos_ry + itaz * sin_ry + tx) ** 2 +
                            (-bay + itax * (sin_rx_sin_ry_cos_rz + sin_rz_cos_rx) - itay * (
                                        sin_rx_sin_ry_sin_rz - cos_rx_cos_rz) - itaz * sin_rx_cos_ry + ty) ** 2 +
                            (-baz + itax * (sin_rx_sin_rz - sin_ry_cos_rx_cos_rz) + itay * (
                                        sin_rx_cos_rz + sin_ry_sin_rz_cos_rx) + itaz * cos_rx_cos_ry + tz) ** 2)

    dtx = (-bax + itax * cos_ry_cos_rz - itay * sin_rz_cos_ry + itaz * sin_ry + tx) / denominator
    dty = (-bay + itax * (sin_rx_sin_ry_cos_rz + sin_rz_cos_rx) - itay * (
                sin_rx_sin_ry_sin_rz - cos_rx_cos_rz) - itaz * sin_rx_cos_ry + ty) / denominator
    dtz = (-baz + itax * (sin_rx_sin_rz - sin_ry_cos_rx_cos_rz) + itay * (
                sin_rx_cos_rz + sin_ry_sin_rz_cos_rx) + itaz * cos_rx_cos_ry + tz) / denominator

    drx = (-(itax * (sin_rx_sin_rz - sin_ry_cos_rx_cos_rz) + itay * (
                sin_rx_cos_rz + sin_ry_sin_rz_cos_rx) + itaz * cos_rx_cos_ry) * (
                      -bay + itax * (sin_rx_sin_ry_cos_rz + sin_rz_cos_rx) + itay * (
                          -sin_rx_sin_ry_sin_rz + cos_rx_cos_rz) - itaz * sin_rx_cos_ry + ty) + (
                      itax * (sin_rx_sin_ry_cos_rz + sin_rz_cos_rx) + itay * (
                          -sin_rx_sin_ry_sin_rz + cos_rx_cos_rz) - itaz * sin_rx_cos_ry) * (
                      -baz + itax * (sin_rx_sin_rz - sin_ry_cos_rx_cos_rz) + itay * (
                          sin_rx_cos_rz + sin_ry_sin_rz_cos_rx) + itaz * cos_rx_cos_ry + tz)) / denominator
    dry = (bax * itax * sin_ry_cos_rz - bax * itay * sin_ry_sin_rz - bax * itaz * cos_ry -
           bay * itax * sin_rx_cos_ry_cos_rz + bay * itay * sin_rx_sin_rz_cos_ry - bay * itaz * sin_rx_sin_ry +
           baz * itax * cos_rx_cos_ry_cos_rz - baz * itay * sin_rz_cos_rx_cos_ry + baz * itaz * sin_ry_cos_rx -
           itax * tx * sin_ry_cos_rz + itax * ty * sin_rx_cos_ry_cos_rz - itax * tz * cos_rx_cos_ry_cos_rz +
           itay * tx * sin_ry_sin_rz - itay * ty * sin_rx_sin_rz_cos_ry + itay * tz * sin_rz_cos_rx_cos_ry +
           itaz * tx * cos_ry + itaz * ty * sin_rx_sin_ry - itaz * tz * sin_ry_cos_rx) / denominator
    drz = (bax * itax * sin_rz_cos_ry + bax * itay * cos_ry_cos_rz + bay * itax * sin_rx_sin_ry_sin_rz -
           bay * itax * cos_rx_cos_rz + bay * itay * sin_rx_sin_ry_cos_rz + bay * itay * sin_rz_cos_rx -
           baz * itax * sin_rx_cos_rz - baz * itax * sin_ry_sin_rz_cos_rx + baz * itay * sin_rx_sin_rz -
           baz * itay * sin_ry_cos_rx_cos_rz - itax * tx * sin_rz_cos_ry - itax * ty * sin_rx_sin_ry_sin_rz +
           itax * ty * cos_rx_cos_rz + itax * tz * sin_rx_cos_rz + itax * tz * sin_ry_sin_rz_cos_rx -
           itay * tx * cos_ry_cos_rz - itay * ty * sin_rx_sin_ry_cos_rz - itay * ty * sin_rz_cos_rx -
           itay * tz * sin_rx_sin_rz + itay * tz * sin_ry_cos_rx * cos_rz) / denominator

    return np.array([dtx, dty, dtz, drx, dry, drz])
