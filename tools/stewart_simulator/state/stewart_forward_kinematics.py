import numpy as np
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
                                                    x0=np.zeros(6),
                                                    bounds=(-np.inf, np.inf))
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
            top_anchors = self._initial_top_anchors.rotate(_x_to_orientation(x)).translate(_x_to_position(x))

            dx = bottom_anchors.x - top_anchors.x
            dy = bottom_anchors.y - top_anchors.y
            dz = bottom_anchors.z - top_anchors.z
            rod_lengths = np.sqrt(dx**2 + dy**2 + dz**2)

            return rod_lengths - self._configuration.rod_length

        result = least_squares(func, x0=x0, bounds=bounds, method='trf')
        return _x_to_position(result.x), _x_to_orientation(result.x)

    def _calculate_bottom_anchors(self, servo_angles):
        h = self._configuration.bottom_configuration.horn_length

        x = np.zeros(len(servo_angles))
        y = np.zeros(len(servo_angles))
        z = np.zeros(len(servo_angles))

        for i in range(len(servo_angles)):
            servo_angle = servo_angles[i]
            if  self._configuration.bottom_configuration.servos[i].is_horn_orientation_reversed:
                servo_angle = -servo_angle

            x[i] = self._bottom_linear_actuator_anchors.x[i] + \
                h * np.cos(servo_angle) * np.cos(self._bottom_horn_orientation_angles[i])
            y[i] = self._bottom_linear_actuator_anchors.y[i] + \
                h * np.cos(servo_angle) * np.sin(self._bottom_horn_orientation_angles[i])
            z[i] = self._bottom_linear_actuator_anchors.z[i] + h * np.sin(servo_angle)

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
