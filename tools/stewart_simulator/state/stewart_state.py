import math

import numpy as np
from scipy.spatial.transform import Rotation

from configuration.shape import ShapeType
from state import Shape
from state.stewart_configuration_preprocessor import StewartConfigurationPreprocessor
from state.stewart_forward_kinematics import StewartForwardKinematics
from state.stewart_inverse_kinematics import StewartInverseKinematics

CIRCLE_ANGLE_RESOLUTION = 2 * math.pi / 360


class StewartState:
    def __init__(self, configuration):
        self._configuration = configuration
        self._configuration_preprocessor = StewartConfigurationPreprocessor(configuration)

        self.top_state = TopState(configuration.top_configuration)
        self.bottom_state = BottomState(configuration.bottom_configuration)

        self._inverse_kinematics = \
            StewartInverseKinematics(configuration,
                                     self.top_state.get_initial_anchors(),
                                     self._configuration_preprocessor.bottom_linear_actuator_anchors,
                                     self._configuration_preprocessor.bottom_horn_orientation_angles)
        self._forward_kinematics = \
            StewartForwardKinematics(configuration,
                                     self.top_state.get_initial_anchors(),
                                     self._configuration_preprocessor.bottom_linear_actuator_anchors,
                                     self._configuration_preprocessor.bottom_horn_orientation_angles,
                                     self._inverse_kinematics)

        zero_servo_angles = np.zeros(len(self._configuration.bottom_configuration.servos))
        initial_position, initial_orientation = self._forward_kinematics.calculate_pose(zero_servo_angles)
        self.top_state._set_initial_position(initial_position)
        self.top_state._set_initial_orientation(initial_orientation)
        self.set_servo_angles(zero_servo_angles)

    def set_top_pose(self, position, orientation):
        old_position = self.top_state.get_position()
        old_orientation = self.top_state.get_orientation()

        try:
            self.top_state._set_position(position)
            self.top_state._set_orientation(orientation)

            servo_angles = self._inverse_kinematics.calculate_servo_angles(position, orientation)
            self.bottom_state._set_servo_angles(servo_angles)
        except Exception:
            self.top_state._set_position(old_position)
            self.top_state._set_orientation(old_orientation)
            raise

    def set_servo_angles(self, servo_angles):
        position, orientation = self._forward_kinematics.calculate_pose(servo_angles)

        self.top_state._set_position(position)
        self.top_state._set_orientation(orientation)
        self.bottom_state._set_servo_angles(servo_angles)

    def get_ball_joint_angles(self):
        rod_orientations = self._get_rod_orientations()
        top_anchor_orientations = self.top_state._get_transformed_anchor_orientations()
        bottom_anchor_orientations = self.bottom_state._get_transformed_anchor_orientations()

        top_ball_joint_angle = [self._get_ball_joint_angle(rod_orientations[i], top_anchor_orientations[i])
                                for i in range(len(rod_orientations))]
        bottom_ball_joint_angle = [self._get_ball_joint_angle(rod_orientations[i], bottom_anchor_orientations[i])
                                   for i in range(len(rod_orientations))]

        return top_ball_joint_angle, bottom_ball_joint_angle

    def _get_rod_orientations(self):
        rod_orientations = []

        top_anchors = self.top_state.get_transformed_anchors()
        bottom_anchors = self.bottom_state.get_transformed_anchors()

        for i in range(len(top_anchors.x)):
            rod_orientation = np.array([bottom_anchors.x[i] - top_anchors.x[i],
                                        bottom_anchors.y[i] - top_anchors.y[i],
                                        bottom_anchors.z[i] - top_anchors.z[i]])
            rod_orientation /= np.linalg.norm(rod_orientation)
            rod_orientations.append(rod_orientation)

        return rod_orientations

    def _get_ball_joint_angle(self, rod_orientation, anchor_orientation):
        rod_orientation /= np.linalg.norm(rod_orientation)
        anchor_orientation /= np.linalg.norm(anchor_orientation)

        return abs(math.acos(rod_orientation.dot(anchor_orientation)) - math.pi / 2)

    def get_kinematics_controller_parameters(self):
        is_horn_orientation_reversed = [self._configuration.bottom_configuration.servos[i].is_horn_orientation_reversed
                                        for i in range(len(self._configuration.bottom_configuration.servos))]
        return {
            'servo_angle_min': self._configuration.servo_angle_min,
            'servo_angle_max': self._configuration.servo_angle_max,
            'rod_length': self._configuration.rod_length,
            'horn_length': self._configuration.bottom_configuration.horn_length,
            'horn_orientation_angles': self._configuration_preprocessor.bottom_horn_orientation_angles,
            'is_horn_orientation_reversed': is_horn_orientation_reversed,
            'top_anchors': self.top_state.get_initial_anchors(),
            'bottom_linear_actuator_anchors': self._configuration_preprocessor.bottom_linear_actuator_anchors,
            'forward_kinematics_x0': self._forward_kinematics.get_x0(),
            'forward_kinematics_bounds': self._forward_kinematics.get_bounds()
        }


class TopState:
    def __init__(self, configuration):
        self._initial_position = np.zeros(3)
        self._initial_orientation = Rotation.from_euler('xyz', [0, 0, 0])

        self._position = np.zeros(3)
        self._orientation = Rotation.from_euler('xyz', [0, 0, 0])

        self._shapes = [_generate_shape_from_configuration(shape) for shape in configuration.shapes]
        self._anchors = self._generate_anchors_from_configuration(configuration.anchors,
                                                                  configuration.ball_joint_offset)
        self._anchor_orientations = [np.array([anchor.orientation_x, anchor.orientation_y, 0.0])
                                     for anchor in configuration.anchors]

    def _generate_anchors_from_configuration(self, anchors_configuration, ball_joint_offset):
        x = np.array([anchor.x + anchor.orientation_x * ball_joint_offset for anchor in anchors_configuration])
        y = np.array([anchor.y + anchor.orientation_y * ball_joint_offset for anchor in anchors_configuration])
        z = np.zeros(len(x))

        return Shape(x, y, z)

    def get_initial_position(self):
        return self._initial_position

    def _set_initial_position(self, position):
        self._initial_position = position

    def get_initial_orientation(self):
        return self._initial_orientation

    def _set_initial_orientation(self, orientation):
        self._initial_orientation = orientation

    def get_position(self):
        return self._position

    def _set_position(self, position):
        self._position = position

    def get_orientation(self):
        return self._orientation

    def _set_orientation(self, orientation):
        self._orientation = orientation

    def get_transformed_shapes(self):
        return [shape.rotate(self._orientation).translate(self._position) for shape in self._shapes]

    def get_transformed_anchors(self):
        return self._anchors.rotate(self._orientation).translate(self._position)

    def _get_transformed_anchor_orientations(self):
        return [self._orientation.apply(anchor_orientation) for anchor_orientation in self._anchor_orientations]

    def get_initial_anchors(self):
        return self._anchors


class BottomState:
    def __init__(self, configuration):
        self._configuration = configuration

        self._shapes = [_generate_shape_from_configuration(shape) for shape in configuration.shapes]
        self._servos = self._generate_servos_from_configuration(configuration.servos)
        self._horns = [self._generate_horns_from_configuration(servo,
                                                               configuration.horn_length,
                                                               configuration.horn_offset)
                       for servo in configuration.servos]

        self._anchor_orientations = [np.array([servo.orientation_x, servo.orientation_y, 0.0])
                                     for servo in configuration.servos]

        self._servo_angles = np.zeros(len(self._horns))

    def _generate_servos_from_configuration(self, servos_configuration):
        x = np.array([servo.x for servo in servos_configuration])
        y = np.array([servo.y for servo in servos_configuration])
        z = np.zeros(len(x))

        return Shape(x, y, z)

    def _generate_horns_from_configuration(self, servo_configuration, horn_length, horn_offset):
        x1 = servo_configuration.x + servo_configuration.orientation_x * horn_offset
        y1 = servo_configuration.y + servo_configuration.orientation_y * horn_offset
        z1 = 0

        horn_orientation = np.cross([servo_configuration.orientation_x, servo_configuration.orientation_y, 0],
                                    [0, 0, 1 if servo_configuration.is_horn_orientation_reversed else -1])
        horn_orientation *= 1.0 / np.linalg.norm(horn_orientation)

        x2 = x1 + horn_length * horn_orientation[0]
        y2 = y1 + horn_length * horn_orientation[1]
        z2 = 0

        return Shape(np.array([x1, x2]), np.array([y1, y2]), np.array([z1, z2]))

    def get_servo_angles(self):
        return self._servo_angles

    def _set_servo_angles(self, angles):
        if len(angles) != len(self._servo_angles):
            raise ValueError('Invalid servo angles')
        self._servo_angles = angles

    def get_transformed_shapes(self):
        return self._shapes

    def get_transformed_servos(self):
        return self._servos

    def get_transformed_horns(self):
        transformed_horns = []

        for i in range(len(self._servo_angles)):
            center = np.array([self._horns[i].x[0], self._horns[i].y[0], self._horns[i].z[0]])
            axis = np.array([self._configuration.servos[i].orientation_x,
                             self._configuration.servos[i].orientation_y, 0])
            r = Rotation.from_rotvec(axis * self._servo_angles[i])

            transformed_horns.append(self._horns[i].rotate(r, center))

        return transformed_horns

    def get_transformed_anchors(self):
        horns = self.get_transformed_horns()
        return self._generate_anchors_from_horns(horns, self._configuration.servos,
                                                 self._configuration.ball_joint_offset)

    def _get_transformed_anchor_orientations(self):
        return self._anchor_orientations

    def get_initial_anchors(self):
        return self._generate_anchors_from_horns(self._horns, self._configuration.servos,
                                                 self._configuration.ball_joint_offset)

    def _generate_anchors_from_horns(self, horns, servos_configuration, ball_joint_offset):
        x = np.zeros(len(horns))
        y = np.zeros(len(x))
        z = np.zeros(len(x))

        for i in range(len(x)):
            x[i] = horns[i].x[1] + servos_configuration[i].orientation_x * ball_joint_offset
            y[i] = horns[i].y[1] + servos_configuration[i].orientation_y * ball_joint_offset
            z[i] = horns[i].z[1]

        return Shape(x, y, z)


def _generate_shape_from_configuration(shape_configuration):
    if shape_configuration.type == ShapeType.CIRCLE:
        return _generate_circle_shape(shape_configuration)
    elif shape_configuration.type == ShapeType.POLYGON:
        return _generate_polygon_shape(shape_configuration)
    else:
        raise ValueError('Invalid shape type (' + shape_configuration.type + ')')


def _generate_circle_shape(shape_configuration):
    angles = np.arange(0, 2 * math.pi, CIRCLE_ANGLE_RESOLUTION)
    x = shape_configuration.radius * np.cos(angles) + shape_configuration.x
    y = shape_configuration.radius * np.sin(angles) + shape_configuration.y
    z = np.zeros(len(x))

    return Shape(x, y, z)


def _generate_polygon_shape(shape_configuration):
    x = np.array(shape_configuration.x)
    y = np.array(shape_configuration.y)
    z = np.zeros(len(x))

    return Shape(x, y, z)
