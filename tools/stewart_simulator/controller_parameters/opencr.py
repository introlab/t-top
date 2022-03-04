import torch.nn as nn


def generate_kinematics_controller_parameters_code(parameters):
    opencr_parameters = _generate_global_controller_parameters_code(parameters)
    opencr_parameters += '\n\n\n\n'
    opencr_parameters += _generate_forward_kinematics_controller_parameters_code(parameters)

    return opencr_parameters


def _generate_global_controller_parameters_code(parameters):
    opencr_parameters = 'static const int SERVO_COUNT 6;\n\n'

    opencr_parameters += 'static const float ROD_LENGTH = ' + str(parameters['rod_length']) + ';\n'
    opencr_parameters += 'static const float HORN_LENGTH = ' + str(parameters['horn_length']) + ';\n'

    opencr_parameters += 'static const float HORN_ORIENTATION_ANGLES[SERVO_COUNT] =\n'
    opencr_parameters += '{\n'
    opencr_parameters += '    '
    opencr_parameters += ',\n    '.join((str(v) for v in parameters['horn_orientation_angles']))
    opencr_parameters += '\n};\n\n'

    opencr_parameters += 'static const bool IS_HORN_ORIENTATION_REVERSED[SERVO_COUNT] =\n'
    opencr_parameters += '{\n'
    opencr_parameters += '    '
    opencr_parameters += ',\n    '.join(('true' if v else 'false' for v in parameters['is_horn_orientation_reversed']))
    opencr_parameters += '\n};\n\n'

    opencr_parameters += 'static const Eigen::Vector3f TOP_ANCHORS[SERVO_COUNT] =\n'
    opencr_parameters += '{\n'
    for i in range(len(parameters['top_anchors'].x)):
        opencr_parameters += '    Eigen::Vector3f(' + \
                             str(parameters['top_anchors'].x[i]) + ', ' + \
                             str(parameters['top_anchors'].y[i]) + ', ' + \
                             str(parameters['top_anchors'].z[i]) + '),\n'
    opencr_parameters += '};\n\n'

    opencr_parameters += 'static const Eigen::Vector3f BOTTOM_LINEAR_ACTUATOR_ANCHORS[SERVO_COUNT] =\n'
    opencr_parameters += '{\n'
    for i in range(len(parameters['bottom_linear_actuator_anchors'].x)):
        opencr_parameters += '    Eigen::Vector3f(' + \
                             str(parameters['bottom_linear_actuator_anchors'].x[i]) + ', ' + \
                             str(parameters['bottom_linear_actuator_anchors'].y[i]) + ', ' + \
                             str(parameters['bottom_linear_actuator_anchors'].z[i]) + '),\n'
    opencr_parameters += '};'

    return opencr_parameters


def _generate_forward_kinematics_controller_parameters_code(parameters):
    opencr_parameters = 'static const Eigen::Vector<float, ' + str(len(parameters['forward_kinematics_x0'])) + '> ' + \
        'FORWARD_KINEMATICS_X0(' + _remove_brackets(str(parameters['forward_kinematics_x0'])) + ');\n\n'

    min_bounds = parameters['forward_kinematics_bounds'][0]
    max_bounds = parameters['forward_kinematics_bounds'][1]

    opencr_parameters += 'static const Eigen::Vector<float, ' + str(len(min_bounds)) + '> ' + \
                         'FORWARD_KINEMATICS_MIN_BOUNDS(' + _remove_brackets(str(min_bounds)) + ');\n\n'
    opencr_parameters += 'static const Eigen::Vector<float, ' + str(len(max_bounds)) + '> ' + \
                         'FORWARD_KINEMATICS_MAX_BOUNDS(' + _remove_brackets(str(max_bounds)) + ');\n\n'

    return opencr_parameters


def _remove_brackets(string):
    return string.replace('[', '').replace(']', '')
