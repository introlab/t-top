import torch.nn as nn


def generate_kinematics_controller_parameters_code(parameters):
    opencr_parameters = _generate_global_controller_parameters_code(parameters)
    opencr_parameters += '\n\n\n\n'
    opencr_parameters += _generate_forward_kinematics_controller_parameters_code(parameters)

    return opencr_parameters


def _generate_global_controller_parameters_code(parameters):
    opencr_parameters = 'static const int SERVO_COUNT 6;\n\n'

    opencr_parameters += 'static const float ROD_LENGTH = ' + _format_float(parameters['rod_length']) + ';\n'
    opencr_parameters += 'static const float HORN_LENGTH = ' + _format_float(parameters['horn_length']) + ';\n'

    opencr_parameters += 'static const float HORN_ORIENTATION_ANGLES[SERVO_COUNT] =\n'
    opencr_parameters += '{\n'
    opencr_parameters += '    '
    opencr_parameters += ',\n    '.join((_format_float(v) for v in parameters['horn_orientation_angles']))
    opencr_parameters += '\n};\n\n'

    opencr_parameters += 'static const bool IS_HORN_ORIENTATION_REVERSED[SERVO_COUNT] =\n'
    opencr_parameters += '{\n'
    opencr_parameters += '    '
    opencr_parameters += ',\n    '.join((_format_bool(v) for v in parameters['is_horn_orientation_reversed']))
    opencr_parameters += '\n};\n\n'

    opencr_parameters += 'static const Eigen::Vector3f TOP_ANCHORS[SERVO_COUNT] =\n'
    opencr_parameters += '{\n'
    opencr_parameters += _format_vector3_array(parameters['top_anchors'], '    ')
    opencr_parameters += '};\n\n'

    opencr_parameters += 'static const Eigen::Vector3f BOTTOM_LINEAR_ACTUATOR_ANCHORS[SERVO_COUNT] =\n'
    opencr_parameters += '{\n'
    opencr_parameters += _format_vector3_array(parameters['bottom_linear_actuator_anchors'], '    ')
    opencr_parameters += '};'

    return opencr_parameters


def _generate_forward_kinematics_controller_parameters_code(parameters):
    opencr_parameters = 'static const Eigen::Vector<float, ' + str(len(parameters['forward_kinematics_x0'])) + '> ' + \
        'FORWARD_KINEMATICS_X0(' + _format_float_array(parameters['forward_kinematics_x0']) + ');\n\n'

    min_bounds = parameters['forward_kinematics_bounds'][0]
    max_bounds = parameters['forward_kinematics_bounds'][1]

    opencr_parameters += 'static const Eigen::Vector<float, ' + str(len(min_bounds)) + '> ' + \
                         'FORWARD_KINEMATICS_MIN_BOUNDS(' + _format_float_array(min_bounds) + ');\n\n'
    opencr_parameters += 'static const Eigen::Vector<float, ' + str(len(max_bounds)) + '> ' + \
                         'FORWARD_KINEMATICS_MAX_BOUNDS(' + _format_float_array(max_bounds) + ');\n\n'

    return opencr_parameters


def _format_float(value):
    return str(value) + 'f'


def _format_bool(value):
    return 'true' if value else 'false'


def _format_float_array(values):
    return ', '.join((_format_float(v) for v in values))


def _format_vector3_array(values, indent=''):
    string = ''
    for i in range(len(values.x)):
        string += indent + 'Eigen::Vector3f(' + \
                  _format_float(values.x[i]) + ', ' + \
                  _format_float(values.y[i]) + ', ' + \
                  _format_float(values.z[i]) + ')'

        if i < len(values.x) - 1:
            string += ',\n'
        else:
            string += '\n'

    return string

