def generate_kinematics_controller_parameters_code(parameters):
    opencr_parameters = '/****************************header**********************************/\n'
    opencr_parameters += _generate_global_controller_parameters_header_code(parameters)
    opencr_parameters += '\n\n'
    opencr_parameters += _generate_forward_kinematics_controller_parameters_header_code(parameters)
    opencr_parameters += '\n\n'
    opencr_parameters += '/****************************src************************************/\n'
    opencr_parameters += _generate_global_controller_parameters_code(parameters)
    opencr_parameters += '\n\n'
    opencr_parameters += _generate_forward_kinematics_controller_parameters_code(parameters)

    return opencr_parameters


def _generate_global_controller_parameters_header_code(parameters):
    opencr_parameters = 'constexpr int STEWART_SERVO_COUNT = 6;\n\n'

    opencr_parameters += 'constexpr float STEWART_SERVO_ANGLE_MIN = ' + \
                         _format_float(parameters['servo_angle_min']) + ';\n'
    opencr_parameters += 'constexpr float STEWART_SERVO_ANGLE_MAX = ' + \
                         _format_float(parameters['servo_angle_max']) + ';\n'

    opencr_parameters += 'constexpr float STEWART_ROD_LENGTH = ' + \
                         _format_float(parameters['rod_length']) + ';\n'
    opencr_parameters += 'constexpr float STEWART_HORN_LENGTH = ' + \
                         _format_float(parameters['horn_length']) + ';\n'

    opencr_parameters += 'extern const Eigen::Matrix<float, STEWART_SERVO_COUNT, 1> STEWART_HORN_ORIENTATION_ANGLES;\n'
    opencr_parameters += 'extern const bool STEWART_IS_HORN_ORIENTATION_REVERSED[STEWART_SERVO_COUNT];\n'
    opencr_parameters += 'extern const Eigen::Vector3f STEWART_TOP_ANCHORS[STEWART_SERVO_COUNT];\n'
    opencr_parameters += 'extern const Eigen::Vector3f STEWART_BOTTOM_LINEAR_ACTUATOR_ANCHORS[STEWART_SERVO_COUNT];'

    return opencr_parameters


def _generate_global_controller_parameters_code(parameters):
    opencr_parameters = 'const float STEWART_HORN_ORIENTATION_ANGLES_DATA[STEWART_SERVO_COUNT] = {\n' + \
                        _format_float_array(parameters['horn_orientation_angles'], indent='  ') + '\n};\n'
    opencr_parameters += 'const Eigen::Matrix<float, STEWART_SERVO_COUNT, 1> STEWART_HORN_ORIENTATION_ANGLES(' + \
                        'STEWART_HORN_ORIENTATION_ANGLES_DATA);\n\n'

    opencr_parameters += 'const bool STEWART_IS_HORN_ORIENTATION_REVERSED[STEWART_SERVO_COUNT] = {'
    opencr_parameters += ', '.join((_format_bool(v) for v in parameters['is_horn_orientation_reversed']))
    opencr_parameters += '};\n\n'

    opencr_parameters += 'const Eigen::Vector3f STEWART_TOP_ANCHORS[STEWART_SERVO_COUNT] = {\n'
    opencr_parameters += _format_vector3_array(parameters['top_anchors'], '  ')
    opencr_parameters += '};\n\n'

    opencr_parameters += 'const Eigen::Vector3f STEWART_BOTTOM_LINEAR_ACTUATOR_ANCHORS[STEWART_SERVO_COUNT] = {\n'
    opencr_parameters += _format_vector3_array(parameters['bottom_linear_actuator_anchors'], '  ')
    opencr_parameters += '};'

    return opencr_parameters


def _generate_forward_kinematics_controller_parameters_header_code(parameters):
    size = str(len(parameters['forward_kinematics_x0']))
    opencr_parameters = 'extern const Eigen::Matrix<float, ' + size + ', 1> STEWART_FORWARD_KINEMATICS_X0;\n'
    opencr_parameters += 'extern const Eigen::Matrix<float, ' + size + ', 1> STEWART_FORWARD_KINEMATICS_MIN_BOUNDS;\n'
    opencr_parameters += 'extern const Eigen::Matrix<float, ' + size + ', 1> STEWART_FORWARD_KINEMATICS_MAX_BOUNDS;\n'

    return opencr_parameters


def _generate_forward_kinematics_controller_parameters_code(parameters):
    x0 = parameters['forward_kinematics_x0']
    opencr_parameters = 'float STEWART_FORWARD_KINEMATICS_X0_DATA[' + str(len(x0)) + '] = {\n' + \
                        _format_float_array(x0, indent='  ') + '\n};\n'
    opencr_parameters += 'const Eigen::Matrix<float, ' + str(len(x0)) + ', 1> ' + \
                         'STEWART_FORWARD_KINEMATICS_X0(STEWART_FORWARD_KINEMATICS_X0_DATA);\n\n'

    min_bounds = parameters['forward_kinematics_bounds'][0]
    max_bounds = parameters['forward_kinematics_bounds'][1]

    opencr_parameters += 'float STEWART_FORWARD_KINEMATICS_MIN_BOUNDS_DATA[' + str(len(min_bounds)) + '] = {\n' + \
                         _format_float_array(min_bounds, indent='  ') + '\n};\n'
    opencr_parameters += 'const Eigen::Matrix<float, ' + str(len(min_bounds)) + ', 1> ' + \
                         'STEWART_FORWARD_KINEMATICS_MIN_BOUNDS(STEWART_FORWARD_KINEMATICS_MIN_BOUNDS_DATA);\n\n'
    opencr_parameters += 'float STEWART_FORWARD_KINEMATICS_MAX_BOUNDS_DATA[' + str(len(max_bounds)) + '] = {\n' + \
                         _format_float_array(max_bounds, indent='  ') + '\n};\n'
    opencr_parameters += 'const Eigen::Matrix<float, ' + str(len(max_bounds)) + ', 1> ' + \
                         'STEWART_FORWARD_KINEMATICS_MAX_BOUNDS(STEWART_FORWARD_KINEMATICS_MAX_BOUNDS_DATA);\n\n'

    return opencr_parameters


def _format_float(value):
    return str(value) + 'f'


def _format_bool(value):
    return 'true' if value else 'false'


def _format_float_array(values, indent=''):
    return indent + (',\n' + indent).join((_format_float(v) for v in values))


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
