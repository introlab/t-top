import torch.nn as nn


def generate_kinematics_controller_parameters_code(parameters):
    opencr_parameters = _generate_inverse_kinematics_controller_parameters_code(parameters)
    opencr_parameters += '\n\n\n\n'
    opencr_parameters += \
        _generate_forward_kinematics_controller_parameters_code(parameters['forward_kinematics_nn_layers'][0],
                                                                'position_')
    opencr_parameters += '\n\n\n\n'
    opencr_parameters += \
        _generate_forward_kinematics_controller_parameters_code(parameters['forward_kinematics_nn_layers'][1],
                                                                'orientation_')

    return opencr_parameters


def _generate_inverse_kinematics_controller_parameters_code(parameters):
    opencr_parameters = 'static const int SERVO_COUNT 6\n\n'

    opencr_parameters += 'static const float ROD_LENGTH = ' + str(parameters['rod_length']) + ';\n'
    opencr_parameters += 'static const float HORN_LENGTH = ' + str(parameters['horn_length']) + ';\n'
    opencr_parameters += 'static const float TOP_INITIAL_Z = ' + str(parameters['top_initial_z']) + ';\n\n'

    opencr_parameters += 'static const float HORN_ORIENTATION_ANGLES[SERVO_COUNT] =\n'
    opencr_parameters += '{\n'
    for horn_orientation_angle in parameters['horn_orientation_angles']:
        opencr_parameters += '    ' + str(horn_orientation_angle) + ',\n'
    opencr_parameters += '};\n\n'

    opencr_parameters += 'static const bool IS_HORN_ORIENTATION_REVERSED[SERVO_COUNT] =\n'
    opencr_parameters += '{\n'
    for is_horn_orientation_reversed in parameters['is_horn_orientation_reversed']:
        opencr_parameters += '    ' + ('true' if is_horn_orientation_reversed else 'false') + ',\n'
    opencr_parameters += '};\n\n'

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


def _generate_forward_kinematics_controller_parameters_code(layers, prefix):
    fully_connected_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]

    members_matrices = _generate_forward_kinematics_controller_members_matrices(fully_connected_layers, prefix)

    nn_code = _generate_forward_kinematics_controller_nn_code(layers, prefix)
    nn_parameters = _generate_forward_kinematics_controller_nn_parameters(fully_connected_layers, prefix)

    return members_matrices + '\n\n' + nn_code + '\n\n' + nn_parameters


def _generate_forward_kinematics_controller_members_matrices(fully_connected_layers, prefix):
    members_matrices = 'Eigen::Matrix<float, ' + str(fully_connected_layers[0].in_features) + ', 1> ' + prefix + \
                       'input;\n'
    for i in range(0, len(fully_connected_layers) - 1):
        output_count = fully_connected_layers[i].out_features
        members_matrices += 'Eigen::Matrix<float, ' + str(output_count) + ', 1> ' + prefix + 'h' + str(i) + ';\n'

    members_matrices += 'Eigen::Matrix<float, ' + str(fully_connected_layers[-1].out_features) + ', 1> ' + prefix + \
                        'output;'

    return members_matrices


def _generate_forward_kinematics_controller_nn_code(layers, prefix):
    nn_code = ''
    i = 0
    layer_id = 0

    upper_prefix = prefix.upper()

    while i < len(layers):
        if not isinstance(layers[i], nn.Linear):
            raise ValueError(f'Invalid layer: {layers[i]}')

        is_next_relu = i + 1 < len(layers) and isinstance(layers[i + 1], nn.ReLU)

        if i == 0:
            line = upper_prefix + 'WEIGHTS_' + str(layer_id) + ' * ' + prefix + 'input + ' + \
                   upper_prefix + 'BIASES_' + str(layer_id)
        else:
            line = upper_prefix + 'WEIGHTS_' + str(layer_id) + ' * ' + prefix + 'h' + str(layer_id - 1) + ' + ' + \
                   upper_prefix + 'BIASES_' + str(layer_id)

        if is_next_relu:
            line = '(' + line + ').cwiseMax(0.f);'
        else:
            line += ';'

        if i == len(layers) - 1:
            line = prefix + 'output = ' + line
        else:
            line = prefix + 'h' + str(layer_id) + ' = ' + line

        nn_code += line + '\n'
        if is_next_relu:
            i += 2
        else:
            i += 1

        layer_id += 1

    if prefix == 'orientation_':
        nn_code += 'orientation_output /= orientation_output.norm();\n'
    elif prefix != 'position_':
        raise ValueError('Invalid prefix')

    return nn_code


def _generate_forward_kinematics_controller_nn_parameters(fully_connected_layers, prefix):
    nn_parameters = ''

    prefix = prefix.upper()

    for i in range(len(fully_connected_layers)):
        weights = fully_connected_layers[i].weight
        data = ', '.join((str(n.item()) for row in weights for n in row))
        nn_parameters += 'static const float ' + prefix + 'WEIGHTS_DATA_' + str(i) + '[] = { ' + data + ' };\n'

        w_type = 'const Eigen::Map<const Eigen::Matrix<float, ' + str(weights.size(0)) + \
                 ', ' + str(weights.size(1)) + ', Eigen::RowMajor>>'
        nn_parameters += 'static ' + w_type + ' ' + prefix + 'WEIGHTS_' + str(i) + \
                         '(' + prefix + 'WEIGHTS_DATA_' + str(i) + ');\n'

        biases = fully_connected_layers[i].bias
        data = ', '.join((str(n.item()) for n in biases))
        nn_parameters += 'static const float ' + prefix + 'BIASES_DATA_' + str(i) + '[] = { ' + data + ' };\n'

        b_type = 'Eigen::Matrix<float, ' + str(biases.size(0)) + ', 1>'
        nn_parameters += 'static const ' + b_type + ' ' + prefix + 'BIASES_' + str(i) + \
                         '(' + prefix + 'BIASES_DATA_' + str(i) + ');\n\n'

    return nn_parameters
