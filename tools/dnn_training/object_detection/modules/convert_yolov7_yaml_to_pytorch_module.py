import argparse
import os

import yaml


class Layer:
    def __init__(self, init_code, forward_code):
        self.init_code = init_code
        self.forward_code = forward_code


def convert(yaml_path, python_output_path, class_name):
    with open(yaml_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)

    all_anchor_counts = [len(a) // 2 for a in yaml_data['anchors']]

    layers, output_strides = _convert_yaml_to_layers(yaml_data['backbone'] + yaml_data['head'], yaml_data['nc'],
                                                     all_anchor_counts)

    with open(python_output_path, 'w') as python_file:
        _write_header(python_file, class_name, yaml_path)
        _write_init(python_file, layers, class_name, yaml_data['anchors'], output_strides)
        _write_getters(python_file)
        _write_forward(python_file, layers)
        _write_load_weights(python_file)


def _convert_yaml_to_layers(yaml_list, class_count, all_anchor_counts):
    if len(yaml_list) == 0:
        raise ValueError('At least one layer is required.')

    all_outputs = []
    all_channels = []
    all_strides = []
    layers = []

    i = 0
    for i, (input_index, c, layer_type, arguments) in enumerate(yaml_list):
        if c != 1:
            raise ValueError('C must be 1.')
        layers.append(
            _convert_to_layer(class_count, all_anchor_counts, all_outputs, all_channels, all_strides, input_index,
                              layer_type, arguments, i))

        if layer_type == 'Detect':
            break

    output_strides = [all_strides[i] for i in yaml_list[i][0]]
    return layers, output_strides


def _convert_to_layer(class_count, all_anchor_counts, all_outputs, all_channels, all_strides, input_index, layer_type,
                      arguments, i):
    input = _input_index_to_input(input_index, all_outputs)
    input_channels = _input_index_to_channels(input_index, all_channels)
    input_stride = _input_index_to_stride(input_index, all_strides)

    if layer_type == 'Conv':
        layer, output, output_channels, stride = _convert_conv_to_layer(input, input_channels, arguments, i)
    elif layer_type == 'MP':
        layer, output, output_channels, stride = _convert_mp_to_layer(input, input_channels, arguments, i)
    elif layer_type == 'SP':
        layer, output, output_channels, stride = _convert_sp_to_layer(input, input_channels, arguments, i)
    elif layer_type == 'SPPCSPC':
        layer, output, output_channels, stride = _convert_sppcspc_to_layer(input, input_channels, arguments, i)
    elif layer_type == 'RepConv':
        layer, output, output_channels, stride = _convert_rep_conv_to_layer(input, input_channels, arguments, i)
    elif layer_type == 'Concat':
        stride = 1
        layer, output, output_channels = _convert_concat_to_layer(input, input_channels, arguments, i)
    elif layer_type == 'nn.Upsample':
        layer, output, output_channels, stride = _convert_upsample_to_layer(input, input_channels, arguments, i)
    elif layer_type == 'Detect':
        return _convert_detect_to_layer(class_count, all_anchor_counts, input_index, all_outputs, all_channels,
                                        all_strides)
    else:
        raise ValueError('Invalid layer type (' + layer_type + ')')

    all_outputs.append(output)
    all_channels.append(output_channels)
    all_strides.append(int(input_stride * stride))

    return layer


def _input_index_to_input(input_index, all_outputs):
    if len(all_outputs) == 0:
        return 'x'
    elif isinstance(input_index, list):
        return '[' + ', '.join((all_outputs[i] for i in input_index)) + ']'
    else:
        return all_outputs[input_index]


def _input_index_to_channels(input_index, all_channels):
    if len(all_channels) == 0:
        return 3
    elif isinstance(input_index, list):
        return sum((all_channels[i] for i in input_index))
    else:
        return all_channels[input_index]


def _input_index_to_stride(input_index, all_strides):
    if len(all_strides) == 0:
        return 1
    elif isinstance(input_index, list):
        return all_strides[input_index[0]]
    else:
        return all_strides[input_index]


def _convert_conv_to_layer(input, input_channels, arguments, i):
    if len(arguments) > 6:
        raise ValueError('Too many arguments')

    output_channels = arguments[0]
    kernel_size = arguments[1] if len(arguments) > 1 else 1
    stride = arguments[2] if len(arguments) > 2 else 1
    padding = arguments[3] if len(arguments) > 3 and arguments[3] != 'None' else kernel_size // 2
    groups = arguments[4] if len(arguments) > 4 else 1
    activation = arguments[5] if len(arguments) > 5 else 'nn.SiLU()'

    layer_name = f'self._conv{i}'
    output = 'y' + str(i)

    init_code = (f'        {layer_name} = nn.Sequential(\n'
                 f'            nn.Conv2d(in_channels={input_channels}, out_channels={output_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, groups={groups}, bias=False),\n'
                 f'            nn.BatchNorm2d({output_channels}, eps=0.001),\n'
                 f'            {activation},\n'
                 f'        )'
                 )
    forward_code = f'        {output} = {layer_name}({input})'

    return Layer(init_code, forward_code), output, output_channels, stride


def _convert_mp_to_layer(input, input_channels, arguments, i):
    if len(arguments) > 1:
        raise ValueError('Too many arguments')

    kernel_size = arguments[0] if len(arguments) > 0 else 2
    stride = kernel_size

    layer_name = f'self._max_pool{i}'
    output = 'y' + str(i)

    init_code = f'        {layer_name} = nn.MaxPool2d(kernel_size={kernel_size}, stride={stride})'
    forward_code = f'        {output} = {layer_name}({input})'

    return Layer(init_code, forward_code), output, input_channels, stride


def _convert_sp_to_layer(input, input_channels, arguments, i):
    if len(arguments) > 2:
        raise ValueError('Too many arguments')

    kernel_size = arguments[0] if len(arguments) > 0 else 3
    stride = arguments[1] if len(arguments) > 1 else 1

    layer_name = f'self._max_pool{i}'
    output = 'y' + str(i)

    init_code = f'        {layer_name} = nn.MaxPool2d(kernel_size={kernel_size}, stride={stride}, padding={kernel_size // 2})'
    forward_code = f'        {output} = {layer_name}({input})'

    return Layer(init_code, forward_code), output, input_channels, stride


def _convert_sppcspc_to_layer(input, input_channels, arguments, i):
    if len(arguments) > 1:
        raise ValueError('Too many arguments')

    output_channels = arguments[0]
    stride = 1

    layer_name = f'self._sppcspc{i}'
    output = 'y' + str(i)

    init_code = f'        {layer_name} = YoloV7SPPCSPC({input_channels}, {output_channels})'
    forward_code = f'        {output} = {layer_name}({input})'

    return Layer(init_code, forward_code), output, output_channels, stride


def _convert_rep_conv_to_layer(input, input_channels, arguments, i):
    if len(arguments) > 6:
        raise ValueError('Too many arguments')

    output_channels = arguments[0]
    kernel_size = arguments[1] if len(arguments) > 1 else 1
    stride = arguments[2] if len(arguments) > 2 else 1
    padding = arguments[3] if len(arguments) > 3 and arguments[3] != 'None' else kernel_size // 2
    groups = arguments[4] if len(arguments) > 4 else 1
    activation = arguments[5] if len(arguments) > 5 else 'nn.SiLU()'

    layer_name = f'self._rep_conv{i}'
    output = 'y' + str(i)

    init_code = f'        {layer_name} = RepConv(in_channels={input_channels}, out_channels={output_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, groups={groups}, activation={activation})'
    forward_code = f'        {output} = {layer_name}({input})'

    return Layer(init_code, forward_code), output, output_channels, stride


def _convert_concat_to_layer(input, input_channels, arguments, i):
    if len(arguments) > 1:
        raise ValueError('Too many arguments')

    dim = arguments[0] if len(arguments) > 0 else 1

    output = 'y' + str(i)

    init_code = ''
    forward_code = f'        {output} = torch.cat({input}, dim={dim})\n'

    return Layer(init_code, forward_code), output, input_channels


def _convert_upsample_to_layer(input, input_channels, arguments, i):
    if len(arguments) > 3:
        raise ValueError('Too many arguments')

    size = arguments[0] if len(arguments) > 0 else None
    scale_factor = arguments[1] if len(arguments) > 1 else None
    mode = arguments[2] if len(arguments) > 2 else 'nearest'

    if size != 'None':
        raise ValueError(f'Size must be None ({size})')

    layer_name = f'self._upsample{i}'
    output = 'y' + str(i)

    init_code = f'        {layer_name} = nn.Upsample(scale_factor={scale_factor}, mode=\'{mode}\')'
    forward_code = f'        {output} = {layer_name}({input})'

    return Layer(init_code, forward_code), output, input_channels, 1.0 / scale_factor


def _convert_detect_to_layer(class_count, all_anchor_counts, input_indexes, all_outputs, all_channels, all_strides):
    init_code = '\n'
    forward_code = '\n'

    layer_names = []
    output_names = []

    for i, input_index in enumerate(input_indexes):
        layer_names.append(f'self._yolo{i}')
        output_names.append(f'd{i}')

        init_code += (f'        {layer_names[i]} = nn.Sequential(\n'
                      f'            nn.Conv2d(in_channels={all_channels[input_index]}, out_channels={all_anchor_counts[i] * (class_count + 5)}, kernel_size=1),\n'
                      f'            YoloV7Layer(IMAGE_SIZE, {all_strides[input_index]}, self._anchors[{i}], {class_count}, class_probs=class_probs)\n'
                      f'        )\n'
                      )
        forward_code += f'        {output_names[i]} = {layer_names[i]}({all_outputs[input_index]})\n'

    forward_code += f'        return [{", ".join(output_names)}]'

    return Layer(init_code, forward_code)


def _write_header(python_file, class_name, yaml_path):
    python_file.write('from collections import OrderedDict\n')
    python_file.write('\n')
    python_file.write('import numpy as np\n')
    python_file.write('\n')
    python_file.write('import torch\n')
    python_file.write('import torch.nn as nn\n')
    python_file.write('\n')
    python_file.write('from object_detection.modules.yolo_layer import YoloV7Layer\n')
    python_file.write('from object_detection.modules.yolo_v7_modules import YoloV7SPPCSPC, RepConv\n')
    python_file.write('\n')
    python_file.write('\n')
    python_file.write(f'IMAGE_SIZE = (640, 640)\n')
    python_file.write('IN_CHANNELS = 3\n')
    python_file.write('\n')
    python_file.write('\n')
    python_file.write(f'# Generated from: {os.path.basename(yaml_path)}:\n')
    python_file.write(f'class {class_name}(nn.Module):\n')


def _write_init(python_file, layers, class_name, anchors, output_strides):
    python_file.write('    def __init__(self, class_probs=False):\n')
    python_file.write(f'        super({class_name}, self).__init__()\n\n')

    python_file.write('        self._anchors = []\n')
    python_file.write(f'        self._output_strides = {output_strides}\n')

    for a in anchors:
        python_file.write(
            f'        self._anchors.append(np.array([({a[0]}, {a[1]}), ({a[2]}, {a[3]}), ({a[4]}, {a[5]})]))\n')

    python_file.write('\n')
    for layer in layers:
        python_file.write(layer.init_code)
        python_file.write('\n')

    python_file.write('\n')


def _write_getters(python_file):
    python_file.write('    def get_image_size(self):\n')
    python_file.write('        return IMAGE_SIZE\n')
    python_file.write('\n')
    python_file.write('    def get_anchors(self):\n')
    python_file.write('        return self._anchors\n')
    python_file.write('\n')
    python_file.write('    def get_output_strides(self):\n')
    python_file.write('        return self._output_strides\n\n')


def _write_forward(python_file, layers):
    python_file.write('    def forward(self, x):\n')

    for layer in layers:
        python_file.write(layer.forward_code)
        python_file.write('\n')

    python_file.write('\n')


def _write_load_weights(python_file):
    python_file.write('    def load_weights(self, weights_path):\n')
    python_file.write('        loaded_state_dict = self._filter_static_dict(torch.load(weights_path), \'anchor\')\n')
    python_file.write('        current_state_dict = self._filter_static_dict(self.state_dict(), \'offset\')\n')
    python_file.write('\n')
    python_file.write(
        '        for i, (kl, kc) in enumerate(zip(loaded_state_dict.keys(), current_state_dict.keys())):\n')
    python_file.write('            if current_state_dict[kc].size() != loaded_state_dict[kl].size():\n')
    python_file.write('                raise ValueError(\'Mismatching size.\')\n')
    python_file.write('            current_state_dict[kc] = loaded_state_dict[kl]\n')
    python_file.write('\n')
    python_file.write('        self.load_state_dict(current_state_dict, strict=False)\n')
    python_file.write('\n')

    python_file.write('    def _filter_static_dict(self, state_dict, x):\n')
    python_file.write('        return OrderedDict([(k, v) for k, v in state_dict.items() if x not in k])\n')


def main():
    parser = argparse.ArgumentParser(description='Convert the specified darknet configuration file to PyTorch')
    parser.add_argument('--yaml_path', type=str, help='Choose the configuration file', required=True)
    parser.add_argument('--python_output_path', type=str, help='Choose the Python output file', required=True)
    parser.add_argument('--class_name', type=str, help='Choose the class name', required=True)

    args = parser.parse_args()
    convert(args.yaml_path, args.python_output_path, args.class_name)


if __name__ == '__main__':
    main()
