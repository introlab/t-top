import argparse
import os

SUPPORTED_LAYER_TYPES = {'net', 'convolutional', 'upsample', 'route', 'shortcut', 'maxpool', 'yolo'}

ACTIVATION_MODULES_BY_NAME = {
    'relu': 'nn.ReLU(inplace=True)',
    'leaky': 'nn.LeakyReLU(0.1, inplace=True)',
    'mish': 'Mish()',
    'swish': 'Swish()',
    'logistic': 'nn.Sigmoid()',
    'tanh': 'nn.Tanh()',
    'linear': ''
}


def convert(cfg_path, python_output_path, class_name):
    layers = _read_layers(cfg_path)
    with open(python_output_path, 'w') as python_file:
        in_channels = _write_header(python_file, layers[0], class_name, cfg_path)
        _write_init(python_file, layers[1:], in_channels, class_name)
        _write_getters(python_file)
        _write_forward(python_file, layers[1:])
        _write_load_weights(python_file, layers[1:])
        _write_load_batch_norm_conv_weights(python_file)
        _write_load_conv_weights(python_file)


def _read_layers(cfg_path):
    with open(cfg_path, 'r') as cfg_file:
        lines = cfg_file.readlines()

    layer = {}
    layers = []

    for i, line in enumerate(lines):
        line = line.strip()
        if len(line) == 0 or line.startswith('#'):
            continue

        if line.startswith('[') and line.endswith(']'):
            if len(layer) != 0:
                layers.append(layer)
                layer = {}
            type = line[1:-1]
            if type in SUPPORTED_LAYER_TYPES:
                layer['type'] = type
                layer['line_no'] = i + 1
            else:
                raise ValueError('Invalid layer type (line={})'.format(i + 1))
        elif line.startswith('[') or line.endswith(']'):
            raise ValueError('Invalid cfg file (line={})'.format(i + 1))
        elif line.count('=') == 1:
            key, value = line.split('=')
            layer[key.strip()] = value.strip()
        else:
            raise ValueError('Invalid cfg file (line={})'.format(i + 1))

    if len(layer) != 0:
        layers.append(layer)
    return layers


def _write_header(python_file, layer, class_name, cfg_path):
    if layer['type'] != 'net':
        raise ValueError('The type of the first cfg block must be "net" (line={}).'.format(layer['line_no']))
    if 'height' not in layer:
        raise ValueError('A "net" block must contain the "height" attribute (line={}).'.format(layer['line_no']))
    if 'width' not in layer:
        raise ValueError('A "net" block must contain the "width" attribute (line={}).'.format(layer['line_no']))
    if 'channels' not in layer:
        raise ValueError('A "net" block must contain the "channels" attribute (line={}).'.format(layer['line_no']))

    python_file.write('import numpy as np\n')
    python_file.write('\n')
    python_file.write('import torch\n')
    python_file.write('import torch.nn as nn\n')
    python_file.write('\n')
    python_file.write('from common.modules import Mish, Swish\n')
    python_file.write('\n')
    python_file.write('from object_detection.modules.yolo_layer import YoloV4Layer\n')
    python_file.write('\n')
    python_file.write('\n')
    python_file.write('IMAGE_SIZE = ({}, {})\n'.format(layer['height'], layer['width']))
    python_file.write('IN_CHANNELS = {}\n'.format(layer['channels']))
    python_file.write('\n')
    python_file.write('\n')
    python_file.write('# Generated from: {}:\n'.format(os.path.basename(cfg_path)))
    python_file.write('class {}(nn.Module):\n'.format(class_name))

    return layer['channels']


def _write_init(python_file, layers, in_channels, class_name):
    python_file.write('    def __init__(self, class_probs=False):\n')
    python_file.write('        super({}, self).__init__()\n'.format(class_name))
    python_file.write('        self._anchors = []\n')
    python_file.write('        self._output_strides = []\n')

    cumulated_strides = [1]
    in_channels = [in_channels]
    for i, layer in enumerate(layers):
        if layer['type'] == 'convolutional':
            out_channels, stride = _write_init_convolutional(python_file, i, layer, in_channels[-1])
            in_channels.append(out_channels)
            cumulated_strides.append(cumulated_strides[-1] * stride)
        elif layer['type'] == 'upsample':
            out_channels, stride = _write_init_upsample(python_file, i, layer, in_channels[-1])
            in_channels.append(out_channels)
            cumulated_strides.append(cumulated_strides[-1] // stride)
        elif layer['type'] == 'maxpool':
            out_channels, stride = _write_init_maxpool(python_file, i, layer, in_channels[-1])
            in_channels.append(out_channels)
            cumulated_strides.append(cumulated_strides[-1] * stride)
        elif layer['type'] == 'yolo':
            in_channels.append(_write_init_yolo(python_file, i, layer, cumulated_strides[-1]))
            cumulated_strides.append(0)
        elif layer['type'] == 'route':
            out_channels, stride = _write_init_route(layer, in_channels, cumulated_strides)
            in_channels.append(out_channels)
            cumulated_strides.append(stride)
        elif layer['type'] == 'shortcut':
            out_channels, stride = _write_init_shortcut(layer, in_channels, cumulated_strides)
            in_channels.append(out_channels)
            cumulated_strides.append(stride)
        else:
            raise ValueError('Not supported layer (type={})'.format(layer['type']))
        python_file.write('\n')
    python_file.write('\n')


def _write_init_convolutional(python_file, i, layer, in_channels):
    not_supported_options = ['stride_x', 'dilation', 'antialiasing', 'padding',
                             'binary', 'xnor', 'bin_output', 'sway', 'rotate', 'stretch', 'stretch_sway',
                             'flipped', 'dot', 'angle', 'grad_centr', 'reverse', 'coordconv',
                             'stream', 'wait_stream']
    mandatory_options = ['activation', 'filters', 'size', 'stride']

    _check_not_supported_options(layer, not_supported_options)
    _check_mandatory_options(layer, mandatory_options)

    activation = layer['activation']
    out_channels = int(layer['filters'])
    kernel_size = int(layer['size'])
    stride = int(layer['stride'])
    padding = 0
    if 'pad' in layer and int(layer['pad']) == 1:
        padding = kernel_size // 2
    has_batch_norm = 'batch_normalize' in layer and int(layer['batch_normalize']) == 1

    groups = 1 if 'groups' not in layer else int(layer['groups'])

    python_file.write('        self._conv{} = nn.Sequential(\n'.format(i))
    python_file.write('            nn.Conv2d({}, {}, {}, stride={}, padding={}, bias={}, groups={}),\n'
                      .format(in_channels, out_channels, kernel_size, stride, padding, not has_batch_norm, groups))
    if has_batch_norm:
        python_file.write('            nn.BatchNorm2d({}),\n'.format(out_channels))
    if activation != 'linear':
        python_file.write('            {}\n'.format(ACTIVATION_MODULES_BY_NAME[activation]))
    python_file.write('        )')

    return out_channels, stride


def _write_init_upsample(python_file, i, layer, in_channels):
    not_supported_options = ['scale']
    mandatory_options = ['stride']

    _check_not_supported_options(layer, not_supported_options)
    _check_mandatory_options(layer, mandatory_options)

    stride = int(layer['stride'])
    python_file.write('        self._upsample{} = nn.Upsample(scale_factor={}, mode=\'nearest\')'.format(i, stride))
    return in_channels, stride


def _write_init_maxpool(python_file, i, layer, in_channels):
    not_supported_options = ['stride_x', 'stride_y', 'padding', 'maxpool_depth', 'out_channels', 'antialiasing']
    mandatory_options = ['size', 'stride']

    _check_not_supported_options(layer, not_supported_options)
    _check_mandatory_options(layer, mandatory_options)

    kernel_size = int(layer['size'])
    stride = int(layer['stride'])

    padding = 0
    if stride == 1:
        padding = kernel_size // 2
    python_file.write('        self._max_pool{} = nn.MaxPool2d({}, stride={}, padding={})'
                      .format(i, kernel_size, stride, padding))

    return in_channels, stride


def _write_init_route(layer, in_channels, cumulated_strides):
    not_supported_options = ['stream', 'wait_stream']
    mandatory_options = ['layers']

    _check_not_supported_options(layer, not_supported_options)
    _check_mandatory_options(layer, mandatory_options)

    if 'layers' not in layer:
        raise ValueError('A "route" block must contain the "layers" attribute (line={}).'.format(layer['line_no']))
    layers = [int(l.strip()) for l in layer['layers'].split(',')]
    out_channels = 0
    for l in layers:
        out_channels += in_channels[l]

    if 'groups' in layer and 'group_id' in layer:
        out_channels //= int(layer['groups'])

    return out_channels, cumulated_strides[layers[0]]


def _write_init_shortcut(layer, in_channels, cumulated_strides):
    not_supported_options = ['weights_type', 'weights_normalization']
    mandatory_options = ['from', 'activation']

    _check_not_supported_options(layer, not_supported_options)
    _check_mandatory_options(layer, mandatory_options)

    if layer['activation'] != 'linear':
        raise ValueError('Not supported activation (line={}).'.format(layer['line_no']))

    from_ = int(layer['from'])
    if in_channels[-1] != in_channels[from_]:
        raise ValueError('The channel counts must be equal (line={}).'.format(layer['line_no']))

    return in_channels[-1], cumulated_strides[-1]


def _write_init_yolo(python_file, i, layer, cumulated_stride):
    not_supported_options = ['show_details', 'counters_per_class', 'label_smooth_eps',
                             'objectness_smooth', 'new_coords', 'focal_loss',
                             'track_history_size', 'sim_thresh', 'dets_for_track', 'dets_for_show',
                             'track_ciou_norm', 'embedding_layer']
    mandatory_options = ['mask', 'anchors', 'classes']

    _check_not_supported_options(layer, not_supported_options)
    _check_mandatory_options(layer, mandatory_options)

    masks = [int(index.strip()) for index in layer['mask'].split(',')]
    anchors = [int(size.strip()) for size in layer['anchors'].split(',')]
    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
    anchors = [anchors[i] for i in masks]
    class_count = int(layer['classes'])
    scale_x_y = 1 if 'scale_x_y' not in layer else layer['scale_x_y']

    python_file.write('        self._anchors.append(np.array({}))\n'.format(anchors))
    python_file.write('        self._output_strides.append({})\n'.format(cumulated_stride))
    python_file.write('        self._yolo{} = YoloV4Layer({}, {}, {}, {}, scale_x_y={}, class_probs=class_probs)'
                      .format(i, 'IMAGE_SIZE', cumulated_stride, 'self._anchors[-1].tolist()', class_count,
                              scale_x_y))
    return 0


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
    output_names = []
    yolo_outputs = []
    for i, layer in enumerate(layers):
        input_name = 'x' if len(output_names) == 0 else output_names[-1]
        output_name = 'y{}'.format(i)
        if layer['type'] == 'convolutional':
            python_file.write('        {} = self._conv{}({})'.format(output_name, i, input_name))
        elif layer['type'] == 'upsample':
            python_file.write('        {} = self._upsample{}({})'.format(output_name, i, input_name))
        elif layer['type'] == 'maxpool':
            python_file.write('        {} = self._max_pool{}({})'.format(output_name, i, input_name))
        elif layer['type'] == 'yolo':
            python_file.write('        {} = self._yolo{}({})\n'.format(output_name, i, input_name))
            yolo_outputs.append(output_name)
        elif layer['type'] == 'route':
            output_name = _write_forward_route(python_file, i, layer, output_names, output_name)
        elif layer['type'] == 'shortcut':
            from_ = int(layer['from'])
            python_file.write('        {} = {} + {}\n'.format(output_name, input_name, output_names[from_]))
        else:
            raise ValueError('Not supported layer (type={})'.format(layer['type']))
        output_names.append(output_name)
        python_file.write('\n')

    python_file.write('        return {}\n\n'.format(str(yolo_outputs).replace('\'', '')))


def _write_forward_route(python_file, i, layer, output_names, output_name):
    routes = [int(l.strip()) for l in layer['layers'].split(',')]
    route_outputs = [output_names[i] for i in routes]

    if len(route_outputs) == 1 and 'groups' in layer and 'group_id' in layer:
        groups = int(layer['groups'])
        group_id = int(layer['group_id'])

        python_file.write('        C = {}.size()[1]\n'.format(route_outputs[0]))
        python_file.write('        {} = {}[:, C // {} * {}:C // {} * ({} + 1), :, :]\n'
                          .format(output_name, route_outputs[0], groups, group_id, groups, group_id))
    elif len(route_outputs) == 1:
        output_name = route_outputs[0]
    elif 'groups' not in layer and 'group_id' not in layer:
        route_outputs = str(route_outputs).replace('\'', '')
        python_file.write('        {} = torch.cat({}, dim=1)\n'.format(output_name, route_outputs))
    else:
        raise ValueError('Invalid route (line={})'.format(layer['line_no']))
    return output_name


def _write_load_weights(python_file, layers):
    python_file.write('    def load_weights(self, weights_file_path):\n')
    python_file.write('        with open(weights_file_path, \'r\') as weights_file:\n')
    python_file.write('            header1 = np.fromfile(weights_file, dtype=np.int32, count=3)\n')
    python_file.write('            header2 = np.fromfile(weights_file, dtype=np.int64, count=1)\n')
    python_file.write('            weights = np.fromfile(weights_file, dtype=np.float32)\n')
    python_file.write('\n')
    python_file.write('        print(\'load_weights - Major version:\', header1[0])\n')
    python_file.write('        print(\'load_weights - Minor version:\', header1[1])\n')
    python_file.write('        print(\'load_weights - Subversion:\', header1[2])\n')
    python_file.write('        print(\'load_weights - # images:\', header2[0])\n')
    python_file.write('\n')
    python_file.write('        offset = 0\n')

    for i, layer in enumerate(layers):
        if layer['type'] == 'convolutional':
            if 'batch_normalize' in layer and int(layer['batch_normalize']) == 1:
                python_file.write('        offset = self._load_batch_norm_conv_weights('
                                  'self._conv{}, weights, offset)\n'.format(i))
            else:
                python_file.write('        offset = self._load_conv_weights('
                                  'self._conv{}, weights, offset)\n'.format(i))

        elif layer['type'] == 'upsample' or layer['type'] == 'maxpool' or layer['type'] == 'yolo' or \
                layer['type'] == 'route' or layer['type'] == 'shortcut':
            continue
        else:
            raise ValueError('Not supported layer (type={})'.format(layer['type']))

    python_file.write('\n')
    python_file.write('        if offset != weights.size:\n')
    python_file.write('            raise ValueError(\'Invalid weights file.\')\n\n')


def _write_load_batch_norm_conv_weights(python_file):
    python_file.write('    def _load_batch_norm_conv_weights(self, conv, weights, offset):\n')
    python_file.write('        n = conv[1].bias.numel()\n')
    python_file.write('        bias_data = torch.from_numpy(weights[offset:offset + n])\n')
    python_file.write('        conv[1].bias.data.copy_(bias_data.view_as(conv[1].bias.data))\n')
    python_file.write('        offset += n\n')
    python_file.write('\n')
    python_file.write('        weight_data = torch.from_numpy(weights[offset:offset + n])\n')
    python_file.write('        conv[1].weight.data.copy_(weight_data.view_as(conv[1].weight.data))\n')
    python_file.write('        offset += n\n')
    python_file.write('\n')
    python_file.write('        running_mean_data = torch.from_numpy(weights[offset:offset + n])\n')
    python_file.write('        conv[1].running_mean.data.copy_(running_mean_data.view_as('
                      'conv[1].running_mean.data))\n')
    python_file.write('        offset += n\n')
    python_file.write('\n')
    python_file.write('        running_var_data = torch.from_numpy(weights[offset:offset + n])\n')
    python_file.write('        conv[1].running_var.data.copy_(running_var_data.view_as('
                      'conv[1].running_var.data))\n')
    python_file.write('        offset += n\n')
    python_file.write('\n')
    python_file.write('        n = conv[0].weight.numel()\n')
    python_file.write('        weight_data = torch.from_numpy(weights[offset:offset + n])\n')
    python_file.write('        conv[0].weight.data.copy_(weight_data.view_as(conv[0].weight.data))\n')
    python_file.write('        offset += n\n')
    python_file.write('\n')
    python_file.write('        return offset\n\n')


def _write_load_conv_weights(python_file):
    python_file.write('    def _load_conv_weights(self, conv, weights, offset):\n')
    python_file.write('        n = conv[0].bias.numel()\n')
    python_file.write('        bias_data = torch.from_numpy(weights[offset:offset + n])\n')
    python_file.write('        conv[0].bias.data.copy_(bias_data.view_as(conv[0].bias.data))\n')
    python_file.write('        offset += n\n')
    python_file.write('\n')
    python_file.write('        n = conv[0].weight.numel()\n')
    python_file.write('        weight_data = torch.from_numpy(weights[offset:offset + n])\n')
    python_file.write('        conv[0].weight.data.copy_(weight_data.view_as(conv[0].weight.data))\n')
    python_file.write('        offset += n\n')
    python_file.write('\n')
    python_file.write('        return offset\n')


def _check_not_supported_options(layer, not_supported_options):
    for option in not_supported_options:
        if option in layer:
            raise NotImplementedError('The "{}" option in "{}" layers is not implemented (line={})'
                                      .format(option, layer['type'], layer['line_no']))


def _check_mandatory_options(layer, mandatory_options):
    for option in mandatory_options:
        if option not in layer:
            raise ValueError('A "{}" block must contain the "activation" attribute (line={}).'
                             .format(layer['type'], layer['line_no']))


def main():
    parser = argparse.ArgumentParser(description='Convert the specified darknet configuration file to PyTorch')
    parser.add_argument('--cfg_path', type=str, help='Choose the configuration file', required=True)
    parser.add_argument('--python_output_path', type=str, help='Choose the Python output file', required=True)
    parser.add_argument('--class_name', type=str, help='Choose the class name', required=True)

    args = parser.parse_args()
    convert(args.cfg_path, args.python_output_path, args.class_name)


if __name__ == '__main__':
    main()
