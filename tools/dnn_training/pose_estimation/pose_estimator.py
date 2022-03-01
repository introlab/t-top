import torch
import torch.nn as nn


class PoseEstimator(nn.Module):
    def __init__(self, backbone, keypoint_count=17, upsampling_count=3):
        super(PoseEstimator, self).__init__()
        self._backbone = backbone
        self._heatmap_layers = self._create_heatmap_layers(self._backbone.last_channel_count(),
                                                           keypoint_count,
                                                           upsampling_count)

    def forward(self, x):
        features = self._backbone(x)
        heatmaps = self._heatmap_layers(features)

        return heatmaps

    def _create_heatmap_layers(self, last_channel_count, keypoint_count, upsampling_count):
        layers = []
        in_channels = last_channel_count
        for _ in range(upsampling_count):
            layers.append(nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))
            layers.append(nn.BatchNorm2d(256))
            layers.append(nn.ReLU(inplace=True))
            in_channels = 256

        layers.append(nn.Conv2d(
            in_channels=256,
            out_channels=keypoint_count,
            kernel_size=1,
            stride=1,
            padding=0))

        return nn.Sequential(*layers)

import time

class Stopwatch:
    def __init__(self, prefix):
        self._prefix = 'training/pose_estimator.py - ' + prefix

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, *args):
        print(self._prefix + ' - elapsed time', time.time() - self._start)


def get_coordinates(heatmaps):
    height = heatmaps.size()[2]
    width = heatmaps.size()[3]

    with Stopwatch('init'):
        coordinates = torch.zeros((heatmaps.size()[0], heatmaps.size()[1], 2), device=heatmaps.device)
        heatmaps = heatmaps.reshape(heatmaps.size()[0], heatmaps.size()[1], heatmaps.size()[2] * heatmaps.size()[3])
        heatmaps = heatmaps.permute(2, 0, 1)

    with Stopwatch('argmax'):
        indexes = torch.argmax(heatmaps, dim=0)
        presence = heatmaps.gather(0, indexes.unsqueeze(0)).squeeze(0)
        heatmaps = heatmaps.permute(1, 2, 0)

    with Stopwatch('unravel_index'):
        y, x = unravel_index(indexes.flatten(), (height, width))
        y = y.reshape((heatmaps.size()[0], heatmaps.size()[1]))
        x = x.reshape((heatmaps.size()[0], heatmaps.size()[1]))

    with Stopwatch('for'):
        for i in range(heatmaps.size()[1]):
            coordinates[:, i, 0] = x[:, i]
            coordinates[:, i, 1] = y[:, i]

    return coordinates, presence


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))
