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


def get_coordinates(heatmaps):
    height = heatmaps.size()[2]
    width = heatmaps.size()[3]

    coordinates = torch.zeros((heatmaps.size()[0], heatmaps.size()[1], 2), device=heatmaps.device)
    heatmaps = heatmaps.reshape(heatmaps.size()[0], heatmaps.size()[1], heatmaps.size()[2] * heatmaps.size()[3])
    heatmaps = heatmaps.permute(2, 0, 1)

    indexes = torch.argmax(heatmaps, dim=0)
    presence = heatmaps.gather(0, indexes.unsqueeze(0)).squeeze(0)
    heatmaps = heatmaps.permute(1, 2, 0)

    y, x = unravel_index(indexes.flatten(), (height, width))
    y = y.reshape((heatmaps.size()[0], heatmaps.size()[1]))
    x = x.reshape((heatmaps.size()[0], heatmaps.size()[1]))

    heatmaps_indexes = torch.arange(heatmaps.size()[1], device=heatmaps.device)
    coordinates[:, heatmaps_indexes, 0] = x[:, heatmaps_indexes].float()
    coordinates[:, heatmaps_indexes, 1] = y[:, heatmaps_indexes].float()

    return coordinates, presence


# Based on https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987/2
def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))
