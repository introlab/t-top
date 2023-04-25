import torch
import torch.nn as nn

import torchvision.models as models


class EfficientNetPoseEstimator(nn.Module):
    SUPPORTED_BACKBONE_TYPES = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                                'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']
    HEATMAP_LAYER_CHANNELS_BY_BACKBONE_TYPE = {'efficientnet_b0': [320, 80, 40, 24],
                                               'efficientnet_b1': [320, 80, 40, 24],
                                               'efficientnet_b2': [352, 88, 48, 24],
                                               'efficientnet_b3': [384, 96, 48, 32],
                                               'efficientnet_b4': [448, 112, 56, 32],
                                               'efficientnet_b5': [512, 128, 64, 40],
                                               'efficientnet_b6': [576, 144, 72, 40],
                                               'efficientnet_b7': [640, 160, 80, 48]}

    def __init__(self, backbone_type, keypoint_count=17, pretrained_backbone=True):
        super(EfficientNetPoseEstimator, self).__init__()

        if pretrained_backbone:
            backbone_weights = 'DEFAULT'
        else:
            backbone_weights = None

        if (backbone_type not in self.SUPPORTED_BACKBONE_TYPES or
                backbone_type not in self.HEATMAP_LAYER_CHANNELS_BY_BACKBONE_TYPE):
            raise ValueError('Invalid backbone type')

        backbone_layers = list(models.__dict__[backbone_type](weights=backbone_weights).features)
        self._features_layers = nn.ModuleList(backbone_layers[:-1])
        self._heatmap_layers = self._create_heatmap_layers(self.HEATMAP_LAYER_CHANNELS_BY_BACKBONE_TYPE[backbone_type], keypoint_count)

    def _create_heatmap_layers(self, channels, keypoint_count):
        heatmap_layers = nn.ModuleList()
        for i in range(len(channels)):
            if i < len(channels) - 1:
                output_channels = channels[i + 1]
            else:
                output_channels = channels[i]

            heatmap_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=channels[i],
                        out_channels=channels[i],
                        kernel_size=2,
                        stride=2,
                        padding=0,
                        output_padding=0,
                        bias=False),
                    nn.BatchNorm2d(channels[i]),
                    nn.SiLU(inplace=True),

                    nn.Conv2d(in_channels=channels[i],
                              out_channels=output_channels,
                              kernel_size=3,
                              padding=1,
                              bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.SiLU(inplace=True),

                    nn.Conv2d(in_channels=output_channels,
                              out_channels=output_channels,
                              kernel_size=3,
                              padding=1,
                              bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.SiLU(inplace=True),
                )
            )

        heatmap_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=channels[-1],
                    out_channels=channels[-1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    output_padding=0,
                    bias=False),
                nn.BatchNorm2d(channels[-1]),
                nn.SiLU(inplace=True),

                nn.Conv2d(
                    in_channels=channels[-1],
                    out_channels=channels[-1],
                    kernel_size=3,
                    padding=1,
                    bias=False),
                nn.BatchNorm2d(channels[-1]),
                nn.SiLU(inplace=True),

                nn.Conv2d(in_channels=channels[-1],
                          out_channels=keypoint_count,
                          kernel_size=1,
                          padding=0,
                          bias=True),
                nn.Sigmoid()
            )
        )

        return heatmap_layers

    def forward(self, x0):
        x3, x4, x5, x8 = self._forward_features(x0)
        y4 = self._forward_heatmap(x3, x4, x5, x8)
        return y4

    def _forward_features(self, x0):
        assert len(self._features_layers) == 8
        x1 = self._features_layers[0](x0)
        x2 = self._features_layers[1](x1)
        x3 = self._features_layers[2](x2)
        x4 = self._features_layers[3](x3)
        x5 = self._features_layers[4](x4)
        x6 = self._features_layers[5](x5)
        x7 = self._features_layers[6](x6)
        x8 = self._features_layers[7](x7)

        return x3, x4, x5, x8

    def _forward_heatmap(self, x3, x4, x5, x8):
        y0 = self._heatmap_layers[0](x8)
        y1 = self._heatmap_layers[1](y0 + x5)
        y2 = self._heatmap_layers[2](y1 + x4)
        y3 = self._heatmap_layers[3](y2 + x3)
        y4 = self._heatmap_layers[4](y3)
        return y4



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
