import torch
import torch.nn as nn

from common.modules import InceptionModule, PaddedLPPool2d, Lrn2d


class TemporalAttention(nn.Module):
    def __init__(self, n_features, in_channels):
        super(TemporalAttention, self).__init__()

        self._feature_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels // 4, kernel_size=3, padding=1,
                      groups=in_channels // 4, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )

        self._attention_layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(n_features, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self._feature_layers(x)
        aggregated_features = torch.cat([torch.mean(features, dim=1, keepdim=True),
                                         torch.max(features, dim=1, keepdim=True)[0]], dim=1)
        attention = self._attention_layers(aggregated_features)
        return x * attention


# Based on OpenFace (https://cmusatyalab.github.io/openface/)
class OpenFaceInceptionWithTemporalAttention(nn.Module):
    def __init__(self, n_features):
        super(OpenFaceInceptionWithTemporalAttention, self).__init__()

        self._feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            TemporalAttention(n_features // 2, in_channels=64),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Lrn2d(size=5, alpha=0.0001, beta=0.75),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),
            TemporalAttention(n_features // 4, in_channels=192),

            Lrn2d(size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            InceptionModule(in_channels=192,
                            kernel_size=[3, 5],
                            kernel_stride=[1, 1],
                            out_channels=[128, 32],
                            reduce_size=[96, 16, 32, 64],
                            pool=nn.MaxPool2d(kernel_size=3, stride=1, padding=1)),
            InceptionModule(in_channels=256,
                            kernel_size=[3, 5],
                            kernel_stride=[1, 1],
                            out_channels=[128, 64],
                            reduce_size=[96, 32, 64, 64],
                            pool=PaddedLPPool2d(norm_type=2, kernel_size=3, stride=1, padding=1)),
            InceptionModule(in_channels=320,
                            kernel_size=[3, 5],
                            kernel_stride=[2, 2],
                            out_channels=[256, 64],
                            reduce_size=[128, 32, None, None],
                            pool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

            InceptionModule(in_channels=640,
                            kernel_size=[3, 5],
                            kernel_stride=[1, 1],
                            out_channels=[192, 64],
                            reduce_size=[96, 32, 128, 256],
                            pool=PaddedLPPool2d(norm_type=2, kernel_size=3, stride=1, padding=1)),
            InceptionModule(in_channels=640,
                            kernel_size=[3, 5],
                            kernel_stride=[2, 2],
                            out_channels=[256, 128],
                            reduce_size=[160, 64, None, None],
                            pool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

            InceptionModule(in_channels=1024,
                            kernel_size=[3],
                            kernel_stride=[1],
                            out_channels=[384],
                            reduce_size=[96, 96, 256],
                            pool=PaddedLPPool2d(norm_type=2, kernel_size=3, stride=1, padding=1)),
            InceptionModule(in_channels=736,
                            kernel_size=[3],
                            kernel_stride=[1],
                            out_channels=[384],
                            reduce_size=[96, 96, 256],
                            pool=nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        )

    def forward(self, x):
        return self._feature_extractor(x)

    def last_channel_count(self):
        return 736
