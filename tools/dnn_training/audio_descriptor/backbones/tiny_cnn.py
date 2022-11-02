import torch.nn as nn

from common.modules import DepthWiseSeparableConv2d


class TinyCnn(nn.Module):
    def __init__(self):
        super(TinyCnn, self).__init__()

        self._layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 11), stride=2, padding=(2, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            DepthWiseSeparableConv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),

            DepthWiseSeparableConv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),

            DepthWiseSeparableConv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )

    def forward(self, x):
        return self._layers(x)

    def last_channel_count(self):
        return 256
