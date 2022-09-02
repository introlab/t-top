import torch.nn as nn
import torch.nn.functional as F


class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBatchNorm, self).__init__()

        padding = kernel_size // 2
        self._layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self._layers(x)


class ConvBatchNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBatchNormReLU, self).__init__()

        padding = kernel_size // 2
        self._layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self._layers(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResnetBlock, self).__init__()
        if len(out_channels) != 3:
            raise ValueError('out_channels must contain 3 values.')

        self._max_pool = nn.MaxPool2d(kernel_size=stride)  # In the paper, they set the stride to the first conv
        self._layers = nn.Sequential(
            ConvBatchNormReLU(in_channels, out_channels[0], kernel_size=1),
            ConvBatchNormReLU(out_channels[0], out_channels[1], kernel_size),
            ConvBatchNorm(out_channels[1], out_channels[2], kernel_size=1)
        )
        self._residual_conv = ConvBatchNorm(in_channels, out_channels[2], kernel_size=1)

    def forward(self, x):
        x = self._max_pool(x)
        return F.relu(self._layers(x) + self._residual_conv(x))


class ResnetBlockIdentity(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResnetBlockIdentity, self).__init__()
        if len(out_channels) != 3:
            raise ValueError('out_channels must contain 3 values.')
        if in_channels != out_channels[2]:
            raise ValueError('in_channels must be equal to out_channels[2].')

        self._layers = nn.Sequential(
            ConvBatchNormReLU(in_channels, out_channels[0], kernel_size=1),
            ConvBatchNormReLU(out_channels[0], out_channels[1], kernel_size),
            ConvBatchNorm(out_channels[1], out_channels[2], kernel_size=1)
        )

    def forward(self, x):
        return F.relu(self._layers(x) + x)


class ThinResnet34(nn.Module):
    def __init__(self):
        super(ThinResnet34, self).__init__()

        self._layers = nn.Sequential(
            ConvBatchNormReLU(in_channels=1, out_channels=64, kernel_size=7),
            nn.MaxPool2d(kernel_size=2),

            ResnetBlock(in_channels=64, out_channels=[48, 48, 96], kernel_size=3, stride=1),
            ResnetBlockIdentity(in_channels=96, out_channels=[48, 48, 96], kernel_size=3),

            ResnetBlock(in_channels=96, out_channels=[96, 96, 128], kernel_size=3, stride=2),
            ResnetBlockIdentity(in_channels=128, out_channels=[96, 96, 128], kernel_size=3),
            ResnetBlockIdentity(in_channels=128, out_channels=[96, 96, 128], kernel_size=3),

            ResnetBlock(in_channels=128, out_channels=[128, 128, 256], kernel_size=3, stride=2),
            ResnetBlockIdentity(in_channels=256, out_channels=[128, 128, 256], kernel_size=3),
            ResnetBlockIdentity(in_channels=256, out_channels=[128, 128, 256], kernel_size=3),

            ResnetBlock(in_channels=256, out_channels=[256, 256, 512], kernel_size=3, stride=2),
            ResnetBlockIdentity(in_channels=512, out_channels=[256, 256, 512], kernel_size=3),
            ResnetBlockIdentity(in_channels=512, out_channels=[256, 256, 512], kernel_size=3),

            nn.MaxPool2d(kernel_size=(3, 1), stride=2),
        )

    def forward(self, x):
        return self._layers(x)

    def last_channel_count(self):
        return 512
