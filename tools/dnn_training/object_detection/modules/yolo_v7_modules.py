import torch
import torch.nn as nn


class YoloV7SPPCSPC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YoloV7SPPCSPC, self).__init__()

        self._conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.SiLU()
        )
        self._conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.SiLU()
        )
        self._conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.SiLU()
        )
        self._conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.SiLU()
        )

        self._max_pool0 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self._max_pool1 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self._max_pool2 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

        self._conv4 = nn.Sequential(
            nn.Conv2d(4 * out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.SiLU()
        )
        self._conv5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.SiLU()
        )
        self._conv6 = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.SiLU()
        )

    def forward(self, x0):
        x1 = self._conv3(self._conv2(self._conv0(x0)))
        x2 = torch.cat([x1, self._max_pool0(x1), self._max_pool1(x1), self._max_pool2(x1)], dim=1)
        y1 = self._conv5(self._conv4(x2))
        y2 = self._conv1(x0)
        return self._conv6(torch.cat([y1, y2], dim=1))


# TODO use only one Layer
class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, activation):
        super(RepConv, self).__init__()

        if padding == None:
            padding = kernel_size // 2
        padding_1x1 = padding - kernel_size // 2

        self._identity = nn.BatchNorm2d(in_channels) if in_channels == out_channels and stride == 1 else None
        self._conv_kxk = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001)
        )

        self._conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding_1x1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001)
        )

        self._activation = activation

    def forward(self, x):
        if self._identity is None:
            return self._activation(self._conv_kxk(x) + self._conv_1x1(x))
        else:
            return self._activation(self._conv_kxk(x) + self._conv_1x1(x) + self._identity(x))
