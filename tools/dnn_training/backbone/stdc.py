import torch
import torch.nn as nn


# Inspired by https://github.com/MichaelFan01/STDC-Seg
class ConvBatchNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBatchNormReLU, self).__init__()

        padding = kernel_size // 2
        self._layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self._layers(x)


class CatBottleneckStride1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CatBottleneckStride1, self).__init__()

        self._conv1 = ConvBatchNormReLU(in_channels, out_channels // 2, kernel_size=1, stride=1)
        self._conv2 = ConvBatchNormReLU(out_channels // 2, out_channels // 4, kernel_size=3, stride=1)
        self._conv3 = ConvBatchNormReLU(out_channels // 4, out_channels // 8, kernel_size=3, stride=1)
        self._conv4 = ConvBatchNormReLU(out_channels // 8, out_channels // 8, kernel_size=3, stride=1)

    def forward(self, x):
        y1 = self._conv1(x)
        y2 = self._conv2(y1)
        y3 = self._conv3(y2)
        y4 = self._conv4(y3)

        return torch.cat([y1, y2, y3, y4], dim=1)


class CatBottleneckStride2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CatBottleneckStride2, self).__init__()

        self._conv1 = ConvBatchNormReLU(in_channels, out_channels // 2, kernel_size=1, stride=1)
        self._conv2 = ConvBatchNormReLU(out_channels // 2, out_channels // 4, kernel_size=3, stride=1)
        self._conv3 = ConvBatchNormReLU(out_channels // 4, out_channels // 8, kernel_size=3, stride=1)
        self._conv4 = ConvBatchNormReLU(out_channels // 8, out_channels // 8, kernel_size=3, stride=1)
        self._avd = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2,
                      kernel_size=3, stride=2, padding=1, groups=out_channels // 2, bias=False),
            nn.BatchNorm2d(out_channels // 2),
        )

        self._avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y1 = self._conv1(x)
        y2 = self._conv2(self._avd(y1))
        y3 = self._conv3(y2)
        y4 = self._conv4(y3)

        return torch.cat([self._avg_pool(y1), y2, y3, y4], dim=1)


class Stdc(nn.Module):
    def __init__(self, class_count=1000, dropout=0.20):
        super(Stdc, self).__init__()

        self._features_1 = ConvBatchNormReLU(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self._features_2 = ConvBatchNormReLU(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self._features_3 = self._create_features_3()
        self._features_4 = self._create_features_4()
        self._features_5 = self._create_features_5()
        self._features_6 = ConvBatchNormReLU(in_channels=1024, out_channels=1024, kernel_size=1, stride=1)

        self._global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self._classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024, bias=False),
            nn.ReLU(),

            nn.Dropout(p=dropout),
            nn.Linear(in_features=1024, out_features=class_count, bias=False),
        )

    def _create_features_3(self):
        raise NotImplementedError()

    def _create_features_4(self):
        raise NotImplementedError()

    def _create_features_5(self):
        raise NotImplementedError()

    def forward(self, x):
        _, _, _, _, y5 = self.forward_features(x)
        y6 = self._features_6(y5).pow(2)
        y7 = self._global_avg_pool(y6)
        return self._classifier(y7.view(y7.size(0), -1))

    def forward_features(self, x):
        y1 = self._features_1(x)
        y2 = self._features_2(y1)
        y3 = self._features_3(y2)
        y4 = self._features_4(y3)
        y5 = self._features_5(y4)

        return y1, y2, y3, y4, y5


class Stdc1(Stdc):
    def _create_features_3(self):
        return nn.Sequential(
            CatBottleneckStride2(in_channels=64, out_channels=256),
            CatBottleneckStride1(in_channels=256, out_channels=256)
        )

    def _create_features_4(self):
        return nn.Sequential(
            CatBottleneckStride2(in_channels=256, out_channels=512),
            CatBottleneckStride1(in_channels=512, out_channels=512),
        )

    def _create_features_5(self):
        return nn.Sequential(
            CatBottleneckStride2(in_channels=512, out_channels=1024),
            CatBottleneckStride1(in_channels=1024, out_channels=1024),
        )


class Stdc2(Stdc):
    def _create_features_3(self):
        return nn.Sequential(
            CatBottleneckStride2(in_channels=64, out_channels=256),
            CatBottleneckStride1(in_channels=256, out_channels=256),
            CatBottleneckStride1(in_channels=256, out_channels=256),
            CatBottleneckStride1(in_channels=256, out_channels=256)
        )

    def _create_features_4(self):
        return nn.Sequential(
            CatBottleneckStride2(in_channels=256, out_channels=512),
            CatBottleneckStride1(in_channels=512, out_channels=512),
            CatBottleneckStride1(in_channels=512, out_channels=512),
            CatBottleneckStride1(in_channels=512, out_channels=512),
            CatBottleneckStride1(in_channels=512, out_channels=512)
        )

    def _create_features_5(self):
        return nn.Sequential(
            CatBottleneckStride2(in_channels=512, out_channels=1024),
            CatBottleneckStride1(in_channels=1024, out_channels=1024),
            CatBottleneckStride1(in_channels=1024, out_channels=1024)
        )
