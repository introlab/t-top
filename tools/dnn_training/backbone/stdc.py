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
        self._conv2 = ConvBatchNormReLU(out_channels // 2, out_channels // 4, kernel_size=3, stride=2)
        self._conv3 = ConvBatchNormReLU(out_channels // 4, out_channels // 8, kernel_size=3, stride=1)
        self._conv4 = ConvBatchNormReLU(out_channels // 8, out_channels // 8, kernel_size=3, stride=1)

        self._avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y1 = self._conv1(x)
        y2 = self._conv2(y1)
        y3 = self._conv3(y2)
        y4 = self._conv4(y3)

        return torch.cat([self._avg_pool(y1), y2, y3, y4], dim=1)



class Stdc1(nn.Module):
    def __init__(self, class_count=1000, dropout=0.20):
        super(Stdc1, self).__init__()

        self._features_1 = ConvBatchNormReLU(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self._features_2 = ConvBatchNormReLU(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self._features_3 = nn.Sequential(
            CatBottleneckStride2(in_channels=64, out_channels=256),
            CatBottleneckStride1(in_channels=256, out_channels=256)
        )
        self._features_4 = nn.Sequential(
            CatBottleneckStride2(in_channels=256, out_channels=512),
            CatBottleneckStride1(in_channels=512, out_channels=512),
        )
        self._features_5 = nn.Sequential(
            CatBottleneckStride2(in_channels=512, out_channels=1024),
            CatBottleneckStride1(in_channels=1024, out_channels=1024),
        )

        self._features = nn.Sequential(
            self._features_1,  # ConvX1
            self._features_2,  # ConvX2
            self._features_3,  # Stage 3
            self._features_4,  # Stage 4
            self._features_5,  # Stage 5

            ConvBatchNormReLU(in_channels=1024, out_channels=1024, kernel_size=1, stride=1),  # ConvX6
        )

        self._global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self._classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),

            nn.Dropout(p=dropout),
            nn.Linear(in_features=1024, out_features=class_count),
        )

    def forward(self, x):
        y1 = self._features(x)
        y2 = self._global_avg_pool(y1)
        return self._classifier(y2.view(y2.size(0), -1))

    def forward_features(self, x):
        y1 = self._features_1(x)
        y2 = self._features_2(y1)
        y3 = self._features_3(y2)
        y4 = self._features_4(y3)
        y5 = self._features_5(y4)

        return y1, y2, y3, y4, y5


class Stdc2(nn.Module):
    def __init__(self, class_count=1000, dropout=0.20):
        super(Stdc2, self).__init__()

        self._features_1 = ConvBatchNormReLU(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self._features_2 = ConvBatchNormReLU(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self._features_3 = nn.Sequential(
            CatBottleneckStride2(in_channels=64, out_channels=256),
            CatBottleneckStride1(in_channels=256, out_channels=256),
            CatBottleneckStride1(in_channels=256, out_channels=256),
            CatBottleneckStride1(in_channels=256, out_channels=256)
        )
        self._features_4 = nn.Sequential(
            CatBottleneckStride2(in_channels=256, out_channels=512),
            CatBottleneckStride1(in_channels=512, out_channels=512),
            CatBottleneckStride1(in_channels=512, out_channels=512),
            CatBottleneckStride1(in_channels=512, out_channels=512),
            CatBottleneckStride1(in_channels=512, out_channels=512)
        )
        self._features_5 = nn.Sequential(
            CatBottleneckStride2(in_channels=512, out_channels=1024),
            CatBottleneckStride1(in_channels=1024, out_channels=1024),
            CatBottleneckStride1(in_channels=1024, out_channels=1024)
        )

        self._features = nn.Sequential(
            self._features_1,  # ConvX1
            self._features_2,  # ConvX2
            self._features_3,  # Stage 3
            self._features_4,  # Stage 4
            self._features_5,  # Stage 5

            ConvBatchNormReLU(in_channels=1024, out_channels=1024, kernel_size=1, stride=1),  # ConvX6
        )

        self._global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self._classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),

            nn.Dropout(p=dropout),
            nn.Linear(in_features=1024, out_features=class_count),
        )

    def forward(self, x):
        y1 = self._features(x)
        y2 = self._global_avg_pool(y1)
        return self._classifier(y2.view(y2.size(0), -1))

    def forward_features(self, x):
        y1 = self._features_1(x)
        y2 = self._features_2(y1)
        y3 = self._features_3(y2)
        y4 = self._features_4(y3)
        y5 = self._features_5(y4)

        return y1, y2, y3, y4, y5
