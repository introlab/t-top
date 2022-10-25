import torch.nn as nn

from common.modules import DepthWiseSeparableConv2d, GlobalAvgPool2d


# Based on : https://github.com/ARM-software/ML-KWS-for-MCU/tree/master
class KeywordSpotter(nn.Module):
    def __init__(self, class_count=2, use_softmax=True):
        super(KeywordSpotter, self).__init__()

        self._features_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 10), stride=2, padding=(2, 5), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            DepthWiseSeparableConv2d(in_channels=64, out_channels=64, kernel_size=3),
            DepthWiseSeparableConv2d(in_channels=64, out_channels=64, kernel_size=3),
            DepthWiseSeparableConv2d(in_channels=64, out_channels=64, kernel_size=3),
            DepthWiseSeparableConv2d(in_channels=64, out_channels=64, kernel_size=3)
        )

        self._global_avg_pool_2d = GlobalAvgPool2d()
        classifier_layers = [nn.Linear(64, class_count)]
        if use_softmax:
            classifier_layers.append(nn.Softmax(dim=1))

        self._classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        y0 = self._features_layers(x)
        y1 = self._global_avg_pool_2d(y0)[:, :, 0, 0]
        return self._classifier(y1)
