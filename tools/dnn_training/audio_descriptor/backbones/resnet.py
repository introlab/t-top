import torch.nn as nn

import torchvision.models as models


class _Resnet(nn.Module):
    def __init__(self, resnet):
        super(_Resnet, self).__init__()

        conv1 = nn.Conv2d(1, resnet.conv1.out_channels, resnet.conv1.kernel_size, stride=resnet.conv1.stride,
                          padding=resnet.conv1.padding, dilation=resnet.conv1.dilation, groups=resnet.conv1.groups,
                          bias=resnet.conv1.bias, padding_mode=resnet.conv1.padding_mode)
        self._resnet_feature_extractor = nn.Sequential(
            conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,

            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

    def forward(self, x):
        return self._resnet_feature_extractor(x)


class Resnet18(_Resnet):
    def __init__(self, pretrained=False):
        super(Resnet18, self).__init__(models.resnet18(pretrained=pretrained))

    def last_channel_count(self):
        return 512


class Resnet34(_Resnet):
    def __init__(self, pretrained=False):
        super(Resnet34, self).__init__(models.resnet34(pretrained=pretrained))

    def last_channel_count(self):
        return 512


class Resnet50(_Resnet):
    def __init__(self, pretrained=False):
        super(Resnet50, self).__init__(models.resnet50(pretrained=pretrained))

    def last_channel_count(self):
        return 2048


class Resnet101(_Resnet):
    def __init__(self, pretrained=False):
        super(Resnet101, self).__init__(models.resnet101(pretrained=pretrained))

    def last_channel_count(self):
        return 2048
