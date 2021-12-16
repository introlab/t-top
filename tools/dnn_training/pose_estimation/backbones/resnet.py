import torch.nn as nn

import torchvision.models as models


class _Resnet(nn.Module):
    def __init__(self, resnet):
        super(_Resnet, self).__init__()
        self._resnet_feature_extractor = nn.Sequential(
            resnet.conv1,
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
