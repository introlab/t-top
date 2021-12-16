import torch.nn as nn

import torchvision.models as models


class _Mnasnet(nn.Module):
    def __init__(self, mnasnet):
        super(_Mnasnet, self).__init__()
        layers = list(mnasnet.layers)
        conv1 = nn.Conv2d(1, layers[0].out_channels, layers[0].kernel_size, stride=layers[0].stride,
                          padding=layers[0].padding, dilation=layers[0].dilation, groups=layers[0].groups,
                          bias=layers[0].bias, padding_mode=layers[0].padding_mode)

        self._mnasnet_feature_extractor = nn.Sequential(conv1, *layers[1:])

    def forward(self, x):
        return self._mnasnet_feature_extractor(x)

    def last_channel_count(self):
        return 1280


class Mnasnet0_5(_Mnasnet):
    def __init__(self, pretrained=False):
        super(Mnasnet0_5, self).__init__(models.mnasnet0_5(pretrained=pretrained))


class Mnasnet1_0(_Mnasnet):
    def __init__(self, pretrained=False):
        super(Mnasnet1_0, self).__init__(models.mnasnet1_0(pretrained=pretrained))
