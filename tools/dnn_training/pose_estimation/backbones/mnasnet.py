import torch.nn as nn

import torchvision.models as models


class _Mnasnet(nn.Module):
    def __init__(self, mnasnet):
        super(_Mnasnet, self).__init__()
        self._mnasnet_feature_extractor = mnasnet.layers

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
