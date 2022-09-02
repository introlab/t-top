import torch.nn as nn
import torch.nn.functional as F


class GlobalAvgPool2d(nn.Module):
    def forward(self, x):
        size = (x.data.size()[2], x.data.size()[3])
        return F.avg_pool2d(x, size, stride=size)


class GlobalHeightAvgPool2d(nn.Module):
    def forward(self, x):
        size = (x.data.size()[2], 1)
        return F.avg_pool2d(x, size, stride=size)
