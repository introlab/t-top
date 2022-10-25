import torch
import torch.nn as nn
import torch.nn.functional as F


def global_avg_pool_2d(x):
    return torch.mean(x, dim=(2, 3), keepdim=True)


class GlobalAvgPool2d(nn.Module):
    def forward(self, x):
        return global_avg_pool_2d(x)


class GlobalHeightAvgPool2d(nn.Module):
    def forward(self, x):
        size = (x.data.size()[2], 1)
        return F.avg_pool2d(x, size, stride=size)
