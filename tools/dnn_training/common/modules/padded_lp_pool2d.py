import torch
import torch.nn as nn
import torch.nn.functional as F


# Adapted from torch.nn.functional.lp_pool2d
class PaddedLPPool2d(nn.Module):
    def __init__(self, norm_type, kernel_size, stride=None, padding=0, ceil_mode=False):
        super(PaddedLPPool2d, self).__init__()
        self._norm_type = float(norm_type)
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._ceil_mode = ceil_mode

        if isinstance(kernel_size, tuple) or isinstance(kernel_size, list):
            self._scale = kernel_size[0] * kernel_size[1]
        else:
            self._scale = kernel_size * kernel_size

    def forward(self, x):
        x = torch.pow(x, self._norm_type)
        x = F.avg_pool2d(x, self._kernel_size, self._stride, self._padding, self._ceil_mode)
        x = self._scale * x
        x = torch.pow(x, 1 / self._norm_type)
        return x
