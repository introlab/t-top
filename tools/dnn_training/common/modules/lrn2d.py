import torch
import torch.nn as nn
import torch.nn.functional as F


class Lrn2d(nn.Module):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1.):
        super(Lrn2d, self).__init__()
        self._size = size
        self._alpha = alpha
        self._beta = beta
        self._k = k

    def forward(self, x):
        N = x.size()[0]
        C = x.size()[1]
        H = x.size()[2]
        W = x.size()[3]

        x = x.view(N, C, H * W)
        y = self._local_response_norm_1d(x)

        return y.view(N, C, H, W)

    def _local_response_norm_1d(self, x):
        div = (x * x).unsqueeze(1)
        div = F.pad(div, [0, 0, self._size // 2, (self._size - 1) // 2])
        div = F.avg_pool2d(div, (self._size, 1), stride=1).squeeze(1)
        div = torch.pow(div * self._alpha + self._k, self._beta)
        return x / div
