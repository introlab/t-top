import torch.nn as nn
import torch.nn.functional as F


class L2Normalization(nn.Module):
    def __init__(self):
        super(L2Normalization, self).__init__()

    def forward(self, x):
        return F.normalize(x, dim=1, p=2.0)
