import torch
import torch.nn as nn


def global_avg_pool_1d(x):
    return torch.mean(x, dim=2, keepdim=True)


class GlobalAvgPool1d(nn.Module):
    def forward(self, x):
        return global_avg_pool_1d(x)
