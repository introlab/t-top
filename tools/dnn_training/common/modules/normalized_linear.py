import torch.nn as nn
import torch.nn.functional as F


class NormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormalizedLinear, self).__init__()
        self._weight = nn.Linear(in_features, out_features, bias=False).weight

    def forward(self, x):
        normalized_weight = F.normalize(self._weight, dim=1)
        return F.linear(F.normalize(x, dim=1), normalized_weight)
