import torch
import torch.nn as nn
import torch.nn.functional as F


# Self-attentive pooling layer
# Inspired by https://github.com/clovaai/voxceleb_trainer/blob/master/models/ResNetSE34L.py#L96
class SAP(nn.Module):
    def __init__(self, channel_count):
        super(SAP, self).__init__()
        self._attention = nn.Sequential(
            nn.Linear(channel_count, channel_count),
            nn.Tanh(),
            nn.Linear(channel_count, 1, bias=False)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).squeeze(3)
        w = F.softmax(self._attention(x).squeeze(2), dim=1).unsqueeze(2)
        return torch.sum(x * w, dim=1)

