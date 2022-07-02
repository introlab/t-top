import torch
import torch.nn as nn
import torch.nn.functional as F


class OhemCrossEntropyLoss(nn.Module):
    def __init__(self, ratio=0.05):
        super(OhemCrossEntropyLoss, self).__init__()
        self._ratio = ratio

    def forward(self, input, target):
        losses = F.cross_entropy(input, target, reduction='none').view(-1)
        sorted_losses, _ = torch.sort(losses, descending=True)

        return sorted_losses[:max(1, int(sorted_losses.size(0) * self._ratio))].mean()
