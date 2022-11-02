import torch
import torch.nn as nn
import torch.nn.functional as F


class OhemCrossEntropyLoss(nn.Module):
    def __init__(self, probability_threshold=0.7, min_ratio=0.0625):
        super(OhemCrossEntropyLoss, self).__init__()
        self._probability_threshold = probability_threshold
        self._min_ratio = min_ratio

    def forward(self, input, target):
        cross_entropy_losses = F.cross_entropy(input, target, reduction='none').view(-1)
        probabilities = torch.exp(-cross_entropy_losses)
        probability_mask = probabilities >= self._probability_threshold
        min_count = max(1, int(cross_entropy_losses.size(0) * self._min_ratio))

        if probability_mask.sum() <= min_count:
            sorted_losses, _ = torch.sort(cross_entropy_losses, descending=True)
            return sorted_losses[:min_count].mean()
        else:
            return cross_entropy_losses[probability_mask].mean()
