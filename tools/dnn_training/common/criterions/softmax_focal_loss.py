import torch
import torch.nn as nn
import torch.nn.functional as F


# Inspired by https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/22
class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(SoftmaxFocalLoss, self).__init__()
        self._gamma = gamma
        self._weight = weight

    def forward(self, input, target):
        cross_entropy_losses = F.cross_entropy(input, target, reduction='none')
        weighted_cross_entropy_losses = F.cross_entropy(input, target, reduction='none')
        probabilities = torch.exp(-cross_entropy_losses)
        softmax_focal_losses = (1 - probabilities) ** self._gamma * weighted_cross_entropy_losses
        return softmax_focal_losses.mean()
