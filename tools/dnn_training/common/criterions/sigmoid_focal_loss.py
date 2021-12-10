import torch
import torch.nn as nn
import torch.nn.functional as F


# Inspired by https://pytorch.org/vision/0.9/_modules/torchvision/ops/focal_loss.html#sigmoid_focal_loss
class SigmoidFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(SigmoidFocalLossWithLogits, self).__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self, prediction, target):
        p = torch.sigmoid(prediction)
        ce_loss = F.binary_cross_entropy_with_logits(prediction, target, reduction='none')
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self._gamma)

        if self._alpha >= 0:
            alpha_t = self._alpha * target + (1 - self._alpha) * (1 - target)
            loss = alpha_t * loss

        return loss.mean()


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(SigmoidFocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self, prediction, target):
        p = prediction
        ce_loss = F.binary_cross_entropy(prediction, target, reduction='none')
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self._gamma)

        if self._alpha >= 0:
            alpha_t = self._alpha * target + (1 - self._alpha) * (1 - target)
            loss = alpha_t * loss

        return loss.mean()
