import torch.nn as nn
import torch.nn.functional as F


class PoseEstimationLoss(nn.Module):
    def __init__(self):
        super(PoseEstimationLoss, self).__init__()

    def forward(self, heatmap_prediction, target):
        heatmap_target, presence_target, _ = target

        heatmap_target = F.interpolate(heatmap_target, (heatmap_prediction.size()[2], heatmap_prediction.size()[3]))

        loss = ((heatmap_prediction - heatmap_target) ** 2 / (1 - heatmap_target + 0.005)).mean()

        return loss
