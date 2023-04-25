import torch.nn as nn
import torch.nn.functional as F


class PoseEstimationLoss(nn.Module):
    def __init__(self):
        super(PoseEstimationLoss, self).__init__()

    def forward(self, heatmap_prediction, target):
        heatmap_target, presence_target, _ = target

        heatmap_target = F.interpolate(heatmap_target, (heatmap_prediction.size()[2], heatmap_prediction.size()[3]))

        return F.binary_cross_entropy(heatmap_prediction, heatmap_target)


class PoseEstimationDistillationLoss(nn.Module):
    def __init__(self, alpha=0.25):
        super(PoseEstimationDistillationLoss, self).__init__()
        self._alpha = alpha

    def forward(self, student_heatmap_prediction, target, teacher_heatmap_prediction):
        heatmap_target, presence_target, _ = target

        heatmap_target = F.interpolate(heatmap_target,
                                       (student_heatmap_prediction.size()[2], student_heatmap_prediction.size()[3]))
        teacher_heatmap_prediction = F.interpolate(
            teacher_heatmap_prediction,
            (student_heatmap_prediction.size()[2], student_heatmap_prediction.size()[3])
        )

        target_loss = F.binary_cross_entropy(student_heatmap_prediction, heatmap_target)
        teacher_loss = F.binary_cross_entropy(student_heatmap_prediction, teacher_heatmap_prediction)
        return self._alpha * target_loss + (1 - self._alpha) * teacher_loss
