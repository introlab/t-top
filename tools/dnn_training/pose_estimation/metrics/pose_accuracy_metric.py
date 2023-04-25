import numpy as np

import torch
import torch.nn.functional as F

from pose_estimation.pose_estimator import get_coordinates

OKS_K = torch.tensor([0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087,
                      0.087, 0.089, 0.089])
DEFAULT_THRESHOLD = 0.5


# Accuracy using Object Keypoint Similarity (OKS)
class PoseAccuracyMetric:
    def __init__(self, threshold=DEFAULT_THRESHOLD):
        self._threshold = threshold
        self._true_positive_count = 0
        self._false_positive_count = 0
        self._true_negative_count = 0
        self._false_negative_count = 0

    def clear(self):
        self._true_positive_count = 0
        self._false_positive_count = 0
        self._true_negative_count = 0
        self._false_negative_count = 0

    def add(self, heatmap_prediction, heatmap_target, presence_target, oks_scale):
        heatmap_prediction = F.interpolate(heatmap_prediction, (heatmap_target.size()[2], heatmap_target.size()[3]))

        predicted_coordinates, presence_prediction = get_coordinates(heatmap_prediction)
        target_coordinates, _ = get_coordinates(heatmap_target)

        true_negative = (presence_target != 1) & (presence_prediction < self._threshold)
        false_positive = (presence_target != 1) & (presence_prediction >= self._threshold)
        false_negative = (presence_target == 1) & (presence_prediction < self._threshold)

        self._true_negative_count += true_negative.sum().item()
        self._false_positive_count += false_positive.sum().item()
        self._false_negative_count += false_negative.sum().item()

        remained_indexes = ~(true_negative | false_positive | false_negative)

        k = OKS_K.to(oks_scale.device)
        oks = _calculate_oks(predicted_coordinates, target_coordinates, oks_scale, k)[remained_indexes]
        true_positive = oks > self._threshold
        self._true_positive_count += true_positive.sum().item()
        self._false_positive_count += (~true_positive).sum().item()

    def get_accuracy(self):
        good = self._true_positive_count + self._true_negative_count
        total = good + self._false_positive_count + self._false_negative_count
        if total == 0:
            return 1.0
        return good / total

    def get_precision(self):
        total = self._true_positive_count + self._false_positive_count
        if total == 0:
            return 1.0
        return self._true_positive_count / total

    def get_recall(self):
        total = self._true_positive_count + self._false_negative_count
        if total == 0:
            return 1.0
        return self._true_positive_count / total


def _calculate_oks(p1, p2, oks_scale, k):
    d = torch.pow(p1 - p2, 2).sum(dim=2)
    k2 = k * k
    oks_scale = (oks_scale + np.spacing(1))

    return torch.exp(-d.permute(1, 0).div(oks_scale).permute(1, 0).div(k2) / 2)
