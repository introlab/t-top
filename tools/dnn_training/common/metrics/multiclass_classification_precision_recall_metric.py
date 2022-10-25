import torch


class MulticlassClassificationPrecisionRecallMetric:
    def __init__(self, threshold=0.5):
        self._threshold = threshold
        self._true_positive_count = 0
        self._false_positive_count = 0
        self._false_negative_count = 0

    def clear(self):
        self._true_positive_count = 0
        self._false_positive_count = 0
        self._false_negative_count = 0

    def add(self, predicted_class_scores, target_classes):
        predicted_classes = (torch.sigmoid(predicted_class_scores) > self._threshold).float()
        target_classes = (target_classes > self._threshold).float()

        self._true_positive_count += ((predicted_classes == 1.0) & (target_classes == 1.0)).sum().item()
        self._false_positive_count += ((predicted_classes == 1.0) & (target_classes == 0.0)).sum().item()
        self._false_negative_count += ((predicted_classes == 0.0) & (target_classes == 1.0)).sum().item()

    def get_precision(self):
        denominator = self._true_positive_count + self._false_positive_count
        if denominator == 0:
            return 1.0
        return self._true_positive_count / denominator

    def get_recall(self):
        denominator = self._true_positive_count + self._false_negative_count
        if denominator == 0:
            return 0.0
        return self._true_positive_count / denominator
