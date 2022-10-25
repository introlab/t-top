import torch


class MulticlassClassificationAccuracyMetric:
    def __init__(self, threshold=0.5):
        self._threshold = threshold
        self._good = 0
        self._total = 0

    def clear(self):
        self._good = 0
        self._total = 0

    def add(self, predicted_class_scores, target_classes):
        predicted_classes = (torch.sigmoid(predicted_class_scores) > self._threshold).float()
        target_classes = (target_classes > self._threshold).float()

        self._good += (predicted_classes == target_classes).sum().item()
        self._total += target_classes.numel()

    def get_accuracy(self):
        if self._total == 0:
            return 0
        return self._good / self._total
