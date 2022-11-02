from abc import ABC, abstractmethod

import torch


class _MeanIoUMetric(ABC):
    def __init__(self, class_count):
        self._class_count = class_count
        self._true_positive_count_by_class_index = None
        self._false_positive_count_by_class_index = None
        self._false_negative_count_by_class_index = None
        self.clear()

    def clear(self):
        self._true_positive_count_by_class_index = {class_index: 0 for class_index in range(self._class_count)}
        self._false_positive_count_by_class_index = {class_index: 0 for class_index in range(self._class_count)}
        self._false_negative_count_by_class_index = {class_index: 0 for class_index in range(self._class_count)}

    def add(self, predictions, targets):
        for n in range(predictions.size()[0]):
            self._add_image(predictions[n], targets[n])

    @abstractmethod
    def _add_image(self, prediction, target):
        pass

    def get_mean_iou(self):
        iou_by_class = self.get_iou_by_class()
        return sum(iou_by_class.values()) / len(iou_by_class)

    def get_iou_by_class(self):
        iou_by_class = {}
        for class_index in range(self._class_count):
            true_positive_count = self._true_positive_count_by_class_index[class_index]
            false_positive_count = self._false_positive_count_by_class_index[class_index]
            false_negative_count = self._false_negative_count_by_class_index[class_index]
            denominator = (true_positive_count + false_positive_count + false_negative_count)
            if denominator != 0:
                iou_by_class[class_index] = true_positive_count / denominator

        return iou_by_class


class MeanIoUMetric(_MeanIoUMetric):
    def _add_image(self, prediction, target):
        prediction = torch.argmax(prediction, dim=0)

        for class_index in range(self._class_count):
            true_positive_count = torch.logical_and(target == class_index, prediction == class_index).sum().item()
            false_positive_count = torch.logical_and(target != class_index, prediction == class_index).sum().item()
            false_negative_count = torch.logical_and(target == class_index, prediction != class_index).sum().item()

            self._true_positive_count_by_class_index[class_index] += true_positive_count
            self._false_positive_count_by_class_index[class_index] += false_positive_count
            self._false_negative_count_by_class_index[class_index] += false_negative_count


class MulticlassMeanIoUMetric(_MeanIoUMetric):
    def __init__(self, class_count, threshold=0.5):
        super(MulticlassMeanIoUMetric, self).__init__(class_count)
        self._threshold = threshold

    def _add_image(self, prediction, target):
        prediction = torch.sigmoid(prediction).view(self._class_count, -1) > self._threshold
        target = target.view(self._class_count, -1)

        for class_index in range(self._class_count):
            true_positive_count = torch.logical_and(target[class_index, :] == 1,
                                                    prediction[class_index, :] == 1).sum().item()
            false_positive_count = torch.logical_and(target[class_index, :] == 0,
                                                     prediction[class_index, :] == 1).sum().item()
            false_negative_count = torch.logical_and(target[class_index, :] == 1,
                                                     prediction[class_index, :] == 0).sum().item()

            self._true_positive_count_by_class_index[class_index] += true_positive_count
            self._false_positive_count_by_class_index[class_index] += false_positive_count
            self._false_negative_count_by_class_index[class_index] += false_negative_count
