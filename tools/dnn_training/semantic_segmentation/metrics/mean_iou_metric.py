import torch


class MeanIoUMetric:
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

    def _add_image(self, prediction, target):
        prediction = torch.argmax(prediction, dim=0)

        for class_index in range(self._class_count):
            true_positive_count = torch.logical_and(target == class_index, prediction == class_index).sum().item()
            false_positive_count = torch.logical_and(target != class_index, prediction == class_index).sum().item()
            false_negative_count = torch.logical_and(target == class_index, prediction != class_index).sum().item()

            self._true_positive_count_by_class_index[class_index] += true_positive_count
            self._false_positive_count_by_class_index[class_index] += false_positive_count
            self._false_negative_count_by_class_index[class_index] += false_negative_count

    def get_mean_iou(self):
        iou_sum = 0.0
        non_empty_class_count = 0
        for class_index in range(self._class_count):
            true_positive_count = self._true_positive_count_by_class_index[class_index]
            false_positive_count = self._false_positive_count_by_class_index[class_index]
            false_negative_count = self._false_negative_count_by_class_index[class_index]
            denominator = (true_positive_count + false_positive_count + false_negative_count)
            if denominator != 0:
                iou_sum += true_positive_count / denominator
                non_empty_class_count += 1

        return iou_sum / non_empty_class_count
