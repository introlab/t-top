import numpy as np
from sklearn.metrics import average_precision_score

import torch


class MulticlassClassificationMeanAveragePrecisionMetric:
    def __init__(self, class_count, apply_sigmoid=True):
        self._class_count = class_count
        self._apply_sigmoid = apply_sigmoid

        self._predictions = []
        self._targets = []

    def clear(self):
        self._predictions = []
        self._targets = []

    def add(self, prediction, target):
        if self._apply_sigmoid:
            prediction = torch.sigmoid(prediction)

        prediction = prediction.cpu().detach().numpy()
        target = (target > 0.0).float().cpu().detach().numpy()

        self._predictions.append(prediction)
        self._targets.append(target)

    def get_value(self):
        prediction = np.concatenate(self._predictions, axis=0)
        target = np.concatenate(self._targets, axis=0)

        mean_average_precision = 0
        for class_index in range(self._class_count):
            mean_average_precision += average_precision_score(target[:, class_index], prediction[:, class_index])

        return mean_average_precision / self._class_count

    def get_values_by_class(self, class_names):
        prediction = np.concatenate(self._predictions, axis=0)
        target = np.concatenate(self._targets, axis=0)

        ap_by_class = {}
        for class_index, class_name in enumerate(class_names):
            ap_by_class[class_name] = average_precision_score(target[:, class_index], prediction[:, class_index])

        ap_by_class = list(ap_by_class.items())
        ap_by_class.sort(key=lambda x: x[1], reverse=True)

        return ap_by_class
