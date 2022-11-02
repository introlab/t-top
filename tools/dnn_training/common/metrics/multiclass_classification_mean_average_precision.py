import numpy as np
from sklearn.metrics import average_precision_score

import torch


class MulticlassClassificationMeanAveragePrecisionMetric:
    def __init__(self, class_count):
        self._class_count = class_count
        self._predictions = []
        self._targets = []

    def clear(self):
        self._predictions = []
        self._targets = []

    def add(self, prediction, target):
        prediction = torch.sigmoid(prediction).cpu().detach().numpy()
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
