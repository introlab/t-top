import numpy as np

from pose_estimation.metrics.pose_accuracy_metric import PoseAccuracyMetric

THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.8, 0.85, 0.9]


class PoseMapMetric:
    def __init__(self):
        self._pose_accuracy_metrics = [PoseAccuracyMetric(threshold=threshold) for threshold in THRESHOLDS]

    def clear(self):
        for pose_accuracy_metric in self._pose_accuracy_metrics:
            pose_accuracy_metric.clear()

    def add(self, heatmap_prediction, heatmap_target, presence_target, oks_scale):
        for pose_accuracy_metric in self._pose_accuracy_metrics:
            pose_accuracy_metric.add(heatmap_prediction, heatmap_target, presence_target, oks_scale)

    def get_map(self):
        recall = np.zeros(len(self._pose_accuracy_metrics))
        precision = np.zeros(len(self._pose_accuracy_metrics))

        for i in range(len(self._pose_accuracy_metrics)):
            recall[i] = self._pose_accuracy_metrics[i].get_recall()
            precision[i] = self._pose_accuracy_metrics[i].get_precision()

        return _calculate_ap(recall, precision)


def _calculate_ap(recall, precision):
    sorted_indexes = np.argsort(recall)
    recall = recall[sorted_indexes]
    precision = precision[sorted_indexes]

    recall_points = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ap = 0
    for point in recall_points:
        ap += _find_interpolated_precision(recall, precision, point)
    ap /= len(recall_points)

    return ap


def _find_interpolated_precision(recall, precision, point):
    s = 0
    while s < recall.shape[0] and recall[s] < point:
        s += 1

    if s == recall.shape[0]:
        return 0
    else:
        return np.amax(precision[s:])
