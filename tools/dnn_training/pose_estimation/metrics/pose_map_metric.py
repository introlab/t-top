import numpy as np

from pose_estimation.metrics.pose_accuracy_metric import PoseAccuracyMetric

THRESHOLDS = np.linspace(0.0, 1.0, num=10).tolist()


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
    return np.trapz(precision, recall)
