import numpy as np


class ClassificationMeanAveragePrecisionMetric:
    def __init__(self, class_count):
        self._class_count = class_count

        self._target_count_by_class = [0 for _ in range(self._class_count)]
        self._results_by_class = [[] for _ in range(self._class_count)]

    def clear(self):
        self._target_count_by_class = [0 for _ in range(self._class_count)]
        self._results_by_class = [[] for _ in range(self._class_count)]

    def add(self, prediction, target):
        prediction = prediction.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        N = prediction.shape[0]
        C = prediction.shape[1]

        for n in range(N):
            self._target_count_by_class[target[n]] += 1
            for c in range(C):
                self._results_by_class[c].append({
                    'confidence': prediction[n, c],
                    'is_target_class': target[n] == c
                })

    def get_value(self):
        mean_average_precision = 0
        for class_index in range(self._class_count):
            mean_average_precision += self._calculate_average_precision(self._results_by_class[class_index],
                                                                        self._target_count_by_class[class_index])

        return mean_average_precision / self._class_count

    def _calculate_average_precision(self, results, target_count):
        sorted_results = sorted(results, key=lambda result: result['confidence'], reverse=True)

        recalls = [0]
        precisions = [1]

        true_positive = 0
        false_positive = 0
        for result in sorted_results:
            true_positive += result['is_target_class'] == 1
            false_positive += result['is_target_class'] == 0

            recalls.append(true_positive / target_count if target_count > 0 else 0)

            precision_denominator = true_positive + false_positive
            precisions.append(true_positive / precision_denominator if precision_denominator > 0 else 1)

        recalls = np.array(recalls)
        precisions = np.array(precisions)

        sorted_index = np.argsort(recalls)
        recalls = recalls[sorted_index]
        precisions = precisions[sorted_index]

        return np.trapz(y=precisions, x=recalls)
