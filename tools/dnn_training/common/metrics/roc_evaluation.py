import os
import time
import json

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


class RocEvaluation:
    def __init__(self, output_path):
        self._output_path = output_path

    def _calculate_accuracy_true_positive_rate_false_positive_rate_frame_counts(self, true_positive,
                                                                                false_positive,
                                                                                true_negative,
                                                                                false_negative):
        accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)

        true_positive_rate = 0
        if true_positive + false_negative > 0:
            true_positive_rate = true_positive / (true_positive + false_negative)

        false_positive_rate = 0
        if false_positive + true_negative > 0:
            false_positive_rate = false_positive / (false_positive + true_negative)

        return accuracy, true_positive_rate, false_positive_rate

    def _calculate_auc(self, true_positive_rate_curve, false_positive_rate_curve):
        return np.trapz(true_positive_rate_curve, false_positive_rate_curve)

    def _calculate_eer(self, true_positive_rate_curve, false_positive_rate_curve):
        false_negative_rate_curve = 1 - true_positive_rate_curve
        abs_diff = np.abs(false_negative_rate_curve - false_positive_rate_curve)
        index = np.argmin(abs_diff)

        return (false_negative_rate_curve[index] + false_positive_rate_curve[index]) / 2

    def _save_roc_curve(self, true_positive_rate_curve, false_positive_rate_curve, prefix=''):
        fig = plt.figure(figsize=(5, 5), dpi=300)
        ax1 = fig.add_subplot(111)

        ax1.plot([0, 1], [0, 1], '--')
        ax1.plot(false_positive_rate_curve, true_positive_rate_curve)
        ax1.set_title(u'ROC curve')
        ax1.set_xlabel(u'False positive rate')
        ax1.set_ylabel(u'True positive rate')

        fig.savefig(os.path.join(self._output_path, prefix + 'roc_curve.png'))
        plt.close(fig)

    def _save_roc_curve_data(self, true_positive_rate_curve, false_positive_rate_curve, thresholds, prefix=''):
        with open(os.path.join(self._output_path, prefix + 'roc_curve.json'), 'w') as file:
            data = {
                'true_positive_rate_curve': true_positive_rate_curve.tolist(),
                'false_positive_rate_curve': false_positive_rate_curve.tolist(),
                'thresholds': thresholds.tolist()
            }
            json.dump(data, file, indent=4, sort_keys=True)

    def _save_performances(self, values_by_name, prefix=''):
        with open(os.path.join(self._output_path, prefix + 'performance.json'), 'w') as file:
            json.dump(values_by_name, file, indent=4, sort_keys=True)



class RocDistancesThresholdsEvaluation(RocEvaluation):
    def __init__(self, output_path, thresholds):
        super(RocDistancesThresholdsEvaluation, self).__init__(output_path)
        self._thresholds = thresholds

    def evaluate(self):
        print('Calculate distances')
        distances = self._calculate_distances()
        is_same_person_target = self._get_is_same_person_target()

        best_accuracy, best_threshold, true_positive_rate_curve, false_positive_rate_curve, thresholds = \
            self._calculate_accuracy_true_positive_rate_false_positive_rate(distances, is_same_person_target)
        auc = self._calculate_auc(true_positive_rate_curve, false_positive_rate_curve)
        eer = self._calculate_eer(true_positive_rate_curve, false_positive_rate_curve)

        print('Best accuracy: {}, threshold: {}, AUC: {}, EER: {}'.format(best_accuracy, best_threshold, auc, eer))
        self._save_roc_curve(true_positive_rate_curve, false_positive_rate_curve)
        self._save_roc_curve_data(true_positive_rate_curve, false_positive_rate_curve, thresholds)
        self._save_performances({
            'best_accuracy': best_accuracy,
            'best_threshold': best_threshold,
            'auc': auc,
            'eer': eer
        })

    def _calculate_distances(self):
        raise NotImplementedError()

    def _get_is_same_person_target(self):
        raise NotImplementedError()

    def _calculate_accuracy_true_positive_rate_false_positive_rate(self, distances, is_same_person_target):
        print('Calculate accuracy and ROC', flush=True)

        best_threshold = 0
        best_accuracy = 0
        true_positive_rate_curve = np.zeros(len(self._thresholds))
        false_positive_rate_curve = np.zeros(len(self._thresholds))

        for i in tqdm(range(len(self._thresholds))):
            accuracy, true_positive_rate, false_positive_rate = \
                self._calculate_accuracy_true_positive_rate_false_positive_rate_for_threshold(distances,
                                                                                              is_same_person_target,
                                                                                              self._thresholds[i])

            true_positive_rate_curve[i] = true_positive_rate
            false_positive_rate_curve[i] = false_positive_rate

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = self._thresholds[i]

        sorted_indexes = np.argsort(false_positive_rate_curve)
        return best_accuracy, best_threshold, \
               true_positive_rate_curve[sorted_indexes], false_positive_rate_curve[sorted_indexes], \
               self._thresholds[sorted_indexes]

    def _calculate_accuracy_true_positive_rate_false_positive_rate_for_threshold(self, distances, is_same_person_target,
                                                                                 threshold):
        is_same_person_prediction = distances < threshold
        true_positive = (is_same_person_prediction & is_same_person_target).sum().item()
        false_positive = (is_same_person_prediction & ~is_same_person_target).sum().item()
        true_negative = (~is_same_person_prediction & ~is_same_person_target).sum().item()
        false_negative = (~is_same_person_prediction & is_same_person_target).sum().item()

        return self._calculate_accuracy_true_positive_rate_false_positive_rate_frame_counts(true_positive,
                                                                                            false_positive,
                                                                                            true_negative,
                                                                                            false_negative)
