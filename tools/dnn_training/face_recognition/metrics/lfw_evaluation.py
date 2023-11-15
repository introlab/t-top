import os
import time

import numpy as np

from PIL import Image

import torch

from tqdm import tqdm

from common.metrics import RocDistancesThresholdsEvaluation


class LfwEvaluation(RocDistancesThresholdsEvaluation):
    def __init__(self, model, device, transforms, lwf_dataset_root, output_path):
        super(LfwEvaluation, self).__init__(output_path, thresholds=np.arange(0, 2, 0.001))

        self._model = model
        self._device = device
        self._transforms = transforms
        self._lfw_dataset_root = lwf_dataset_root

        self._fold_count, self._fold_size, self._image_pairs = self._read_image_pairs()

    def evaluate(self):
        print('Calculate distances')
        distances = self._calculate_distances()
        is_same_person_target = self._get_is_same_person_target()

        _, best_threshold, true_positive_rate_curve, false_positive_rate_curve, thresholds = \
            self._calculate_accuracy_true_positive_rate_false_positive_rate(distances, is_same_person_target)
        auc = self._calculate_auc(true_positive_rate_curve, false_positive_rate_curve)
        eer = self._calculate_eer(true_positive_rate_curve, false_positive_rate_curve)

        accuracy, accuracy_std = self._calculate_fold_accuracy(distances, is_same_person_target, best_threshold)

        print('Accuracy: {}±{}, threshold: {}, AUC: {}, EER: {}'.format(accuracy, accuracy_std, best_threshold, auc,
                                                                        eer))
        self._save_roc_curve(true_positive_rate_curve, false_positive_rate_curve)
        self._save_roc_curve_data(true_positive_rate_curve, false_positive_rate_curve, thresholds)
        self._save_performances({
            'accuracy': accuracy,
            'accuracy_std': accuracy_std,
            'best_threshold': best_threshold,
            'auc': auc,
            'eer': eer
        })

    def _read_image_pairs(self):
        image_pairs = []
        with open(os.path.join(self._lfw_dataset_root, 'pairs.txt'), 'r') as f:
            lines = f.readlines()

        p = lines[0].strip().split()
        fold_count = int(p[0])
        fold_size = int(p[1])

        lines = lines[1:]
        not_available_pairs = 0
        for line in lines:
            p = line.strip().split()
            if len(p) == 3:
                image_path1 = os.path.join(self._lfw_dataset_root, p[0], p[0] + '_' + ('%04d' % int(p[1])) + '.jpg')
                image_path2 = os.path.join(self._lfw_dataset_root, p[0], p[0] + '_' + ('%04d' % int(p[2])) + '.jpg')
                is_same_person = True
            elif len(p) == 4:
                image_path1 = os.path.join(self._lfw_dataset_root, p[0], p[0] + '_' + ('%04d' % int(p[1])) + '.jpg')
                image_path2 = os.path.join(self._lfw_dataset_root, p[2], p[2] + '_' + ('%04d' % int(p[3])) + '.jpg')
                is_same_person = False
            else:
                raise ValueError('Invalid pair values ({})'.format(p))

            if os.path.exists(image_path1) and os.path.exists(image_path2):
                image_pairs.append((image_path1, image_path2, is_same_person))
            else:
                not_available_pairs += 1

        print('Not available pairs:', not_available_pairs)
        return fold_count, fold_size, image_pairs

    def _calculate_distances(self):
        distances = []

        for image_path1, image_path2, _ in tqdm(self._image_pairs):
            image1, image2 = self._load_images(image_path1, image_path2)

            embedding1 = self._model(image1.to(self._device))[0]
            embedding2 = self._model(image2.to(self._device))[0]
            distance = torch.dist(embedding1, embedding2, p=2).item()
            distances.append(distance)

        return torch.tensor(distances)

    def _load_images(self, image_path1, image_path2):
        image1 = self._load_image(image_path1).unsqueeze(0)
        image2 = self._load_image(image_path2).unsqueeze(0)

        return image1, image2

    def _load_image(self, path):
        image = Image.open(path).convert('RGB')
        if self._transforms is not None:
            image = self._transforms(image)

        return image

    def _get_is_same_person_target(self):
        return torch.tensor([image_pair[2] for image_pair in self._image_pairs])

    def _calculate_fold_accuracy(self, distances, is_same_person_target, threshold):
        accuracies = []

        for i in range(self._fold_count):
            start = i * self._fold_size
            end = (i + 1) * self._fold_size

            accuracy, _, _ = \
                self._calculate_accuracy_true_positive_rate_false_positive_rate_for_threshold(
                    distances[start:end],
                    is_same_person_target[start:end],
                    threshold)

            accuracies.append(accuracy)

        return np.mean(accuracies), np.std(accuracies)
