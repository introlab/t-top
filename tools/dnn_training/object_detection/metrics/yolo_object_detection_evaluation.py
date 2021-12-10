import numpy as np

import torch

from tqdm import tqdm

from object_detection.criterions.yolo_v4_loss import calculate_iou
from object_detection.modules.yolo_layer import CONFIDENCE_INDEX, CLASSES_INDEX
from object_detection.filter_yolo_predictions import group_predictions, filter_yolo_predictions


class YoloObjectDetectionEvaluation:
    def __init__(self, model, device, dataset_loader, class_count,
                 confidence_threshold=0.005, nms_threshold=0.45, iou_threshold=0.5):
        self._model = model
        self._device = device
        self._dataset_loader = dataset_loader
        self._dataset = dataset_loader.dataset

        self._class_count = class_count
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self._iou_threshold = iou_threshold

        self._target_count_by_class = [0 for _ in range(self._class_count)]
        self._bbox_results_by_class = [[] for _ in range(self._class_count)]
        self._all_results_by_class = [[] for _ in range(self._class_count)]

    def evaluate(self):
        for image, target, metadata in tqdm(self._dataset_loader):
            predictions = self._model(image.to(self._device))
            predictions = group_predictions(predictions).cpu()

            for n in range(image.size()[0]):
                self._calculate_result_for_one_image(predictions[n], target[n])

        print('BBox mAP={}'.format(self._calculate_map(self._bbox_results_by_class)))
        print('All mAP={}'.format(self._calculate_map(self._all_results_by_class)))

    def _calculate_result_for_one_image(self, predictions, target):
        predictions = filter_yolo_predictions(predictions, self._confidence_threshold, self._nms_threshold)

        self._count_target_for_one_image(target)

        found_target = set()
        for prediction in predictions:
            confidence = prediction[CONFIDENCE_INDEX].item()
            predicted_class = torch.argmax(prediction[CLASSES_INDEX:CLASSES_INDEX + self._class_count]).item()

            true_positive_bbox = 0
            true_positive_all = 0
            false_positive = 0
            if target['bbox'].size()[0] > 0:
                iou, target_index = self._find_best_target(prediction, target)
                target_class = self._get_target_class_index(target['class'][target_index])


                if target_index in found_target or iou < self._iou_threshold:
                    false_positive = 1
                elif iou > self._iou_threshold and predicted_class == target_class:
                    true_positive_bbox = 1
                    true_positive_all = 1
                elif iou > self._iou_threshold:
                    true_positive_bbox = 1
                found_target.add(target_index)
            else:
                false_positive = 1

            self._bbox_results_by_class[predicted_class].append({
                'confidence': confidence,
                'true_positive': true_positive_bbox,
                'false_positive': false_positive,
            })
            self._all_results_by_class[predicted_class].append({
                'confidence': confidence,
                'true_positive': true_positive_all,
                'false_positive': false_positive,
            })

    def _count_target_for_one_image(self, target):
        for bbox_index in range(target['class'].size()[0]):
            target_class_index = self._get_target_class_index(target['class'][bbox_index])
            self._target_count_by_class[target_class_index] += 1

    def _find_best_target(self, prediction, target):
        iou = calculate_iou(prediction[:CONFIDENCE_INDEX].repeat(target['bbox'].size()[0], 1), target['bbox'])
        target_index = torch.argmax(iou)

        return iou[target_index], target_index

    def _get_target_class_index(self, classes):
        if classes.numel() == 1:
            return classes.item()
        else:
            return torch.argmax(classes).item()

    def _calculate_map(self, results_by_class):
        mean_average_precision = 0
        for class_index in range(self._class_count):
            mean_average_precision += self._calculate_average_precision(results_by_class[class_index],
                                                                        self._target_count_by_class[class_index])

        return mean_average_precision / self._class_count

    def _calculate_average_precision(self, results, target_count):
        sorted_results = sorted(results, key=lambda result: result['confidence'], reverse=True)

        recalls = [0]
        precisions = [1]

        true_positive = 0
        false_positive = 0
        for result in sorted_results:
            true_positive += result['true_positive']
            false_positive += result['false_positive']

            recalls.append(true_positive / target_count if target_count > 0 else 0)

            precision_denominator = true_positive + false_positive
            precisions.append(true_positive / precision_denominator if precision_denominator > 0 else 1)

        recalls = np.array(recalls)

        sorted_index = np.argsort(recalls)
        recalls = recalls[sorted_index]
        precisions = np.array(precisions)[sorted_index]

        return np.trapz(y=precisions, x=recalls)
