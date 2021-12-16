import torch

from object_detection.criterions.yolo_v4_loss import calculate_iou
from object_detection.modules.yolo_layer import CLASSES_INDEX
from object_detection.filter_yolo_predictions import filter_yolo_predictions, group_predictions


class YoloAccuracyMetric:
    def __init__(self, class_count, iou_threshold=0.5, confidence_threshold=0.5, nms_threshold=0.1):
        self._class_count = class_count
        self._iou_threshold = iou_threshold
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold

        self._good_bbox = 0.0
        self._good_class = 0.0
        self._total = 0.0

    def clear(self):
        self._good_bbox = 0.0
        self._good_class = 0.0
        self._total = 0.0

    def add(self, predictions, target):
        predictions = group_predictions(predictions)
        for n in range(predictions.size()[0]):
            self._add_image(predictions[n], target[n])

    def _add_image(self, predictions, target):
        predictions = filter_yolo_predictions(predictions,
                                              confidence_threshold=self._confidence_threshold,
                                              nms_threshold=self._nms_threshold)
        if len(predictions) == 0:
            self._total += target['class'].size()[0]
        else:
            predictions = torch.stack(predictions)
            for i in range(target['class'].size()[0]):
                classes_scores, iou = _find_best_prediction_class_and_iou(target['bbox'][i],
                                                                           predictions,
                                                                           self._class_count)

                if iou >= self._iou_threshold:
                    self._good_bbox += 1.0

                    predicted_class = torch.argmax(classes_scores).item()
                    if target['class'][i].numel() == 1:
                        target_class = target['class'][i].item()
                    else:
                        target_class = torch.argmax(target['class'][i]).item()

                    if target_class == predicted_class:
                        self._good_class += 1.0

                self._total += 1.0

    def get_bbox_accuracy(self):
        if self._total == 0.0:
            return 1
        return self._good_bbox / self._total

    def get_class_accuracy(self):
        if self._good_bbox == 0.0:
            return 1
        return self._good_class / self._good_bbox


def _find_best_prediction_class_and_iou(target, prediction, class_count):
    iou = calculate_iou(target.repeat(prediction.size()[0], 1), prediction[:, 0:4])
    best_iou_index = torch.argmax(iou)

    classes_scores = prediction[best_iou_index][CLASSES_INDEX:CLASSES_INDEX + class_count]

    return classes_scores, iou[best_iou_index]
