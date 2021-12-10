import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.criterions import SigmoidFocalLossWithLogits
from object_detection.modules.yolo_layer import CONFIDENCE_INDEX, CLASSES_INDEX

CIOU_NORMALIZER = 0.07
OBJECT_NORMALIZER = 1
CLASS_NORMALIZER = 1


# Use Complete IoU Loss for the bounding box (https://arxiv.org/abs/1911.08287)
class YoloV4Loss(nn.Module):
    def __init__(self, image_size, anchors, output_strides, class_count, class_criterion_type='cross_entropy_loss'):
        super(YoloV4Loss, self).__init__()
        self._image_size = image_size
        self._anchors = anchors
        self._output_strides = output_strides
        self._class_count = class_count
        self._class_criterion_type = class_criterion_type
        self._sigmoid_focal_loss_with_logits = SigmoidFocalLossWithLogits()

    def forward(self, prediction, target):
        bbox_count = _count_bboxes(target)
        predicted_bboxes = torch.zeros(bbox_count, target[0]['bbox'].size()[1], dtype=torch.float,
                                       device=prediction[0].device)
        target_bboxes = torch.zeros(bbox_count, target[0]['bbox'].size()[1], dtype=torch.float,
                                    device=prediction[0].device)
        class_scores = torch.zeros(bbox_count, self._class_count, dtype=torch.float, device=prediction[0].device)
        if self._class_criterion_type == 'cross_entropy_loss':
            class_target = torch.zeros(bbox_count, dtype=torch.long, device=prediction[0].device)
        else:
            class_target = torch.zeros(bbox_count, self._class_count, dtype=torch.float, device=prediction[0].device)

        bbox_index = 0
        predictors = self._find_best_predictors(target)

        object_loss = 0
        no_object_loss = 0
        for i in range(len(prediction)):
            confidence_target, bbox_index = _build_target(prediction[i], predictors[i], predicted_bboxes,
                                                          target_bboxes, class_scores, class_target, bbox_index,
                                                          self._class_count)

            obj_confidence_index = (confidence_target[:, :, :, :, 0] != 0)
            noobj_confidence_index = (confidence_target[:, :, :, :, 0] == 0)

            object_loss += F.binary_cross_entropy(
                prediction[i][:, :, :, :, CONFIDENCE_INDEX][obj_confidence_index],
                confidence_target[:, :, :, :, 0][obj_confidence_index])
            no_object_loss += F.binary_cross_entropy(
                prediction[i][:, :, :, :, CONFIDENCE_INDEX][noobj_confidence_index],
                confidence_target[:, :, :, :, 0][noobj_confidence_index])

        object_loss = object_loss / len(prediction) * OBJECT_NORMALIZER
        no_object_loss /= len(prediction)
        class_loss = self._calculate_class_loss(class_scores, class_target)
        ciou_loss = _calculate_ciou(predicted_bboxes, target_bboxes) * CIOU_NORMALIZER

        loss = ciou_loss + object_loss + no_object_loss + class_loss

        if torch.isfinite(loss).all():
            return loss
        else:
            print('Warning: The loss is not finite')
            return torch.tensor(0.0, dtype=torch.float, device=prediction[0].device)

    def _find_best_predictors(self, target):
        predictors = [[] for _ in range(len(self._anchors))]
        anchors = torch.from_numpy(np.concatenate(self._anchors, axis=0))
        anchors = torch.cat([torch.zeros(anchors.size()[0], 2), anchors], dim=1)

        for n in range(len(target)):
            for i in range(len(target[n]['bbox'])):
                predictor_index, c = _find_best_anchor(anchors, target[n]['bbox'][i][2], target[n]['bbox'][i][3],
                                                       self._anchors[0].shape[0])

                x = int(target[n]['bbox'][i][0]) // self._output_strides[predictor_index]
                y = int(target[n]['bbox'][i][1]) // self._output_strides[predictor_index]
                predictors[predictor_index].append({
                    'bbox': target[n]['bbox'][i],
                    'n': n,
                    'c': c,
                    'x': np.clip(x, a_min=0, a_max=self._image_size[1] // self._output_strides[predictor_index] - 1),
                    'y': np.clip(y, a_min=0, a_max=self._image_size[0] // self._output_strides[predictor_index] - 1),
                    'class': target[n]['class'][i]
                })

        return predictors

    def _calculate_class_loss(self, class_scores, class_target):
        if self._class_criterion_type == 'cross_entropy_loss':
            class_loss = F.cross_entropy(class_scores, class_target)
        elif self._class_criterion_type == 'bce_loss':
            class_loss = F.binary_cross_entropy_with_logits(class_scores, class_target)
        elif self._class_criterion_type == 'sigmoid_focal_loss':
            class_loss = self._sigmoid_focal_loss_with_logits(class_scores, class_target)
        else:
            raise ValueError('Invalid class criterion type')

        return class_loss * CLASS_NORMALIZER


def _find_best_anchor(anchors, w, h, c):
    iou = calculate_iou(torch.tensor([0, 0, w, h]).repeat(anchors.size()[0], 1), anchors)
    i = torch.argmax(iou).item()
    return i // c, i % c


def _build_target(prediction, predictors, predicted_bboxes, target_bboxes, class_scores, class_target, bbox_index,
                  class_count):
    N = prediction.size()[0]
    H = prediction.size()[1]
    W = prediction.size()[2]
    N_ANCHORS = prediction.size()[3]

    confidence_target = torch.zeros(N, H, W, N_ANCHORS, 1, dtype=torch.float, device=prediction.device)

    for predictor in predictors:
        predicted_bboxes[bbox_index] = \
            prediction[predictor['n'], predictor['y'], predictor['x'], predictor['c'], :predicted_bboxes.size()[1]]
        target_bboxes[bbox_index] = predictor['bbox']

        confidence_target[predictor['n'], predictor['y'], predictor['x'], predictor['c'], 0] = \
            calculate_iou(predicted_bboxes[bbox_index].unsqueeze(0), target_bboxes[bbox_index].unsqueeze(0)).detach()

        class_scores[bbox_index] = prediction[predictor['n'], predictor['y'], predictor['x'], predictor['c'],
                                   CLASSES_INDEX:CLASSES_INDEX + class_count]
        class_target[bbox_index] = predictor['class']

        bbox_index += 1

    return confidence_target, bbox_index


def _count_bboxes(target):
    count = 0
    for i in range(len(target)):
        for _ in range(len(target[i]['bbox'])):
            count += 1
    return count


def _calculate_ciou(bboxes_a, bboxes_b, iou=None):
    """
    :param bboxes_a: (N, 4) tensor center_x, center_y, w, h
    :param bboxes_b: (N, 4) tensor center_x, center_y, w, h
    :return: ciou loss
    """
    if iou is None:
        iou = calculate_iou(bboxes_a, bboxes_b)
    normalized_center_distance_squared = _calculate_normalized_center_distance_squared(bboxes_a, bboxes_b)
    v = _calculate_aspect_ratio_consistency(bboxes_a, bboxes_b)
    alpha = _calculate_ciou_trade_off_parameter(iou, v)

    ciou = 1 - iou + normalized_center_distance_squared + alpha * v
    return ciou.mean(dim=0)


def calculate_iou(bboxes_a, bboxes_b):
    """
    :param bboxes_a: (N, 4) tensor center_x, center_y, w, h
    :param bboxes_b: (N, 4) tensor center_x, center_y, w, h
    :return: iou (N) tensor
    """
    areas_a = bboxes_a[:, 2] * bboxes_a[:, 3]
    areas_b = bboxes_b[:, 2] * bboxes_b[:, 3]

    a_tl_x, a_tl_y, a_br_x, a_br_y = _get_tl_br_points(bboxes_a)
    b_tl_x, b_tl_y, b_br_x, b_br_y = _get_tl_br_points(bboxes_b)

    intersection_w = torch.min(a_br_x, b_br_x) - torch.max(a_tl_x, b_tl_x)
    intersection_h = torch.min(a_br_y, b_br_y) - torch.max(a_tl_y, b_tl_y)
    intersection_w = torch.max(intersection_w, torch.zeros_like(intersection_w))
    intersection_h = torch.max(intersection_h, torch.zeros_like(intersection_h))

    intersection_area = intersection_w * intersection_h

    return intersection_area / (areas_a + areas_b - intersection_area)


def _calculate_normalized_center_distance_squared(bboxes_a, bboxes_b):
    """
    :param bboxes_a: (N, 4) tensor center_x, center_y, w, h
    :param bboxes_b: (N, 4) tensor center_x, center_y, w, h
    :return: normalized center distance squared (N) tensor
    """
    a_tl_x, a_tl_y, a_br_x, a_br_y = _get_tl_br_points(bboxes_a)
    b_tl_x, b_tl_y, b_br_x, b_br_y = _get_tl_br_points(bboxes_b)

    box_tl_x = torch.min(a_tl_x, b_tl_x)
    box_tl_y = torch.min(a_tl_y, b_tl_y)
    box_br_x = torch.max(a_br_x, b_br_x)
    box_br_y = torch.max(a_br_y, b_br_y)

    box_w = box_br_x - box_tl_x
    box_h = box_br_y - box_tl_y

    c_squared = torch.pow(box_w, 2) + torch.pow(box_h, 2)
    center_distance_x = bboxes_a[:, 0] - bboxes_b[:, 0]
    center_distance_y = bboxes_a[:, 1] - bboxes_b[:, 1]
    center_distance_squared = torch.pow(center_distance_x, 2) + torch.pow(center_distance_y, 2)

    return center_distance_squared / c_squared


def _get_tl_br_points(bboxes):
    """
    :param bboxes: (N, 4) tensor center_x, center_y, w, h
    :return: top left x, top left y, bottom right x and bottom right y  (N) tensors
    """
    tl_x = bboxes[:, 0] - bboxes[:, 2] / 2
    tl_y = bboxes[:, 1] - bboxes[:, 3] / 2
    br_x = bboxes[:, 0] + bboxes[:, 2] / 2
    br_y = bboxes[:, 1] + bboxes[:, 3] / 2

    return tl_x, tl_y, br_x, br_y


def _calculate_aspect_ratio_consistency(bboxes_a, bboxes_b):
    """
    :param bboxes_a: (N, 4) tensor center_x, center_y, w, h
    :param bboxes_b: (N, 4) tensor center_x, center_y, w, h
    :return: aspect ratio consistency value (N) tensor
    """
    a = torch.atan(bboxes_a[:, 2] / bboxes_a[:, 3]) - torch.atan(bboxes_b[:, 2] / bboxes_b[:, 3])
    return 4 / math.pi ** 2 * torch.pow(a, 2)


def _calculate_ciou_trade_off_parameter(iou, v):
    return v / (1 - iou + v)
