import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from common.datasets import RandomSharpnessChange, RandomAutocontrast, RandomEqualize, RandomPosterize
from object_detection.datasets.coco_detection_transforms import _resize_image


def _convert_bbox_to_yolo(target, scale, image_size, one_hot_class, class_count):
    if one_hot_class:
        converted_target = {'bbox': torch.zeros(len(target), 4, dtype=torch.float),
                            'class': torch.zeros(len(target), class_count, dtype=torch.float)}
    else:
        converted_target = {'bbox': torch.zeros(len(target), 4, dtype=torch.float),
                            'class': torch.zeros(len(target), dtype=torch.long)}

    for i in range(len(target)):
        x_min = target[i]['x_min'] * scale
        x_max = target[i]['x_max'] * scale
        y_min = target[i]['y_min'] * scale
        y_max = target[i]['y_max'] * scale

        x_min = np.clip(x_min, 0, image_size[1] - 1)
        x_max = np.clip(x_max, 0, image_size[1] - 1)
        y_min = np.clip(y_min, 0, image_size[0] - 1)
        y_max = np.clip(y_max, 0, image_size[0] - 1)

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min

        converted_target['bbox'][i] = torch.tensor([center_x, center_y, w, h], dtype=torch.float)
        if one_hot_class:
            converted_target['class'][i, target[i]['class_index']] = 1.0
        else:
            converted_target['class'][i] = target[i]['class_index']

    return converted_target


class OpenImagesDetectionTrainingTransforms:
    def __init__(self, image_size, one_hot_class, class_count):
        self._image_size = image_size
        self._one_hot_class = one_hot_class
        self._class_count = class_count

        self._image_only_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, hue=0.2),
            transforms.RandomGrayscale(p=0.1),
            RandomSharpnessChange(),
            RandomAutocontrast(),
            RandomEqualize(),
            RandomPosterize(),
        ])

    def __call__(self, image, target):
        image = self._image_only_transform(image)

        resized_image, scale, offset_x, offset_y = _resize_image(image, self._image_size)
        target = _convert_bbox_to_yolo(target, scale, self._image_size, self._one_hot_class, self._class_count)

        resized_image_tensor = F.to_tensor(resized_image)

        metadata = {
            'scale': scale,
            'offset_x': offset_x,
            'offset_y': offset_y
        }
        return resized_image_tensor, target, metadata


class OpenImagesDetectionValidationTransforms:
    def __init__(self, image_size, one_hot_class, class_count):
        self._image_size = image_size
        self._one_hot_class = one_hot_class
        self._class_count = class_count

    def __call__(self, image, target):
        resized_image, scale, offset_x, offset_y = _resize_image(image, self._image_size)
        resized_image_tensor = F.to_tensor(resized_image)

        if target is not None:
            target = _convert_bbox_to_yolo(target, scale, self._image_size, self._one_hot_class, self._class_count)

        metadata = {
            'scale': scale,
            'offset_x': offset_x,
            'offset_y': offset_y
        }
        return resized_image_tensor, target, metadata
