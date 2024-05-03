import copy
import random

from PIL import Image

import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

CATEGORY_ID_TO_CLASS_INDEX_MAPPING = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
    22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
    35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
    46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59,
    67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69,
    80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
}

CLASS_INDEX_TO_CATEGORY_ID_MAPPING = {value: key for key, value in CATEGORY_ID_TO_CLASS_INDEX_MAPPING.items()}

CROP_FACTOR = 0.1


def _random_crop(image, target):
    w, h = image.size
    x0 = int(random.random() * CROP_FACTOR * w)
    y0 = int(random.random() * CROP_FACTOR * h)
    x1 = int(w - random.random() * CROP_FACTOR * w)
    y1 = int(h - random.random() * CROP_FACTOR * h)

    image = image.crop((x0, y0, x1, y1))
    cropped_target = []
    for annotation in target:
        x = annotation['bbox'][0]
        y = annotation['bbox'][1]
        w = annotation['bbox'][2]
        h = annotation['bbox'][3]

        if (x <= x0 and x + w <= x0) or (y <= y0 and y + h <= y0) or \
                (x >= x1 and x + w >= x1) or (y >= y1 and y + h >= y1):
            continue

        w = w - max(0, x + w - x1)
        h = h - max(0, y + h - y1)
        x = max(0, x - x0)
        y = max(0, y - y0)

        if w > 0 and h > 0:
            annotation = copy.deepcopy(annotation)
            annotation['bbox'] = [x, y, w, h]
            cropped_target.append(annotation)

    return image, cropped_target


def _resize_image(image, size):
    w, h = image.size
    scale = min(size[0] / h, size[1] / w)

    image = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    offset_x = int((size[0] - image.width) / 2)
    offset_y = int((size[1] - image.height) / 2)
    padded_image = Image.new('RGB', (size[0], size[1]), (114, 114, 114))
    padded_image.paste(image, (offset_x, offset_y))

    return padded_image, scale, offset_x, offset_y


def _hflip_bbox(target, image_size):
    for annotation in target['bbox']:
        center_x = annotation[0]
        annotation[0] = image_size[1] - center_x


def _convert_bbox_to_yolo(target, scale, image_size, offset_x, offset_y, one_hot_class):
    if one_hot_class:
        class_count = len(CATEGORY_ID_TO_CLASS_INDEX_MAPPING)
        converted_target = {'bbox': torch.zeros(len(target), 4, dtype=torch.float),
                            'class': torch.zeros(len(target), class_count, dtype=torch.float)}
    else:
        converted_target = {'bbox': torch.zeros(len(target), 4, dtype=torch.float),
                            'class': torch.zeros(len(target), dtype=torch.long)}

    for i in range(len(target)):
        x = target[i]['bbox'][0] * scale
        y = target[i]['bbox'][1] * scale
        w = target[i]['bbox'][2] * scale
        h = target[i]['bbox'][3] * scale

        x = np.clip(x, 0, image_size[1] - 1)
        y = np.clip(y, 0, image_size[0] - 1)
        w = min([w, image_size[1] - x])
        h = min([h, image_size[0] - y])

        center_x = x + w / 2 + offset_x
        center_y = y + h / 2 + offset_y

        converted_target['bbox'][i] = torch.tensor([center_x, center_y, w, h], dtype=torch.float)
        if one_hot_class:
            converted_target['class'][i, CATEGORY_ID_TO_CLASS_INDEX_MAPPING[target[i]['category_id']]] = 1.0
        else:
            converted_target['class'][i] = CATEGORY_ID_TO_CLASS_INDEX_MAPPING[target[i]['category_id']]

    return converted_target


class CocoDetectionTrainingTransforms:
    def __init__(self, image_size, one_hot_class):
        self._image_size = image_size
        self._one_hot_class = one_hot_class
        self._image_only_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, saturation=0.3, contrast=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1)])

        self._horizontal_flip_p = 0.5

    def __call__(self, image, target):
        image = self._image_only_transform(image)

        image, target = _random_crop(image, target)

        resized_image, scale, offset_x, offset_y = _resize_image(image, self._image_size)
        target = _convert_bbox_to_yolo(target, scale, self._image_size, offset_x, offset_y, self._one_hot_class)

        if random.random() < self._horizontal_flip_p:
            resized_image = F.hflip(resized_image)
            _hflip_bbox(target, self._image_size)

        resized_image_tensor = F.to_tensor(resized_image)

        metadata = {
            'scale': scale,
            'offset_x': offset_x,
            'offset_y': offset_y
        }
        return resized_image_tensor, target, metadata


class CocoDetectionValidationTransforms:
    def __init__(self, image_size, one_hot_class):
        self._image_size = image_size
        self._one_hot_class = one_hot_class

    def __call__(self, image, target):
        resized_image, scale, offset_x, offset_y = _resize_image(image, self._image_size)
        resized_image_tensor = F.to_tensor(resized_image)

        if target is not None:
            target = _convert_bbox_to_yolo(target, scale, self._image_size, offset_x, offset_y, self._one_hot_class)

        metadata = {
            'scale': scale,
            'offset_x': offset_x,
            'offset_y': offset_y
        }
        return resized_image_tensor, target, metadata
