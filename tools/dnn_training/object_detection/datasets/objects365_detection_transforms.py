import torch
import torchvision.transforms.functional as F

from object_detection.datasets.coco_detection_transforms import _resize_image
from object_detection.datasets.object_detection_objects365 import CLASS_COUNT


def _convert_bbox_to_yolo(target, scale, offset_x, offset_y, one_hot_class):
    if one_hot_class:
        converted_target = {'bbox': torch.zeros(len(target), 4, dtype=torch.float),
                            'class': torch.zeros(len(target), CLASS_COUNT, dtype=torch.float)}
    else:
        converted_target = {'bbox': torch.zeros(len(target), 4, dtype=torch.float),
                            'class': torch.zeros(len(target), dtype=torch.long)}

    for i in range(len(target)):
        converted_target['bbox'][i] = torch.tensor([target[i]['x_center'] * scale + offset_x,
                                                    target[i]['y_center'] * scale + offset_y,
                                                    target[i]['width'] * scale,
                                                    target[i]['height'] * scale], dtype=torch.float)
        if one_hot_class:
            converted_target['class'][i, target[i]['class_index']] = 1.0
        else:
            converted_target['class'][i] = target[i]['class_index']

    return converted_target


class Objects365DetectionValidationTransforms:
    def __init__(self, image_size, one_hot_class):
        self._image_size = image_size
        self._one_hot_class = one_hot_class

    def __call__(self, image, target):
        resized_image, scale, offset_x, offset_y = _resize_image(image, self._image_size)
        resized_image_tensor = F.to_tensor(resized_image)

        if target is not None:
            target = _convert_bbox_to_yolo(target, scale, offset_x, offset_y, self._one_hot_class)

        metadata = {
            'scale': scale,
            'offset_x': offset_x,
            'offset_y': offset_y
        }
        return resized_image_tensor, target, metadata
