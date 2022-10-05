import copy
import os

import numpy as np
from PIL import Image, ImageDraw

import torchvision.datasets as datasets

from object_detection.datasets.coco_detection_transforms import CATEGORY_ID_TO_CLASS_INDEX_MAPPING


CLASS_COUNT = 81


class SemanticSegmentationCoco(datasets.CocoDetection):
    def __init__(self, root, train=True, transforms=None):
        if train:
            ann_file = os.path.join(root, 'instances_train2017.json')
            root = os.path.join(root, 'train2017')
        else:
            ann_file = os.path.join(root, 'instances_val2017.json')
            root = os.path.join(root, 'val2017')

        super(SemanticSegmentationCoco, self).__init__(root, ann_file)
        self._ann_file = ann_file

        if transforms is None:
            raise ValueError('Invalid transforms')
        self._transforms = transforms

    def __getitem__(self, index):
        image, objects = super(SemanticSegmentationCoco, self).__getitem__(index)

        initial_width, initial_height = image.size
        objects = copy.deepcopy(objects)
        target = [self._object_to_target(obj, initial_width, initial_height) for obj in objects]

        image, target, transforms_metadata = self._transforms(image, target)
        metadata = {
            'initial_width': initial_width,
            'initial_height': initial_height
        }
        return image, target, metadata

    def _object_to_target(self, obj, width, height):
        class_index = CATEGORY_ID_TO_CLASS_INDEX_MAPPING[obj['category_id']]
        class_index += 1  # Because the background is 0.
        if obj['iscrowd'] == 0:
            return self._polygon_to_mask(obj['segmentation'], width, height), class_index
        elif obj['iscrowd'] == 1:
            return self._rle_to_mask(obj['segmentation'], width, height), class_index
        else:
            raise ValueError('Invalid iscrowd value')

    def _polygon_to_mask(self, polygons, width, height):
        mask = Image.new('1', (width, height))
        mask_draw = ImageDraw.Draw(mask)

        for polygon in polygons:
            mask_draw.polygon(polygon, fill=255)

        return mask

    def _rle_to_mask(self, rle, width, height):
        mask_np = np.zeros(width * height, dtype=np.uint8)

        ranges = []
        last_end = 0
        for size in rle['counts']:
            start = last_end
            end = start + size
            ranges.append((start, end))
            last_end = end

        for i in range(1, len(ranges), 2):
            start, end = ranges[i]
            mask_np[start:end] = 255

        mask_np = mask_np.reshape((width, height)).T
        mask = Image.fromarray(mask_np)

        return mask
