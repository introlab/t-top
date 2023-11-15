import os
import random

import numpy as np

from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

COCO_PERSON_CATEGORY_ID = 1
NONE_VISIBILITY = 0

RANDOM_CROP_RATIO = 0.2
RANDOM_KEYPOINT_MASK_P = 1.0
RANDOM_KEYPOINT_MASK_RATIO = 0.2


class PoseEstimationCoco(Dataset):
    def __init__(self, root, train=True, data_augmentation=False, image_transforms=None, heatmap_sigma=10):
        self._data_augmentation = data_augmentation
        self._image_transforms = image_transforms
        self._heatmap_sigma = heatmap_sigma

        if train:
            self._image_root = os.path.join(root, 'train2017')
            self._annotation_file_path = os.path.join(root, 'person_keypoints_train2017.json')
        else:
            self._image_root = os.path.join(root, 'val2017')
            self._annotation_file_path = os.path.join(root, 'person_keypoints_val2017.json')

        self._coco = COCO(self._annotation_file_path)

        self._pose_indexes = []
        self._index_poses()

    def _index_poses(self):
        for image_id in self._coco.getImgIds():
            try:
                annotation_ids = self._coco.getAnnIds(imgIds=image_id, iscrowd=False)

                for annotation_id in annotation_ids:
                    object = self._coco.loadAnns(annotation_id)[0]
                    if self._is_person_valid(object):
                        self._pose_indexes.append({
                            'image_id': image_id,
                            'path': os.path.join(self._image_root, '%012d.jpg' % image_id),
                            'annotation_id': annotation_id
                        })
            except IndexError:
                pass

    def _is_person_valid(self, object):
        return 'keypoints' in object and \
               object['category_id'] == COCO_PERSON_CATEGORY_ID and \
               object['bbox'][2] > 0 and \
               object['bbox'][3] > 0

    def __len__(self):
        return len(self._pose_indexes)

    def __getitem__(self, index):
        pose = self._pose_indexes[index]
        image = Image.open(pose['path']).convert('RGB')
        object = self._coco.loadAnns(pose['annotation_id'])[0]
        cropped_image, crop_offset_x, crop_offset_y = self._crop_image(image, object)

        if self._data_augmentation:
            self._mask_random_keypoint(cropped_image, crop_offset_x, crop_offset_y, object)

        transformed_image = self._image_transforms(cropped_image)
        _, transformed_image_height, transformed_image_width = transformed_image.size()

        def transform_keypoint(keypoint_x, keypoint_y):
            cropped_x = keypoint_x - crop_offset_x
            cropped_y = keypoint_y - crop_offset_y

            cropped_image_width, cropped_image_height = cropped_image.size
            transformed_x = int(cropped_x / cropped_image_width * transformed_image_width)
            transformed_y = int(cropped_y / cropped_image_height * transformed_image_height)
            return transformed_x, transformed_y

        heatmaps, presence = self._generate_heatmaps(transformed_image_width, transformed_image_height,
                                                     transform_keypoint, object)

        oks_scale = object['area']
        metadata = {'image_id': self._pose_indexes[index]['image_id'],
                    'annotation_id': self._pose_indexes[index]['annotation_id']}
        return transformed_image, (heatmaps, presence, oks_scale), metadata

    def _crop_image(self, image, object):
        image_width, image_height = image.size

        x0 = object['bbox'][0]
        y0 = object['bbox'][1]
        w = object['bbox'][2]
        h = object['bbox'][3]
        x1 = x0 + w
        y1 = y0 + h

        if self._data_augmentation:
            x0 -= int(random.random() * RANDOM_CROP_RATIO * w)
            y0 -= int(random.random() * RANDOM_CROP_RATIO * h)
            x1 += int(random.random() * RANDOM_CROP_RATIO * w)
            y1 += int(random.random() * RANDOM_CROP_RATIO * h)

        x0 = np.clip(x0, a_min=0, a_max=image_width)
        y0 = np.clip(y0, a_min=0, a_max=image_height)
        x1 = np.clip(x1, a_min=0, a_max=image_width)
        y1 = np.clip(y1, a_min=0, a_max=image_height)

        return image.crop((x0, y0, x1, y1)), x0, y0

    def _mask_random_keypoint(self, image, crop_offset_x, crop_offset_y, object):
        if random.random() < RANDOM_KEYPOINT_MASK_P:
            image_width, image_height = image.size

            keypoint_count = len(object['keypoints']) // 3
            keypoint_index = random.randrange(keypoint_count)

            x = object['keypoints'][3 * keypoint_index] - crop_offset_x
            y = object['keypoints'][3 * keypoint_index + 1] - crop_offset_y
            w = image_width * RANDOM_KEYPOINT_MASK_RATIO
            h = image_height * RANDOM_KEYPOINT_MASK_RATIO

            draw = ImageDraw.Draw(image)
            draw.rectangle([x - w // 2, y - h // 2, x + w // 2, y + h // 2], fill='black')

    def _generate_heatmaps(self, image_width, image_height, transform_keypoint, object):
        keypoint_count = len(object['keypoints']) // 3

        heatmaps = []
        presence = torch.zeros((keypoint_count,))
        for i in range(keypoint_count):
            x = object['keypoints'][3 * i]
            y = object['keypoints'][3 * i + 1]
            is_none = object['keypoints'][3 * i + 1] == NONE_VISIBILITY
            presence[i] = 0 if is_none else 1

            x, y = transform_keypoint(x, y)
            heatmap = self._generate_heatmap(x, y, image_width, image_height, is_none)
            heatmaps.append(heatmap.unsqueeze(0))

        return torch.cat(heatmaps, dim=0), presence

    def _generate_heatmap(self, keypoint_x, keypoint_y, image_width, image_height, is_none):
        if is_none:
            return torch.zeros((image_height, image_width))

        heatmap_x = torch.arange(image_width, dtype=torch.float)
        heatmap_y = torch.arange(image_height, dtype=torch.float)
        heatmap_grid_y, heatmap_grid_x = torch.meshgrid(heatmap_y, heatmap_x, indexing='ij')

        return torch.exp(-(torch.pow(heatmap_grid_x - keypoint_x, 2) + torch.pow(heatmap_grid_y - keypoint_y, 2)) /
                         (2 * self._heatmap_sigma ** 2))

    def evaluate(self, result_file):
        coco_gt = COCO(self._annotation_file_path)
        coco_dt = coco_gt.loadRes(result_file)
        coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        return coco_eval.stats, names

    def transform_keypoints(self, annotation_id, keypoints, heatmap_width, heatmap_height):
        if self._data_augmentation:
            raise NotImplementedError()

        object = self._coco.loadAnns(annotation_id)[0]
        x0 = object['bbox'][0]
        y0 = object['bbox'][1]
        original_width = object['bbox'][2]
        original_height = object['bbox'][3]

        transformed_keypoints = []
        for i in range(len(keypoints) // 3):
            transformed_keypoints.append(keypoints[3 * i + 0] / heatmap_width * original_width + x0)
            transformed_keypoints.append(keypoints[3 * i + 1] / heatmap_height * original_height + y0)
            transformed_keypoints.append(keypoints[3 * i + 2])

        return transformed_keypoints
