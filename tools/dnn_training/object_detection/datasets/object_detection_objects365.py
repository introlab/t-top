import os
from collections import defaultdict

import torch
from PIL import Image

CLASS_COUNT = 365
COCO_OBJECTS365_CLASS_INDEXES = {0, 46, 5, 58, 114, 55, 116, 65, 21, 40, 176, 127, 249, 24, 56, 139, 92, 78, 99, 96,
                                 144, 295, 178, 180, 38, 39, 13, 43, 194, 219, 119, 173, 154, 137, 113, 145, 146, 204,
                                 8, 35, 10, 88, 84, 93, 26, 112, 82, 265, 104, 141, 152, 234, 143, 150, 97, 2, 50, 25,
                                 75, 98, 153, 37, 73, 115, 132, 106, 64, 163, 149, 277, 81, 133, 18, 94, 30, 169, 328,
                                 226, 239, 156, 165, 177, 206}


class ObjectDetectionObjects365(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transforms=None, ignored_classes=None):
        if ignored_classes is None:
            ignored_classes = set()
        else:
            ignored_classes = set(ignored_classes)

        if split == 'training':
            self._image_root = os.path.join(root, 'images', 'train')
            self._label_root = os.path.join(root, 'labels', 'train')
        elif split == 'validation':
            self._image_root = os.path.join(root, 'images', 'val')
            self._label_root = os.path.join(root, 'labels', 'val')
        else:
            raise ValueError('Invalid split')

        self._image_files, self._bboxes = self._list_images(self._image_root, self._label_root, ignored_classes)
        self._transforms = transforms

    def _list_images(self, image_path, label_path, ignored_classes):
        image_files = os.listdir(image_path)
        bboxes = defaultdict(list)

        for image_file in image_files:
            with open(os.path.join(label_path, os.path.splitext(image_file)[0] + '.txt'), 'r') as f:
                for line in f:
                    values = line.split(' ')
                    class_index = int(values[0])
                    if class_index in ignored_classes:
                        continue

                    x_center = float(values[1])
                    y_center = float(values[2])
                    width = float(values[3])
                    height = float(values[4])

                    bboxes[image_file].append({
                        'class_index': class_index,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })

        return image_files, bboxes

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, index):
        image_file = self._image_files[index]
        image = Image.open(os.path.join(self._image_root, image_file)).convert('RGB')

        initial_width, initial_height = image.size

        target = []
        for i in range(len(self._bboxes[image_file])):
            target.append({
                'class_index': self._bboxes[image_file][i]['class_index'],
                'x_center': self._bboxes[image_file][i]['x_center'] * initial_width,
                'y_center': self._bboxes[image_file][i]['y_center'] * initial_height,
                'width': self._bboxes[image_file][i]['width'] * initial_width,
                'height': self._bboxes[image_file][i]['height'] * initial_height
            })

        image, target, transforms_metadata = self._transforms(image, target)
        metadata = {
            'initial_width': initial_width,
            'initial_height': initial_height,
            'scale': transforms_metadata['scale'],
            'offset_x': transforms_metadata['offset_x'],
            'offset_y': transforms_metadata['offset_y']
        }

        return image, target, metadata
