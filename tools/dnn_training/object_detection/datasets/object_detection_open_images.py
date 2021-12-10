import os
import csv
from collections import defaultdict
import copy

from PIL import Image

from torch.utils.data import Dataset


HUMAN_BODY_PART_CLASS_NAMES = ['Human eye', 'Human beard', 'Human mouth', 'Human foot', 'Human leg', 'Human ear', 'Human hair', 'Human head', 'Human face', 'Human arm', 'Human nose', 'Human hand']
CLASS_COUNT_WITHOUT_HUMAN_BODY_PART = 589

class ObjectDetectionOpenImages(Dataset):
    def __init__(self, root, split=None, transforms=None, ignored_class_names=None):
        if ignored_class_names is None:
            ignored_class_names = []

        if split == 'training':
            self._root = os.path.join(root, 'train')
        elif split == 'validation':
            self._root = os.path.join(root, 'validation')
        elif split == 'testing':
            self._root = os.path.join(root, 'test')
        else:
            raise ValueError('Invalid split')

        if transforms is None:
            raise ValueError('Invalid transforms')
        self._transforms = transforms

        self._indexes_by_class_id, self._class_names = self._list_classes(ignored_class_names)
        self._rotation_by_image_id = self._list_rotations()
        self._images, self._bboxes = self._list_images()

    def _list_classes(self, ignored_class_names):
        classes = []
        ignored_class_names = set(ignored_class_names)

        with open(os.path.join(self._root, 'metadata', 'classes.csv'), newline='') as class_file:
            class_reader = csv.reader(class_file, delimiter=',', quotechar='"')
            for id, class_name in class_reader:
                if class_name in ignored_class_names:
                    continue

                classes.append({
                    'id': id,
                    'class_name': class_name
                })

        classes.sort(key=lambda x: x['class_name'])
        indexes_by_class_id = {}
        class_names = []

        for i, class_data in enumerate(classes):
            indexes_by_class_id[class_data['id']] = i
            class_names.append(class_data['class_name'])

        return indexes_by_class_id, class_names

    def _list_rotations(self):
        rotation_by_image_id = {}

        with open(os.path.join(self._root, 'metadata', 'image_ids.csv'), newline='') as image_id_file:
            image_id_reader = csv.reader(image_id_file, delimiter=',', quotechar='"')
            next(image_id_reader)
            for row in image_id_reader:
                image_id = row[0]
                rotation = 0.0 if row[11] == '' else float(row[11])

                rotation_by_image_id[image_id] = rotation

        return rotation_by_image_id

    def _list_images(self):
        image_ids = set()
        bboxes = defaultdict(list)

        with open(os.path.join(self._root, 'labels', 'detections.csv'), newline='') as detection_file:
            detection_reader = csv.reader(detection_file, delimiter=',', quotechar='"')
            next(detection_reader)

            for row in detection_reader:
                image_id = row[0]
                class_id = row[2]
                x_min = float(row[4])
                x_max = float(row[5])
                y_min = float(row[6])
                y_max = float(row[7])

                if class_id not in self._indexes_by_class_id:
                    continue

                image_ids.add(image_id)
                bboxes[image_id].append({
                    'class_index': self._indexes_by_class_id[class_id],
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max,
                })

        image_ids = list(image_ids)
        image_ids.sort()

        images = []
        for i, image_id in enumerate(image_ids):
            path = os.path.join(self._root, 'data', '{}.jpg'.format(image_id))
            if not self._is_valid_image_path(path):
                continue
            images.append({
                'image_id': image_id,
                'path': path,
                'rotation': self._rotation_by_image_id[image_id]
            })

        return images, dict(bboxes)

    def _is_valid_image_path(self, path):
        try:
            _ = Image.open(path).verify()
            return True
        except:
            return False

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = Image.open(self._images[index]['path']).convert('RGB')
        target = self._bboxes[self._images[index]['image_id']]

        target = copy.deepcopy(target)
        image, target = self._rotate(image, target, self._images[index]['rotation'])

        initial_width, initial_height = image.size
        target = [self._scale_target(t, initial_width, initial_height) for t in target]

        image, target, transforms_metadata = self._transforms(image, target)
        metadata = {
            'initial_width': initial_width,
            'initial_height': initial_height,
            'scale': transforms_metadata['scale']
        }
        return image, target, metadata

    def _rotate(self, image, target, rotation):
        return image.rotate(rotation, expand=True), [self._rotate_target(t, rotation) for t in target]

    def _rotate_target(self, target, rotation):
        if rotation == 0.0:
            return target
        elif rotation == 90.0:
            return {
                'class_index': target['class_index'],
                'x_min': target['y_min'],
                'x_max': target['y_max'],
                'y_min': 1.0 - target['x_max'],
                'y_max': 1.0 - target['x_min'],
            }
        elif rotation == 180.0:
            return {
                'class_index': target['class_index'],
                'x_min': 1.0 - target['x_max'],
                'x_max': 1.0 - target['x_min'],
                'y_min': 1.0 - target['y_max'],
                'y_max': 1.0 - target['y_min'],
            }
        elif rotation == 270.0:
            return {
                'class_index': target['class_index'],
                'x_min': 1.0 - target['y_max'],
                'x_max': 1.0 - target['y_min'],
                'y_min': target['x_min'],
                'y_max': target['x_max'],
            }
        else:
            raise ValueError('Invalid rotation')

    def _scale_target(self, target, width, height):
        return {
            'class_index': target['class_index'],
            'x_min': target['x_min'] * width,
            'x_max': target['x_max'] * width,
            'y_min': target['y_min'] * height,
            'y_max': target['y_max'] * height,
        }
