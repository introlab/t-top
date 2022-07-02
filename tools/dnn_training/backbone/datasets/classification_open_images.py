import os
import csv

from common.datasets import OpenImages


CLASS_COUNT = 601


class ClassificationOpenImages(OpenImages):
    def __init__(self, root, split=None, image_transforms=None):
        def transforms(image, target):
            return image_transforms(image), target, {}
        super(ClassificationOpenImages, self).__init__(root, split=split, transforms=transforms)

        if len(self._indexes_by_class_id) != CLASS_COUNT:
            raise ValueError('Invalid dataset root')

    def _list_available_class_ids(self, root):
        class_ids = set()

        with open(os.path.join(root, 'labels', 'classifications.csv'), newline='') as detection_file:
            detection_reader = csv.reader(detection_file, delimiter=',', quotechar='"')
            next(detection_reader)

            for row in detection_reader:
                class_id = row[2]
                class_ids.add(class_id)

        return class_ids

    def _list_images(self):
        images = []
        class_indexes_by_image_id = {}

        with open(os.path.join(self._root, 'labels', 'classifications.csv'), newline='') as detection_file:
            detection_reader = csv.reader(detection_file, delimiter=',', quotechar='"')
            next(detection_reader)

            for row in detection_reader:
                image_id = row[0]
                class_id = row[2]

                if class_id not in self._indexes_by_class_id:
                    continue

                path = os.path.join(self._root, 'data', '{}.jpg'.format(image_id))
                if not self._is_valid_image_path(path):
                    continue
                images.append({
                    'image_id': image_id,
                    'path': path,
                    'rotation': self._rotation_by_image_id[image_id]
                })

                class_indexes_by_image_id[image_id] = self._indexes_by_class_id[class_id]

        return images, class_indexes_by_image_id





