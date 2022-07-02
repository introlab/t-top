import os
import csv
from collections import defaultdict

from common.datasets.open_images import OpenImages

HUMAN_BODY_PART_CLASS_NAMES = ['Human eye', 'Human beard', 'Human mouth', 'Human foot', 'Human leg', 'Human ear',
                               'Human hair', 'Human head', 'Human face', 'Human arm', 'Human nose', 'Human hand']
CLASS_COUNT_WITHOUT_HUMAN_BODY_PART = 589


class ObjectDetectionOpenImages(OpenImages):
    def _list_available_class_ids(self, root):
        return self._list_available_class_ids_from_csv(root, 'detections.csv', class_id_index=2)

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

        images = self._image_ids_to_images(image_ids)
        return images, dict(bboxes)

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
        return [{
            'class_index': t['class_index'],
            'x_min': t['x_min'] * width,
            'x_max': t['x_max'] * width,
            'y_min': t['y_min'] * height,
            'y_max': t['y_max'] * height,
        } for t in target]
