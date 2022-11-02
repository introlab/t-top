import os
import csv
from collections import defaultdict

from PIL import Image

from common.datasets import OpenImages


class SemanticSegmentationOpenImages(OpenImages):
    def _list_classes(self, available_class_ids, ignored_class_names=None):
        indexes_by_class_id, class_names = super(SemanticSegmentationOpenImages, self)._list_classes(
            available_class_ids, ignored_class_names)

        indexes_by_class_id = {class_id: index + 1 for class_id, index in indexes_by_class_id.items()}
        indexes_by_class_id['__background__'] = 0
        class_names.insert(0, 'background')

        return indexes_by_class_id, class_names

    def _list_images(self):
        image_ids = set()
        masks_by_image_id = defaultdict(list)

        with open(os.path.join(self._root, 'labels', 'segmentations.csv'), newline='') as detection_file:
            segmentation_reader = csv.reader(detection_file, delimiter=',', quotechar='"')
            next(segmentation_reader)

            for row in segmentation_reader:
                mask_filename = row[0]
                image_id = row[1]
                class_id = row[2]

                mask_path = os.path.join(self._root, 'labels', 'masks', mask_filename[0].upper(), mask_filename)
                if class_id not in self._indexes_by_class_id or not self._is_valid_image_path(mask_path):
                    continue

                image_ids.add(image_id)
                masks_by_image_id[image_id].append({
                    'path': mask_path,
                    'class_index': self._class_id_to_index(class_id),
                })

        images = self._image_ids_to_images(image_ids)
        return images, dict(masks_by_image_id)

    def _class_id_to_index(self, class_id):
        return self._indexes_by_class_id[class_id]

    def _load_target(self, target):
        return [(Image.open(t['path']), t['class_index']) for t in target]


KITCHEN_AVAILABLE_CLASS_NAMES = ['Apple', 'Artichoke', 'Bagel', 'Banana', 'Band-aid', 'Beer', 'Bell pepper', 'Book',
                                 'Bottle', 'Bottle opener', 'Bowl', 'Bread', 'Broccoli', 'Burrito', 'Cabbage', 'Cake',
                                 'Cantaloupe', 'Carrot', 'Cheese', 'Chopsticks', 'Cocktail shaker', 'Coffee cup',
                                 'Common fig', 'Cookie', 'Croissant', 'Cucumber', 'Doughnut', 'Drinking straw',
                                 'Food processor', 'Frying pan', 'Garden Asparagus', 'Grape', 'Grapefruit',
                                 'Guacamole', 'Hamburger', 'Hot dog', 'Juice', 'Kitchen knife', 'Lemon', 'Mango',
                                 'Milk', 'Mixing bowl', 'Orange', 'Pancake', 'Person', 'Pizza', 'Pizza cutter',
                                 'Pomegranate', 'Popcorn', 'Potato', 'Pressure cooker', 'Pretzel', 'Sandwich',
                                 'Submarine sandwich', 'Tomato', 'Waffle', 'Wine', 'Winter melon', 'Zucchini']
KITCHEN_AVAILABLE_CLASS_IDS = ['/m/014j1m', '/m/047v4b', '/m/01fb_0', '/m/09qck', '/m/01599', '/m/0jg57', '/m/09728',
                               '/m/0hkxq', '/m/01j3zr', '/m/0fbw6', '/m/0fszt', '/m/0kpt_', '/m/0fj52s', '/m/01nkt',
                               '/m/02p5f1q', '/m/043nyj', '/m/021mn', '/m/015wgc', '/m/015x4r', '/m/0jy4k', '/m/0cjs7',
                               '/m/0388q', '/m/0hqkz', '/m/02g30s', '/m/0cdn1', '/m/01b9xk', '/m/01z1kdw', '/m/09k_b',
                               '/m/0fldg', '/m/04zpv', '/m/0cyhj_', '/m/01dwwc', '/m/0663v', '/m/0jwn_', '/m/01hrv5',
                               '/m/05vtc', '/m/01f91_', '/m/0l515', '/m/06pcq', '/m/07j87', '/m/01dwsz', '/m/081qc',
                               '/m/02cvgx', '/m/027pcv', '/m/0j496', '/m/0bt_c3', '/m/04dr76w', '/m/04f5ws',
                               '/m/04kkgm', '/m/01_5g', '/m/0440zs', '/m/03v5tg', '/m/03y6mg', '/m/04v6l4',
                               '/m/01g317', '/m/058qzx', '/m/03hj559', '/m/08ks85', '/m/0h8ntjv']
KITCHEN_CLASS_COUNT = len(KITCHEN_AVAILABLE_CLASS_IDS) + 1


class SemanticSegmentationKitchenOpenImages(SemanticSegmentationOpenImages):
    def __init__(self, root, split=None, transforms=None):
        super(SemanticSegmentationKitchenOpenImages, self).__init__(root, split=split, transforms=transforms)

        if len(self._indexes_by_class_id) != KITCHEN_CLASS_COUNT:
            raise ValueError('Invalid dataset root')

    def _list_available_class_ids(self, root):
        return set(KITCHEN_AVAILABLE_CLASS_IDS)


PERSON_CLASS_ID = '/m/01g317'
PERSON_OTHER_CLASS_COUNT = 3


class SemanticSegmentationPersonOtherOpenImages(SemanticSegmentationOpenImages):
    def _list_available_class_ids(self, root):
        return self._list_available_class_ids_from_csv(root, 'segmentations.csv', class_id_index=2)

    def _class_id_to_index(self, class_id):
        if class_id == PERSON_CLASS_ID:
            return 1
        else:
            return 2
