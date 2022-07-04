import os
import csv
import copy

from PIL import Image

from torch.utils.data import Dataset


class OpenImages(Dataset):
    def __init__(self, root, split=None, transforms=None, ignored_class_names=None):
        if split == 'training':
            self._root = os.path.join(root, 'train')
        elif split == 'validation':
            self._root = os.path.join(root, 'validation')
        elif split == 'testing':
            self._root = os.path.join(root, 'test')
        else:
            raise ValueError('Invalid split')

        if ignored_class_names is None:
            ignored_class_names = []
        if transforms is None:
            raise ValueError('Invalid transforms')
        self._transforms = transforms

        available_class_ids = self._list_available_class_ids(os.path.join(root, 'train'))
        available_class_ids |= self._list_available_class_ids(os.path.join(root, 'validation'))
        available_class_ids |= self._list_available_class_ids(os.path.join(root, 'test'))
        self._indexes_by_class_id, self._class_names = self._list_classes(available_class_ids, ignored_class_names)
        self._rotation_by_image_id = self._list_rotations()
        self._images, self._targets_by_image_id = self._list_images()

    def _list_available_class_ids(self, root):
        raise NotImplementedError()

    def _list_available_class_ids_from_csv(self, root, label_filename, class_id_index):
        class_ids = set()

        with open(os.path.join(root, 'labels', label_filename), newline='') as detection_file:
            detection_reader = csv.reader(detection_file, delimiter=',', quotechar='"')
            next(detection_reader)

            for row in detection_reader:
                class_id = row[class_id_index]
                class_ids.add(class_id)

        return class_ids

    def _list_classes(self, available_class_ids, ignored_class_names=None):
        if ignored_class_names is None:
            ignored_class_names = []

        classes = []
        available_class_ids = set(available_class_ids)
        ignored_class_names = set(ignored_class_names)

        with open(os.path.join(self._root, 'metadata', 'classes.csv'), newline='') as class_file:
            class_reader = csv.reader(class_file, delimiter=',', quotechar='"')
            for id, class_name in class_reader:
                if class_name in ignored_class_names or id not in available_class_ids:
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
        raise NotImplementedError()

    def _image_ids_to_images(self, image_ids):
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

        return images

    def _is_valid_image_path(self, path):
        try:
            _ = Image.open(path).verify()
            return True
        except:
            return False

    def _rotate_image(self, image, rotation):
        return image.rotate(rotation, expand=True)

    def _rotate_target(self, target, rotation):
        return target

    def _scale_target(self, target, width, height):
        return target

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = Image.open(self._images[index]['path']).convert('RGB')
        target = self._targets_by_image_id[self._images[index]['image_id']]
        target = self._load_target(target)

        image = self._rotate_image(image, self._images[index]['rotation'])
        target = self._rotate_target(target, self._images[index]['rotation'])

        initial_width, initial_height = image.size
        target = self._scale_target(target, initial_width, initial_height)

        image, target, transforms_metadata = self._transforms(image, target)
        metadata = {
            'initial_width': initial_width,
            'initial_height': initial_height,
        }
        metadata.update(transforms_metadata)

        return image, target, metadata

    def _load_target(self, target):
        return copy.deepcopy(target)
