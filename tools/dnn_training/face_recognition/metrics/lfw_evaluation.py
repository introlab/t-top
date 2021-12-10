import os
import time

import numpy as np

from PIL import Image

import torch

from tqdm import tqdm

from common.metrics import RocDistancesThresholdsEvaluation


class LfwEvaluation(RocDistancesThresholdsEvaluation):
    def __init__(self, model, device, transforms, lwf_dataset_root, output_path):
        super(LfwEvaluation, self).__init__(output_path, thresholds=np.arange(0, 2, 0.001))

        self._model = model
        self._device = device
        self._transforms = transforms
        self._lfw_dataset_root = lwf_dataset_root

        self._image_pairs = self._read_image_pairs()

    def _read_image_pairs(self):
        image_pairs = []
        with open(os.path.join(self._lfw_dataset_root, 'pairs.txt'), 'r') as f:
            lines = f.readlines()[1:]

        for line in lines:
            p = line.strip().split()
            if len(p) == 3:
                image_path1 = os.path.join(self._lfw_dataset_root, p[0], p[0] + '_' + ('%04d' % int(p[1])) + '.jpg')
                image_path2 = os.path.join(self._lfw_dataset_root, p[0], p[0] + '_' + ('%04d' % int(p[2])) + '.jpg')
                is_same_person = True
            elif len(p) == 4:
                image_path1 = os.path.join(self._lfw_dataset_root, p[0], p[0] + '_' + ('%04d' % int(p[1])) + '.jpg')
                image_path2 = os.path.join(self._lfw_dataset_root, p[2], p[2] + '_' + ('%04d' % int(p[3])) + '.jpg')
                is_same_person = False
            else:
                raise ValueError('Invalid pair values ({})'.format(p))

            if os.path.exists(image_path1) and os.path.exists(image_path2):
                image_pairs.append((image_path1, image_path2, is_same_person))

        return image_pairs

    def _calculate_distances(self):
        distances = []

        for image_path1, image_path2, _ in tqdm(self._image_pairs):
            image1, image2 = self._load_images(image_path1, image_path2)

            embedding1 = self._model(image1.to(self._device))[0]
            embedding2 = self._model(image2.to(self._device))[0]
            distance = torch.dist(embedding1, embedding2, p=2).item()
            distances.append(distance)

        return torch.tensor(distances)

    def _load_images(self, image_path1, image_path2):
        image1 = self._load_image(image_path1).unsqueeze(0)
        image2 = self._load_image(image_path2).unsqueeze(0)

        return image1, image2

    def _load_image(self, path):
        image = Image.open(path).convert('RGB')
        if self._transforms is not None:
            image = self._transforms(image)

        return image

    def _get_is_same_person_target(self):
        return torch.tensor([image_pair[2] for image_pair in self._image_pairs])
