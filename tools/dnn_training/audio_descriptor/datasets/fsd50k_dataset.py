import os
import csv

import torch
from torch.utils.data import Dataset
import torchaudio

FSDK50k_POS_WEIGHT = torch.tensor([0.7338, 0.8041, 0.6351, 0.8928, 0.5104, 0.4726, 0.1612, 0.5149, 0.4401,
                                   0.6335, 0.8455, 0.5956, 0.7096, 0.3779, 0.5692, 0.6052, 0.9863, 0.8621,
                                   0.8431, 0.6962, 0.5056, 0.6680, 0.2258, 0.9046, 0.9028, 0.7834, 0.7219,
                                   0.8295, 0.2960, 0.7174, 0.3468, 0.6835, 0.5297, 0.8753, 0.6377, 0.7965,
                                   0.6251, 0.9220, 0.5947, 0.6936, 0.7650, 0.8653, 0.6766, 0.9115, 0.5325,
                                   0.9313, 0.8945, 0.9072, 0.3052, 0.6038, 0.7854, 0.5308, 0.5061, 0.9271,
                                   0.7507, 0.9641, 0.7642, 0.2918, 0.8807, 0.5610, 0.4850, 0.4533, 0.7345,
                                   0.7911, 0.7891, 0.8974, 0.7543, 0.4572, 0.5238, 0.6131, 0.5744, 0.8436,
                                   0.5576, 0.6896, 0.7624, 0.7615, 0.7788, 0.6311, 0.8367, 0.7113, 0.9351,
                                   0.6191, 0.8664, 0.7885, 0.6974, 0.7209, 0.6433, 0.3915, 0.8157, 0.9371,
                                   0.3868, 0.8175, 0.6616, 0.8593, 0.9325, 0.6868, 0.5081, 0.9128, 0.4899,
                                   0.9232, 0.4754, 0.3290, 0.7942, 0.6604, 0.7809, 0.6588, 0.5945, 0.5489,
                                   0.5455, 0.7772, 0.8825, 0.6914, 0.5136, 0.8564, 0.9264, 0.8001, 0.2907,
                                   0.7851, 0.5862, 0.7988, 0.1782, 0.1717, 0.8442, 0.8762, 0.8615, 0.3469,
                                   0.8515, 0.3823, 0.6947, 0.9003, 0.9052, 0.4125, 0.7261, 0.7648, 0.6972,
                                   0.9166, 0.9427, 0.9608, 0.6545, 0.5231, 0.8408, 0.7982, 0.8674, 0.8879,
                                   0.6754, 0.4362, 0.9303, 0.5483, 0.7874, 0.8617, 0.5708, 0.6580, 0.6056,
                                   0.8703, 0.8085, 0.9041, 0.5257, 0.5094, 0.5711, 0.9553, 0.6207, 0.8447,
                                   0.4141, 0.7199, 0.8452, 0.6891, 0.6261, 0.9688, 0.7129, 0.6197, 0.9018,
                                   0.3128, 0.3049, 0.9547, 0.7415, 0.4216, 0.8908, 0.8353, 0.8915, 0.7006,
                                   0.9723, 0.3804, 0.9310, 0.6987, 0.5095, 0.7279, 0.8016, 0.4709, 0.5535,
                                   0.8422, 0.7171, 0.7837, 0.5120, 0.9053, 0.7978, 0.7037, 0.9545, 0.9135,
                                   0.8693, 0.4788], dtype=torch.float64)


class Fsd50kDataset(Dataset):
    def __init__(self, root, split=None, transforms=None):
        self._class_indexes_by_name = self._list_classes(root)

        if split == 'training':
            self._sounds = self._list_sounds(root, 'dev')
        elif split == 'validation':
            self._sounds = self._list_sounds(root, 'eval')
        else:
            raise ValueError('Invalid split')

        self._transforms = transforms

    def _list_classes(self, root):
        class_indexes_by_name = {}
        with open(os.path.join(root, 'FSD50K.ground_truth', 'vocabulary.csv'), newline='') as vocabulary_file:
            vocabulary_reader = csv.reader(vocabulary_file, delimiter=',', quotechar='"')
            for class_index, class_name, _ in vocabulary_reader:
                class_indexes_by_name[class_name] = int(class_index)

        return class_indexes_by_name

    def _list_sounds(self, root, fsd50k_split):
        sounds = []
        with open(os.path.join(root, 'FSD50K.ground_truth', '{}.csv'.format(fsd50k_split)), newline='') as sound_file:
            sound_reader = csv.reader(sound_file, delimiter=',', quotechar='"')
            next(sound_reader)
            for row in sound_reader:
                id = row[0]
                class_names = row[1]
                class_names = class_names.split(',')
                sounds.append({
                    'path': os.path.join(root, 'FSD50K.{}_audio'.format(fsd50k_split), '{}.wav'.format(id)),
                    'target': self._create_target(class_names)
                })

        return sounds

    def _create_target(self, class_names):
        target = torch.zeros(len(self._class_indexes_by_name), dtype=torch.float)
        for class_name in class_names:
            target[self._class_indexes_by_name[class_name]] = 1.0

        return target

    def __len__(self):
        return len(self._sounds)

    def __getitem__(self, index):
        waveform, sample_rate = torchaudio.load(self._sounds[index]['path'])
        target = self._sounds[index]['target'].clone()

        metadata = {
            'original_sample_rate': sample_rate
        }

        if self._transforms is not None:
            waveform, target, metadata = self._transforms(waveform, target, metadata)

        return waveform, target, metadata

    def transforms(self):
        return self._transforms
