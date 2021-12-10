import os
import csv

import torch
from torch.utils.data import Dataset
import torchaudio


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
