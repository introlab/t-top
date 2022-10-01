import os
import csv

import torch
from torch.utils.data import Dataset
import torchaudio


# 1 - AP of each class during the training
FSDK50k_POS_WEIGHT = torch.tensor([0.5651, 0.1267, 0.1715, 0.4201, 0.1626, 0.1172, 0.1135, 0.1636, 0.2194,
                                   0.1674, 0.6011, 0.1615, 0.4142, 0.1635, 0.1679, 0.3407, 0.6103, 0.5347,
                                   0.4462, 0.0848, 0.0880, 0.2101, 0.1174, 0.3452, 0.3135, 0.4181, 0.3304,
                                   0.5456, 0.1882, 0.7136, 0.2944, 0.5132, 0.1999, 0.2401, 0.1726, 0.4264,
                                   0.3671, 0.5323, 0.3620, 0.2148, 0.2273, 0.3999, 0.5439, 0.5621, 0.2584,
                                   0.0998, 0.8355, 0.5168, 0.1512, 0.2777, 0.3545, 0.3391, 0.5969, 0.7260,
                                   0.3474, 0.3558, 0.4137, 0.0947, 0.3535, 0.1371, 0.1311, 0.1961, 0.2600,
                                   0.1426, 0.6236, 0.2040, 0.4149, 0.1309, 0.1180, 0.1205, 0.4038, 0.5866,
                                   0.2246, 0.1319, 0.1774, 0.1733, 0.5300, 0.1873, 0.6034, 0.3428, 0.3794,
                                   0.1900, 0.3687, 0.6971, 0.3248, 0.3333, 0.2516, 0.1687, 0.1891, 0.4374,
                                   0.0784, 0.4582, 0.3191, 0.4168, 0.5475, 0.2403, 0.1014, 0.2052, 0.1083,
                                   0.5914, 0.1457, 0.0759, 0.5624, 0.2385, 0.1468, 0.4627, 0.2696, 0.1478,
                                   0.2991, 0.2345, 0.2632, 0.2202, 0.1617, 0.1774, 0.5143, 0.3080, 0.1851,
                                   0.4282, 0.2364, 0.4489, 0.0200, 0.0198, 0.3145, 0.2032, 0.7746, 0.0797,
                                   0.1643, 0.0784, 0.4913, 0.1963, 0.4238, 0.2741, 0.4078, 0.2325, 0.4009,
                                   0.3452, 0.3277, 0.7384, 0.2016, 0.1768, 0.1613, 0.3781, 0.3899, 0.7663,
                                   0.1736, 0.1545, 0.4978, 0.2567, 0.2145, 0.2204, 0.1840, 0.4632, 0.3326,
                                   0.2809, 0.4492, 0.5211, 0.1237, 0.3485, 0.1280, 0.2009, 0.3581, 0.4747,
                                   0.3072, 0.1372, 0.2562, 0.3254, 0.1270, 0.6841, 0.4878, 0.1764, 0.5468,
                                   0.2546, 0.2394, 0.9123, 0.3800, 0.2726, 0.3231, 0.5599, 0.4232, 0.4885,
                                   0.7342, 0.0858, 0.5061, 0.4035, 0.1759, 0.1579, 0.5611, 0.2854, 0.4690,
                                   0.3196, 0.2173, 0.3066, 0.1498, 0.5192, 0.2322, 0.0570, 0.7181, 0.5055,
                                   0.2353, 0.3516], dtype=torch.float64)


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
