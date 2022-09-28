import os
import csv

import torch
from torch.utils.data import Dataset
import torchaudio

FSDK50k_POS_WEIGHT = torch.tensor(
    [185.1709, 500.9314, 75.1860, 187.2243, 26.4662, 10.7505, 92.0855, 94.5168, 150.0236, 126.9925, 249.9657, 42.2407,
     163.6206, 395.8760, 28.6108, 52.2747, 367.3237, 500.9314, 249.9657, 23.8047, 48.4657, 76.8070, 163.6206, 178.6386,
     327.1859, 135.1622, 50.2482, 219.6767, 116.6943, 111.7687, 156.0460, 198.9883, 294.9364, 199.7726, 190.0336,
     117.2379, 92.4252, 460.2342, 362.0993, 89.4541, 145.6963, 95.7807, 228.5830, 405.3254, 131.9792, 216.8596,
     272.7808, 155.5658, 178.6386, 184.4964, 460.2342, 116.6943, 174.9347, 184.4964, 338.0530, 333.6209, 126.9925,
     55.6966, 105.8831, 54.0505, 36.5070, 6.6895, 32.6380, 354.5347, 231.7136, 241.6398, 153.6737, 32.8157, 126.6733,
     73.5226, 35.3615, 231.7136, 35.8854, 80.2651, 246.3285, 78.9953, 344.9257, 256.2714, 99.5835, 108.1620, 349.6644,
     181.1957, 464.4273, 415.2358, 464.4273, 271.3245, 40.2546, 357.0210, 183.1619, 415.2358, 21.9377, 491.2788,
     105.2178, 217.7906, 217.7906, 57.8471, 228.5830, 217.7906, 83.7632, 176.1523, 34.9782, 8.0566, 239.3615, 84.6137,
     27.3797, 202.9721, 136.2574, 42.1678, 30.9781, 79.6252, 340.3133, 58.6702, 118.8993, 242.7952, 429.2269, 32.3748,
     203.7880, 247.5291, 24.7790, 203.7880, 2.4736, 2.4821, 153.2078, 146.1178, 392.8231, 11.8733, 58.8795, 20.7860,
     157.5046, 238.2383, 313.0920, 398.9766, 415.2358, 60.2404, 71.3121, 262.9021, 460.2342, 169.6567, 160.5047,
     41.5932, 301.9408, 153.2078, 294.9364, 309.2849, 164.6861, 134.8011, 331.4481, 99.3863, 129.2723, 375.4485,
     70.4045, 124.7912, 386.8560, 378.2370, 91.7482, 172.5491, 56.7844, 408.5760, 21.7138, 425.6417, 97.8359, 83.4835,
     160.5047, 218.7296, 131.2920, 526.8041, 169.6567, 132.3255, 131.6347, 73.7401, 107.9298, 86.8165, 84.1864,
     468.6972, 300.1588, 182.5018, 49.9930, 182.5018, 113.0245, 167.4112, 331.4481, 79.3721, 500.9314, 118.0628,
     12.7997, 278.7650, 87.2707, 25.5958, 110.7838, 207.9673, 207.9673, 136.6263, 20.4034, 118.6192, 429.2269, 18.2978,
     117.5116, 148.2624, 256.2714, 128.6127])


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
