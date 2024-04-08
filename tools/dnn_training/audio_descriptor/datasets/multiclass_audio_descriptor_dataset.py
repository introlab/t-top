import os
import random

import torch
from torch.utils.data import Dataset
import torchaudio


class MulticlassAudioDescriptorDataset(Dataset):
    def __init__(self, root, split=None, transforms=None, mixup_rate=0.5, mixup_alpha=10.0, enhanced_targets=True):
        self._root = root
        self._class_indexes_by_name = self._list_classes(root)

        self._sounds = self._list_sounds(root, split, enhanced_targets)
        self._transforms = transforms

        self._mixup_rate = mixup_rate if split == 'training' else -1.0
        self._mixup_alpha = mixup_alpha

    def _list_classes(self, root):
        raise NotImplementedError()

    def _list_sounds(self, root, split, enhanced_targets):
        raise NotImplementedError()

    def class_names(self):
        class_names = [''] * len(self._class_indexes_by_name)
        for name, index in self._class_indexes_by_name.items():
            class_names[index] = name
        return class_names

    def _create_target(self, class_names):
        target = torch.zeros(len(self._class_indexes_by_name), dtype=torch.float)
        for class_name in class_names:
            target[self._class_indexes_by_name[class_name]] = 1.0

        return target

    def class_count(self):
        if len(self._sounds) > 0:
            return self._sounds[0]['target'].size(0)
        else:
            return 0

    def __len__(self):
        return len(self._sounds)

    def __getitem__(self, index):
        waveform, target, metadata = self._get_item_without_mixup(index)

        if random.random() < self._mixup_rate:
            mixup_index = random.randrange(len(self._sounds))
            mixup_waveform, mixup_target, _ = self._get_item_without_mixup(mixup_index)
            l = random.betavariate(self._mixup_alpha, self._mixup_alpha)

            waveform = l * waveform + (1.0 - l) * mixup_waveform
            target = l * target + (1.0 - l) * mixup_target

        return waveform, target, metadata

    def get_target(self, index):
        return self._sounds[index]['target'].clone()

    def _get_item_without_mixup(self, index):
        waveform, sample_rate = torchaudio.load(os.path.join(self._root, self._sounds[index]['path']))
        target = self._sounds[index]['target'].clone()

        metadata = {
            'original_sample_rate': sample_rate
        }

        if self._transforms is not None:
            waveform, target, metadata = self._transforms(waveform, target, metadata)

        return waveform, target, metadata

    def transforms(self):
        return self._transforms


