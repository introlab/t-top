import os

from torch.utils.data import Dataset
import torchaudio


class AudioDescriptorDataset(Dataset):
    def __init__(self, root, split=None, transforms=None):
        sounds_path = os.path.join(root, 'sounds')
        self._class_names = [o for o in os.listdir(sounds_path) if os.path.isdir(os.path.join(sounds_path, o))]
        self._class_names.sort()

        if split == 'training':
            self._sounds, self._sounds_by_class = self._list_sounds(root, 'train.txt')
        elif split == 'validation':
            self._sounds, self._sounds_by_class = self._list_sounds(root, 'validation.txt')
        elif split == 'testing':
            self._sounds, self._sounds_by_class = self._list_sounds(root, 'test.txt')
        else:
            raise ValueError('Invalid split')

        self._transforms = transforms

    def _list_sounds(self, root, filename):
        class_indexes_by_class_name = {self._class_names[i]: i for i in range(len(self._class_names))}

        with open(os.path.join(root, filename), 'r') as sound_name_file:
            sound_names = [line.strip() for line in sound_name_file.readlines()]

        sounds = []
        sounds_by_class = [[] for _ in self._class_names]
        for sound_name in sound_names:
            class_name, video_id, filename = sound_name.split(' ')

            class_index = class_indexes_by_class_name[class_name]
            sound_index = len(sounds)
            sounds.append({
                'path': os.path.join(root, 'sounds', class_name, video_id, filename),
                'class_index': class_index
            })
            sounds_by_class[class_index].append({'index': sound_index})

        return sounds, sounds_by_class

    def __len__(self):
        return len(self._sounds)

    def __getitem__(self, index):
        waveform, sample_rate = torchaudio.load(self._sounds[index]['path'])
        class_index = self._sounds[index]['class_index']

        metadata = {
            'original_sample_rate': sample_rate
        }

        if self._transforms is not None:
            waveform, class_index, metadata = self._transforms(waveform, class_index, metadata)

        return waveform, class_index, metadata

    def lens_by_class(self):
        return [len(x) for x in self._sounds_by_class]

    def get_all_indexes(self, class_, index):
        return self._sounds_by_class[class_][index]['index']

    def transforms(self):
        return self._transforms
