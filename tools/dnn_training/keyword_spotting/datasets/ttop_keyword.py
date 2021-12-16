import os

from torch.utils.data import Dataset
import torchaudio


class TtopKeyword(Dataset):
    def __init__(self, root, split=None, transforms=None):
        self._transforms = transforms

        self._ttop_sounds, self._t_and_top_sounds, self._text_sounds, self._noise_sounds = \
            self._list_all_sounds(root, split)
        self._ttop_count_factor, self._text_count_factor, self._noise_count_factor = self._calculate_count_factors()

    def _list_all_sounds(self, root, split):
        if split == 'training':
            ttop_sounds = self._list_sounds(os.path.join(root, 'train', 'ttop'), 0)
            t_and_top_sounds = self._list_sounds(os.path.join(root, 'train', 't_and_top'), 1)
            text_sounds = self._list_sounds(os.path.join(root, 'train', 'text'), 1)
            noise_sounds = self._list_sounds(os.path.join(root, 'train', 'noise'), 1)
        elif split == 'validation':
            ttop_sounds = self._list_sounds(os.path.join(root, 'val', 'ttop'), 0)
            t_and_top_sounds = self._list_sounds(os.path.join(root, 'train', 't_and_top'), 1)
            text_sounds = self._list_sounds(os.path.join(root, 'val', 'text'), 1)
            noise_sounds = []
        elif split == 'testing':
            ttop_sounds = self._list_sounds(os.path.join(root, 'test', 'ttop'), 0)
            t_and_top_sounds = self._list_sounds(os.path.join(root, 'train', 't_and_top'), 1)
            text_sounds = self._list_sounds(os.path.join(root, 'test', 'text'), 1)
            noise_sounds = []
        else:
            raise ValueError('Invalid split')

        return ttop_sounds, t_and_top_sounds, text_sounds, noise_sounds

    def _list_sounds(self, path, class_index):
        filenames = [o for o in os.listdir(path) if o.endswith('.wav')]

        sounds = []
        for filename in filenames:
            sounds.append({
                'path': os.path.join(path, filename),
                'class_index': class_index
            })

        return sounds

    def _calculate_count_factors(self):
        difference = max(0, (len(self._ttop_sounds) - len(self._text_sounds) - len(self._noise_sounds)))

        ttop_count_factor = 3
        if len(self._text_sounds) == 0:
            text_count_factor = 0
            noise_count_factor = difference // len(self._noise_sounds)
        elif len(self._noise_sounds) == 0:
            text_count_factor = difference // len(self._text_sounds)
            noise_count_factor = 0
        else:
            text_count_factor = difference // len(self._text_sounds) // 2
            noise_count_factor = difference // len(self._noise_sounds) // 2

        return ttop_count_factor, text_count_factor, noise_count_factor

    def __len__(self):
        return self._ttop_count_factor * len(self._ttop_sounds) + \
               len(self._t_and_top_sounds) + \
               self._text_count_factor * len(self._text_sounds) + \
               self._noise_count_factor * len(self._noise_sounds)

    def __getitem__(self, index):
        t_and_top_offset = self._ttop_count_factor * len(self._ttop_sounds)
        text_offset = t_and_top_offset + len(self._t_and_top_sounds)
        noise_offset = text_offset + self._text_count_factor * len(self._text_sounds)

        if index < self._ttop_count_factor * len(self._ttop_sounds):
            index = index % len(self._ttop_sounds)
            path = self._ttop_sounds[index]['path']
            class_index = self._ttop_sounds[index]['class_index']
            is_noise = False
        elif index < t_and_top_offset + len(self._t_and_top_sounds):
            index = index - t_and_top_offset
            path = self._t_and_top_sounds[index]['path']
            class_index = self._t_and_top_sounds[index]['class_index']
            is_noise = False
        elif index < text_offset + self._text_count_factor * len(self._text_sounds):
            index = (index - text_offset) % len(self._text_sounds)
            path = self._text_sounds[index]['path']
            class_index = self._text_sounds[index]['class_index']
            is_noise = False
        else:
            index = (index - noise_offset) % len(self._noise_sounds)
            path = self._noise_sounds[index]['path']
            class_index = self._noise_sounds[index]['class_index']
            is_noise = True

        waveform, sample_rate = torchaudio.load(path)
        metadata = {
            'original_sample_rate': sample_rate,
            'is_noise': is_noise
        }

        if self._transforms is not None:
            waveform, class_index, metadata = self._transforms(waveform, class_index, metadata)

        return waveform, class_index, metadata
