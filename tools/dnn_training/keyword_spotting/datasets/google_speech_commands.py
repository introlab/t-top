import os

from torch.utils.data import Dataset
import torchaudio

BACKGROUND_NOISE_COUNT_FACTOR = 300


class GoogleSpeechCommands(Dataset):
    def __init__(self, root, split=None, transforms=None):
        self._transforms = transforms

        self._class_names = [o for o in os.listdir(root) if os.path.isdir(os.path.join(root, o))]
        self._class_names.sort()

        # Set '_background_noise_' as the last classes
        self._class_names = self._class_names[1:] + self._class_names[0:1]

        if split == 'training':
            all_sounds = self._list_training_sounds(root)
        elif split == 'validation':
            all_sounds = self._list_validation_sounds(root)
        elif split == 'testing':
            all_sounds = self._list_testing_sounds(root)
        else:
            raise ValueError('Invalid split')

        self._sounds, self._background_noises = self._split_sounds(all_sounds)

    def _list_training_sounds(self, root):
        invalid_training_paths = self._list_invalid_training_paths(root)
        sounds = []

        for class_name in self._class_names:
            sound_names = [o for o in os.listdir(os.path.join(root, class_name)) if o.endswith('.wav')]

            for sound_name in sound_names:
                path = os.path.join(root, class_name, sound_name)

                if path not in invalid_training_paths:
                    sounds.append({
                        'path': path,
                        'class_index': self._class_names.index(class_name)
                    })
        return sounds

    def _list_invalid_training_paths(self, root):
        validation_paths = [o['path'] for o in self._list_validation_sounds(root)]
        testing_paths = [o['path'] for o in self._list_testing_sounds(root)]
        return set(validation_paths + testing_paths)

    def _list_validation_sounds(self, root):
        with open(os.path.join(root, 'validation_list.txt'), 'r') as file:
            lines = file.readlines()
            return self._create_sound_list_from_path_list(root, lines)

    def _list_testing_sounds(self, root):
        with open(os.path.join(root, 'testing_list.txt'), 'r') as file:
            lines = file.readlines()
            return self._create_sound_list_from_path_list(root, lines)

    def _create_sound_list_from_path_list(self, root, lines):
        sounds = []
        for line in lines:
            split_line = line.strip().split('/')
            class_name = split_line[0]
            sound_name = split_line[1]

            sounds.append({
                'path': os.path.join(root, class_name, sound_name),
                'class_index': self._class_names.index(class_name)
            })

        return sounds

    def _split_sounds(self, all_sounds):
        sounds = []
        background_noises = []

        for s in all_sounds:
            if s['class_index'] == len(self._class_names) - 1:
                background_noises.append(s)
            else:
                sounds.append(s)

        return sounds, background_noises

    def __len__(self):
        return len(self._sounds) + BACKGROUND_NOISE_COUNT_FACTOR * len(self._background_noises)

    def __getitem__(self, index):
        if index < len(self._sounds):
            waveform, sample_rate = torchaudio.load(self._sounds[index]['path'])
            class_index = self._sounds[index]['class_index']
            is_noise = False
        else:
            index = (index - len(self._sounds)) % len(self._background_noises)
            waveform, sample_rate = torchaudio.load(self._background_noises[index]['path'])
            class_index = self._background_noises[index]['class_index']
            is_noise = True

        metadata = {
            'original_sample_rate': sample_rate,
            'is_noise': is_noise
        }

        if self._transforms is not None:
            waveform, class_index, metadata = self._transforms(waveform, class_index, metadata)

        return waveform, class_index, metadata
