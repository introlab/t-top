import os
import time

import numpy as np

import torch
import torchaudio

from tqdm import tqdm

from common.metrics import RocDistancesThresholdsEvaluation


class AudioDescriptorEvaluation(RocDistancesThresholdsEvaluation):
    def __init__(self, model, device, transforms, dataset_root, output_path):
        super(AudioDescriptorEvaluation, self).__init__(output_path, thresholds=np.arange(0, 2, 0.00001))

        self._model = model
        self._device = device
        self._transforms = transforms
        self._dataset_root = dataset_root

        self._sound_pairs = self._read_sound_pairs()

    def _read_sound_pairs(self):
        sound_pairs = []
        with open(os.path.join(self._dataset_root, 'test_pairs.txt'), 'r') as f:
            lines = f.readlines()[1:]

        for line in lines:
            values = line.strip().split(' ')
            if len(values) == 4:
                directory, filename1, filename2, is_same_person = values

                sound_path1 = os.path.join(self._dataset_root, directory, filename1)
                sound_path2 = os.path.join(self._dataset_root, directory, filename2)
                is_same_person = True if is_same_person == 'true' else False
            elif len(values) == 6:
                class_name1, video_id1, filename1, class_name2, video_id2, filename2 = values

                sound_path1 = os.path.join(self._dataset_root, 'sounds', class_name1, video_id1, filename1)
                sound_path2 = os.path.join(self._dataset_root, 'sounds', class_name2, video_id2, filename2)
                is_same_person = class_name1 == class_name2
            else:
                raise ValueError('Unsupported format ({})'.format(line))

            if os.path.exists(sound_path1) and os.path.exists(sound_path2):
                sound_pairs.append((sound_path1, sound_path2, is_same_person))

        return sound_pairs

    def _calculate_distances(self):
        distances = []

        for sound_path1, sound_path2, _ in tqdm(self._sound_pairs):
            sound1, sound2 = self._load_sounds(sound_path1, sound_path2)

            model_output1 = self._model_forward(sound1)[0]
            model_output2 = self._model_forward(sound2)[0]
            distance = torch.dist(model_output1, model_output2, p=2).item()
            distances.append(distance)

        return torch.tensor(distances)

    def _load_sounds(self, sound_path1, sound_path2):
        sound1 = self._load_sound(sound_path1).unsqueeze(0)
        sound2 = self._load_sound(sound_path2).unsqueeze(0)

        return sound1, sound2

    def _load_sound(self, path):
        waveform, sample_rate = torchaudio.load(path)
        class_index = 0

        metadata = {
            'original_sample_rate': sample_rate
        }

        if self._transforms is not None:
            waveform, _, _ = self._transforms(waveform, class_index, metadata)

        return waveform

    def _model_forward(self, sound):
        model_output = self._model(sound.to(self._device))
        if isinstance(model_output, tuple):
            model_output = model_output[0]
        return model_output

    def _get_is_same_person_target(self):
        return torch.tensor([sound_pair[2] for sound_pair in self._sound_pairs])
