import os

from torch.utils.data import Dataset
import torchaudio


class LibriSpeechDataset(Dataset):
    def __init__(self, root, split=None, transforms=None):
        if split == 'training':
            self._sounds = self._list_sounds(os.path.join(root, 'train-clean-100'))
            self._sounds.extend(self._list_sounds(os.path.join(root, 'train-clean-360')))
        elif split == 'validation':
            self._sounds = self._list_sounds(os.path.join(root, 'dev-clean'))
        elif split == 'testing':
            self._sounds = self._list_sounds(os.path.join(root, 'test-clean'))
        else:
            raise ValueError('Invalid split')

        self._transforms = transforms

    def _list_sounds(self, directory):
        sounds = []
        for root, _, filenames in os.walk(directory):
            for filename in filter(lambda x: x.endswith('.flac'), filenames):
                sounds.append(os.path.join(root, filename))

        return sounds

    def __len__(self):
        return len(self._sounds)

    def __getitem__(self, index):
        waveform, sample_rate = torchaudio.load(self._sounds[index])

        metadata = {
            'original_sample_rate': sample_rate
        }

        if self._transforms is not None:
            waveform, _ = self._transforms(waveform, metadata)

        return waveform, waveform.clone()
