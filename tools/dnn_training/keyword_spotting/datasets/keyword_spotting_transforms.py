import random
import os

import torchaudio
import torchaudio.transforms as transforms

from common.datasets.audio_transform_utils import resample, resize_waveform, resize_waveform_random, normalize


class _KeywordSpottingTransforms:
    def __init__(self, sample_rate=16000, waveform_size=16000, n_mfcc=40, window_size_ms=40):
        self._sample_rate = sample_rate
        self._waveform_size = waveform_size

        melkwargs = {
            'n_fft': int(self._sample_rate / 1000 * window_size_ms)
        }
        self._mfcc_transform = transforms.MFCC(sample_rate=self._sample_rate, n_mfcc=n_mfcc, melkwargs=melkwargs)

    def __call__(self, waveform, target, metadata):
        raise NotImplementedError()


class KeywordSpottingTrainingTransforms(_KeywordSpottingTransforms):
    def __init__(self, noise_root, sample_rate=16000, waveform_size=16000, n_mfcc=40, window_size_ms=40,
                 noise_volume=0.25, noise_p=0.5):
        super(KeywordSpottingTrainingTransforms, self).__init__(sample_rate=sample_rate,
                                                                waveform_size=waveform_size,
                                                                n_mfcc=n_mfcc,
                                                                window_size_ms=window_size_ms)
        self._noise_volume = noise_volume
        self._noise_p = noise_p

        self._noises = self._load_noises(noise_root)

    def _load_noises(self, noise_root):
        noise_paths = [o for o in os.listdir(noise_root) if o.endswith('.wav')]

        noises = []
        for noise_path in noise_paths:
            waveform, sample_rate = torchaudio.load(os.path.join(noise_root, noise_path))
            noises.append(resample(waveform, sample_rate, self._sample_rate))

        return noises

    def __call__(self, waveform, target, metadata):
        waveform = resample(waveform, metadata['original_sample_rate'], self._sample_rate)
        waveform = resize_waveform_random(waveform, self._waveform_size)

        if not metadata['is_noise'] and random.random() < self._noise_p:
            waveform = self._add_noise(waveform)

        waveform = normalize(waveform)
        return self._mfcc_transform(waveform), target, metadata

    def _add_noise(self, waveform):
        noise = random.choice(self._noises)
        i = random.randrange(noise.size()[1] - self._waveform_size)
        return noise[:, i:(i + self._waveform_size)] * self._noise_volume + waveform


class KeywordSpottingValidationTransforms(_KeywordSpottingTransforms):
    def __call__(self, waveform, target, metadata):
        waveform = resample(waveform, metadata['original_sample_rate'], self._sample_rate)
        waveform = resize_waveform(waveform, self._waveform_size)

        waveform = normalize(waveform)
        return self._mfcc_transform(waveform), target, metadata
