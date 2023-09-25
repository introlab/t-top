import os
import random

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms

from common.datasets.audio_transform_utils import to_mono, resample, resize_waveform, resize_waveform_random, \
    normalize, RandomPitchShift, RandomTimeStretch


class LogModule(nn.Module):
    def forward(self, x, eps=1e-6):
        return torch.log10(x + eps)


class _AudioDescriptorTransforms:
    def __init__(self, sample_rate=16000, waveform_size=64000, n_features=128, n_fft=400,
                 audio_transform_type='mel_spectrogram'):
        self._sample_rate = sample_rate
        self._waveform_size = waveform_size

        if audio_transform_type == 'mfcc':
            melkwargs = {
                'n_fft': n_fft
            }
            self._audio_transform = transforms.MFCC(sample_rate=self._sample_rate,
                                                    n_mfcc=n_features,
                                                    melkwargs=melkwargs)
        elif audio_transform_type == 'mel_spectrogram':
            self._audio_transform = transforms.MelSpectrogram(sample_rate=self._sample_rate,
                                                              n_fft=n_fft,
                                                              n_mels=n_features)
        elif audio_transform_type == 'log_mel_spectrogram':
            self._audio_transform = nn.Sequential(
                transforms.MelSpectrogram(sample_rate=self._sample_rate,
                                          n_fft=n_fft,
                                          n_mels=n_features),
                LogModule()
            )
        elif audio_transform_type == 'spectrogram':
            if n_features != (n_fft // 2 + 1):
                raise ValueError('n_features must be equal to (n_fft // 2 + 1) '
                                 'when audio_transform_type is spectrogram.')
            self._audio_transform = transforms.Spectrogram(n_fft=n_fft, power=2)
        else:
            raise ValueError('Invalid audio_transform_type')

    def __call__(self, waveform, target, metadata):
        raise NotImplementedError()


class AudioDescriptorTrainingTransforms(_AudioDescriptorTransforms):
    def __init__(self, sample_rate=16000, waveform_size=64000, n_features=128, n_fft=400,
                 min_time_stretch=0.9, max_time_stretch= 1.1, time_stretching_p=0.5,
                 min_pitch_shift=-2, max_pitch_shift=2, pitch_shift_p=0.5,
                 noise_root=None, noise_volume=0.05, noise_p=0.5,
                 time_masking_p=0.1, time_masking_max_length=None,
                 frequency_masking_p=0.1, frequency_masking_max_length=None,
                 enable_pitch_shifting=False, enable_time_stretching=False,
                 enable_time_masking=False, enable_frequency_masking=False,
                 audio_transform_type='mel_spectrogram'):
        super(AudioDescriptorTrainingTransforms, self).__init__(sample_rate=sample_rate,
                                                                waveform_size=waveform_size,
                                                                n_features=n_features,
                                                                n_fft=n_fft,
                                                                audio_transform_type=audio_transform_type)
        self._noise_volume = noise_volume
        self._noise_p = noise_p

        self._enable_pitch_shifting = enable_pitch_shifting
        self._enable_time_stretching = enable_time_stretching
        self._min_time_stretch = min_time_stretch if self._enable_time_stretching else 1.0
        self._time_stretch = RandomTimeStretch(min_time_stretch, max_time_stretch, time_stretching_p)
        self._pitch_shift = RandomPitchShift(sample_rate, min_pitch_shift, max_pitch_shift, pitch_shift_p)

        self._enable_time_masking = enable_time_masking
        self._time_masking_p = time_masking_p
        time_mask_param = waveform_size // 10 if time_masking_max_length is None else time_masking_max_length
        self._time_masking = transforms.TimeMasking(time_mask_param)

        self._enable_frequency_masking = enable_frequency_masking
        self._frequency_masking_p = frequency_masking_p
        freq_mask_param = n_features // 10 if frequency_masking_max_length is None else frequency_masking_max_length
        self._frequency_masking = transforms.FrequencyMasking(freq_mask_param)

        if noise_root is not None:
            self._noises = self._load_noises(noise_root)
        else:
            self._noises = []

    def _load_noises(self, noise_root):
        noise_paths = [o for o in os.listdir(noise_root) if o.endswith('.wav')]

        noises = []
        for noise_path in noise_paths:
            waveform, sample_rate = torchaudio.load(os.path.join(noise_root, noise_path))
            waveform = resample(waveform, sample_rate, self._sample_rate)

            if self._waveform_size < waveform.size()[1]:
                noises.append(waveform)

        return noises

    def __call__(self, waveform, target, metadata):
        waveform = to_mono(waveform)
        waveform = resample(waveform, metadata['original_sample_rate'], self._sample_rate)

        waveform = resize_waveform_random(waveform, int(self._waveform_size / self._min_time_stretch))
        if self._enable_time_stretching:
            waveform = self._time_stretch(waveform)
        if self._enable_pitch_shifting:
            waveform = self._pitch_shift(waveform)

        waveform = resize_waveform(waveform, self._waveform_size)
        waveform = normalize(waveform)

        if random.random() < self._noise_p:
            waveform = self._add_noise(waveform)
            waveform = normalize(waveform)

        spectrogram = self._audio_transform(waveform)

        if self._enable_time_masking and random.random() < self._time_masking_p:
            spectrogram = self._time_masking(spectrogram)

        if self._enable_frequency_masking and random.random() < self._frequency_masking_p:
            spectrogram = self._frequency_masking(spectrogram)

        return spectrogram, target, metadata

    def _add_noise(self, waveform):
        if len(self._noises) == 0:
            return waveform + self._noise_volume * torch.randn_like(waveform)
        else:
            noise = random.choice(self._noises)
            i = random.randrange(noise.size()[1] - self._waveform_size)
            return noise[:, i:(i + self._waveform_size)] * self._noise_volume + waveform


class AudioDescriptorValidationTransforms(_AudioDescriptorTransforms):
    def __call__(self, waveform, target, metadata):
        waveform = to_mono(waveform)
        waveform = resample(waveform, metadata['original_sample_rate'], self._sample_rate)
        waveform = resize_waveform(waveform, self._waveform_size)
        waveform = normalize(waveform)

        spectrogram = self._audio_transform(waveform)
        return spectrogram, target, metadata


class AudioDescriptorTestTransforms(_AudioDescriptorTransforms):
    def __call__(self, waveform, target, metadata):
        waveform = to_mono(waveform)
        waveform = resample(waveform, metadata['original_sample_rate'], self._sample_rate)
        waveform = normalize(waveform)

        spectrogram = self._audio_transform(waveform)
        return spectrogram, target, metadata
