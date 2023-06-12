import numpy as np

import torch
import torchaudio.transforms as transforms


def normalize(waveform):
    s = max(waveform.min().abs(), waveform.max().abs()) + 1e-6
    return waveform / s


def standardize_every_frame(spectrogram, eps=1e-8):
    return (spectrogram - spectrogram.mean(dim=1, keepdim=True)) / (spectrogram.std(dim=1, keepdim=True) + eps)


class MelSpectrogram:
    def __init__(self, sample_rate, n_fft, n_mels) -> None:
        self._transform = transforms.MelSpectrogram(sample_rate=sample_rate,
                                                    n_fft=n_fft,
                                                    n_mels=n_mels)

    def __call__(self, x):
        return self._transform(x)


class MFCC:
    def __init__(self, sample_rate, n_fft, n_mfcc) -> None:
        melkwargs = {
            'n_fft': n_fft
        }
        self._transform = transforms.MFCC(sample_rate=sample_rate,
                                          n_mfcc=n_mfcc,
                                          melkwargs=melkwargs)

    def __call__(self, x):
        return self._transform(x)
