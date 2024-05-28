import numpy as np

import torch


def normalize(waveform):
    s = max(waveform.min().abs(), waveform.max().abs()) + 1e-6
    return waveform / s


def standardize_every_frame(spectrogram, eps=1e-8):
    return (spectrogram - spectrogram.mean(dim=1, keepdim=True)) / (spectrogram.std(dim=1, keepdim=True) + eps)
