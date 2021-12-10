import random

import torch
import torch.nn as nn
import torchaudio.transforms as transforms

from librosa.effects import pitch_shift, time_stretch


def to_mono(waveform):
    return torch.mean(waveform, dim=0, keepdim=True)


def normalize(waveform):
    s = max(waveform.min().abs(), waveform.max().abs()) + 1e-6
    return waveform / s


def resize_waveform(waveform, output_size):
    if waveform.size()[1] < output_size:
        padding_size = output_size - waveform.size()[1]
        left_padding_size = padding_size // 2
        right_padding_size = padding_size - left_padding_size

        return torch.cat([torch.zeros(1, left_padding_size), waveform, torch.zeros(1, right_padding_size)], dim=1)

    elif waveform.size()[1] > output_size:
        i = (waveform.size()[1] - output_size) // 2
        return waveform[:, i:(i + output_size)]

    else:
        return waveform


def resize_waveform_random(waveform, output_size):
    if waveform.size()[1] < output_size:
        padding_size = output_size - waveform.size()[1]
        left_padding_size = random.randrange(padding_size)
        right_padding_size = padding_size - left_padding_size

        return torch.cat([torch.zeros(1, left_padding_size), waveform, torch.zeros(1, right_padding_size)], dim=1)

    elif waveform.size()[1] > output_size:
        i = random.randrange(waveform.size()[1] - output_size)
        return waveform[:, i:(i + output_size)]

    else:
        return waveform


def resample(waveform, input_sample_rate, output_sample_rate):
    if input_sample_rate != output_sample_rate:
        return transforms.Resample(orig_freq=input_sample_rate, new_freq=output_sample_rate)(waveform)
    else:
        return waveform


class RandomPitchShift(nn.Module):
    def __init__(self, sample_rate, min_steps, max_steps, p):
        super(RandomPitchShift, self).__init__()
        self._sample_rate = sample_rate
        self._min_steps = min_steps
        self._max_steps = max_steps
        self._p = p

    def forward(self, x):
        if x.size(0) != 1:
            raise ValueError('x must have one channel')

        if random.random() < self._p:
            n_steps = random.randint(self._min_steps, self._max_steps)
            return torch.from_numpy(pitch_shift(x[0].numpy(), self._sample_rate, n_steps)).unsqueeze(0)
        else:
            return x


class RandomTimeStretch(nn.Module):
    def __init__(self, min_rate, max_rate, p):
        super(RandomTimeStretch, self).__init__()
        self._min_rate = min_rate
        self._max_rate = max_rate
        self._p = p

    def forward(self, x):
        if x.size(0) != 1:
            raise ValueError('x must have one channel')

        if random.random() < self._p:
            rate = random.uniform(self._min_rate, self._max_rate)
            return torch.from_numpy(time_stretch(x[0].numpy(), rate)).unsqueeze(0)
        else:
            return x


class RandomTimeRoll(nn.Module):
    def __init__(self, p):
        super(RandomTimeRoll, self).__init__()
        self._p = p

    def forward(self, x):
        if random.random() < self._p:
            shifts = random.randrange(x.size(1))
            return torch.roll(x, shifts, dims=1)
        else:
            return x


class RandomTimeSwap(nn.Module):
    def __init__(self, min_size_ratio, max_size_ratio, p):
        super(RandomTimeSwap, self).__init__()
        self._min_size_ratio = min_size_ratio
        self._max_size_ratio = max_size_ratio
        self._p = p

    def forward(self, x):
        if random.random() < self._p:
            min_size = int(x.size(1) * self._min_size_ratio)
            max_size = int(x.size(1) * self._max_size_ratio)
            size = random.randrange(min_size, max_size)
            from_index = random.randrange(0, x.size(1) - size)
            to_index = random.randrange(0, x.size(1) - size)

            y = x.clone()
            y[:, from_index:from_index + size] = x[:, to_index:to_index + size]
            y[:, to_index:to_index + size] = x[:, from_index:from_index + size]
            return y
        else:
            return x
