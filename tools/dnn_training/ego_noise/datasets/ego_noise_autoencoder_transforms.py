import torch

from common.datasets.audio_transform_utils import to_mono, resample


class EgoNoiseAutoencoderTransforms:
    def __init__(self, sample_rate=44100, n_fft=2048):
        self._sample_rate = sample_rate
        self._n_fft = n_fft
        self._hop_length = n_fft // 2
        self._win_length = n_fft
        self._window = torch.sqrt(torch.hann_window(n_fft))

    def __call__(self, waveform, metadata):
        waveform = to_mono(waveform)
        waveform = resample(waveform, metadata['original_sample_rate'], self._sample_rate)

        X = torch.stft(waveform, n_fft=self._n_fft, hop_length=self._hop_length, win_length=self._win_length,
                   window=self._window, center=False)
        X_magnitude = torch.sqrt(X[:, :, :, 0]**2 + X[:, :, :, 1]**2)
        X_magnitude = X_magnitude[0].permute(1, 0)

        return X_magnitude, metadata
