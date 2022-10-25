import numpy as np

import torch
import torchaudio.functional as F

try:
    import cupy as cp
    GPU_SUPPORTED = True
except ImportError:
    GPU_SUPPORTED = False


def normalize(waveform):
    s = max(waveform.min().abs(), waveform.max().abs()) + 1e-6
    return waveform / s


def standardize_every_frame(spectrogram, eps=1e-8):
    return (spectrogram - spectrogram.mean(dim=1, keepdim=True)) / (spectrogram.std(dim=1, keepdim=True) + eps)


class MelSpectrogram:
    def __init__(self, sample_rate, n_fft, n_mels) -> None:
        self._cpu_tranform = _CpuMelSpectrogram(sample_rate, n_fft, n_mels)
        if GPU_SUPPORTED:
            self._gpu_tranform = _GpuMelSpectrogram(sample_rate, n_fft, n_mels)

    def __call__(self, x):
        if not x.is_cuda:
            return torch.from_numpy(self._cpu_tranform(x)).float()
        elif GPU_SUPPORTED:
            return torch.as_tensor(self._gpu_tranform(x), device='cuda')
        else:
            raise ValueError('GPU is not supported (cupy must be installed)')


class _CpuMelSpectrogram:
    def __init__(self, sample_rate, n_fft, n_mels):
        if n_fft % 2 > 0:
            raise ValueError('Not supported n_fft: n_fft must be event.')

        self._n_fft = n_fft

        self._fb_matrix = F.melscale_fbanks(n_fft // 2 + 1, 0, sample_rate // 2, n_mels, sample_rate).float().numpy()
        self._window = np.hanning(n_fft)

    def __call__(self, x):
        x = x.numpy()
        x = np.pad(x, self._n_fft // 2, mode='reflect')
        frames = self._create_frames(x)
        power_spectrogram = np.abs(np.fft.rfft(frames, self._n_fft)) ** 2
        mel_spectrogram = np.dot(power_spectrogram, self._fb_matrix).T
        return mel_spectrogram

    def _create_frames(self, x):
        complete_frame_count = 2 * x.shape[0] // self._n_fft - 1
        hop_lengh = self._n_fft // 2
        if x.shape[0] % self._n_fft > 0:
            raise ValueError('Not supported x: x must be a multiple of n_fft.')

        indices = np.tile(np.arange(0, self._n_fft, dtype=np.int32), (complete_frame_count, 1)) + \
                  np.tile(np.arange(0, complete_frame_count * hop_lengh, hop_lengh, dtype=np.int32), (self._n_fft, 1)).T
        return x[indices] * np.tile(self._window, (complete_frame_count, 1))


if GPU_SUPPORTED:
    class _GpuMelSpectrogram:
        def __init__(self, sample_rate, n_fft, n_mels) -> None:
            if n_fft % 2 > 0:
                raise ValueError('Not supported n_fft: n_fft must be event.')

            self._n_fft = n_fft

            fb_matrix = F.melscale_fbanks(n_fft // 2 + 1, 0, sample_rate // 2, n_mels, sample_rate).float()
            self._fb_matrix = cp.asarray(fb_matrix)
            self._window = cp.hanning(n_fft).astype(cp.float32)

        def __call__(self, x):
            x = cp.asarray(x)
            x = cp.pad(x, self._n_fft // 2, mode='reflect')
            frames = self._create_frames(x)
            power_spectrogram = cp.abs(cp.fft.rfft(frames, self._n_fft)) ** 2
            mel_spectrogram = cp.dot(power_spectrogram, self._fb_matrix).T
            return mel_spectrogram

        def _create_frames(self, x):
            complete_frame_count = 2 * x.shape[0] // self._n_fft - 1
            hop_lengh = self._n_fft // 2
            if x.shape[0] % self._n_fft > 0:
                raise ValueError('Not supported x: x must be a multiple of n_fft.')

            indices = cp.tile(cp.arange(0, self._n_fft, dtype=cp.int32), (complete_frame_count, 1)) + \
                    cp.tile(cp.arange(0, complete_frame_count * hop_lengh, hop_lengh, dtype=cp.int32), (self._n_fft, 1)).T
            return x[indices] * cp.tile(self._window, (complete_frame_count, 1))


class MFCC:
    def __init__(self, sample_rate, n_fft, n_mfcc) -> None:
        self._cpu_tranform = _CpuMFCC(sample_rate, n_fft, n_mfcc)
        if GPU_SUPPORTED:
            self._gpu_tranform = _GpuMFCC(sample_rate, n_fft, n_mfcc)

    def __call__(self, x):
        if not x.is_cuda:
            return torch.from_numpy(self._cpu_tranform(x)).float()
        elif GPU_SUPPORTED:
            return torch.as_tensor(self._gpu_tranform(x), device='cuda')
        else:
            raise ValueError('GPU is not supported (cupy must be installed)')


class _CpuMFCC:
    def __init__(self, sample_rate, n_fft, n_mfcc) -> None:
        n_mels = 128
        self._mel_spectrogram = _CpuMelSpectrogram(sample_rate, n_fft, n_mels)

        self._dct_mat = F.create_dct(n_mfcc, n_mels, 'ortho').float().numpy().T

    def __call__(self, x):
        mel_spectrogram = self._mel_spectrogram(x)
        mel_spectrogram_db = self._power_to_db(mel_spectrogram)
        return np.dot(self._dct_mat, mel_spectrogram_db)

    def _power_to_db(self, x):
        top_db = 80.0
        x_db = 10.0 * np.log10(np.clip(x, a_min=1e-10, a_max=None))
        x_db = np.maximum(x_db, (x_db.max() - top_db))
        return x_db


if GPU_SUPPORTED:
    class _GpuMFCC:
        def __init__(self, sample_rate, n_fft, n_mfcc) -> None:
            n_mels = 128
            self._mel_spectrogram = _GpuMelSpectrogram(sample_rate, n_fft, n_mels)

            dct_mat = F.create_dct(n_mfcc, n_mels, 'ortho').float().numpy().T
            self._dct_mat = cp.asarray(dct_mat)

        def __call__(self, x):
            mel_spectrogram = self._mel_spectrogram(x)
            mel_spectrogram_db = self._power_to_db(mel_spectrogram)
            return cp.dot(self._dct_mat, mel_spectrogram_db)

        def _power_to_db(self, x):
            top_db = 80.0
            x_db = 10.0 * cp.log10(cp.clip(x, a_min=1e-10))
            x_db = cp.maximum(x_db, (x_db.max() - top_db))
            return x_db
