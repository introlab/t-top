import os

import numpy as np
import scipy.io.wavfile
import scipy.signal


class FolderExtractor:
    def __init__(self, noise_threshold, mean_avg_sample_count, output_fs,
                 waveform_padding, output_sample_count, output_max_offset):
        self._noise_threshold = noise_threshold
        self._mean_avg_sample_count = mean_avg_sample_count
        self._output_fs = output_fs
        self._waveform_padding = waveform_padding
        self._output_sample_count = output_sample_count
        self._output_max_offset = output_max_offset

    def extract(self, input, output):
        filenames = [o for o in os.listdir(input)]
        os.makedirs(output, exist_ok=True)

        for i, filename in enumerate(filenames):
            print('{}/{}'.format(i + 1, len(filenames)), filename)
            self._extract(input, output, filename)

    def _extract(self, input, output, filename):
        pass

    def _load_waveform(self, input, filename):
        rate, waveform = scipy.io.wavfile.read(os.path.join(input, filename))
        if len(waveform.shape) > 1:
            waveform = np.sum(waveform, axis=1)

        waveform = waveform / np.abs(waveform).max()
        waveform = scipy.signal.resample(waveform, int(waveform.shape[0] / rate * self._output_fs))
        waveform = np.concatenate([np.zeros(self._waveform_padding), waveform, np.zeros(self._waveform_padding)])
        return waveform

    def _get_waveform_avg_envelope(self, waveform):
        waveform_envelope = np.abs(scipy.signal.hilbert(waveform))
        waveform_envelope_padded = np.concatenate([waveform_envelope, np.zeros(self._mean_avg_sample_count)])
        waveform_avg_envelope = scipy.signal.lfilter(np.ones(self._mean_avg_sample_count) / self._mean_avg_sample_count,
                                                     1, waveform_envelope_padded)
        waveform_avg_envelope = waveform_avg_envelope[self._mean_avg_sample_count // 2:]
        return waveform_avg_envelope

    def _get_rectangles(self, waveform_avg_envelope):
        is_voice = waveform_avg_envelope > self._noise_threshold

        rectangles = []

        start_index = 0
        current_value = is_voice[0]

        for i in range(1, is_voice.shape[0]):
            if current_value != is_voice[i]:
                rectangles.append({'start_index': start_index, 'end_index': i, 'value': current_value})
                start_index = i
                current_value = is_voice[i]

        return rectangles
