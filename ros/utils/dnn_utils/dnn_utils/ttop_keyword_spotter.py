import os

import torch
import torch.nn.functional as F
import torchaudio.transforms as transforms

from dnn_utils.dnn_model import PACKAGE_PATH, DnnModel
from dnn_utils.audio_transforms import normalize


DURATION = 16000
SAMPLING_FREQUENCY = 16000
N_MFCC = 20
WINDOW_SIZE_MS = 40
N_FFT = int(SAMPLING_FREQUENCY / 1000 * WINDOW_SIZE_MS)


class TTopKeywordSpotter(DnnModel):
    def __init__(self, inference_type=None):
        torch_script_model_path = os.path.join(PACKAGE_PATH, 'models', 'ttop_keyword_spotter.ts.pth')
        tensor_rt_model_path = os.path.join(PACKAGE_PATH, 'models', 'ttop_keyword_spotter.trt.pth')
        sample_input = torch.ones((1, 1, N_MFCC, int(2 * DURATION / N_FFT) + 1))

        super(TTopKeywordSpotter, self).__init__(torch_script_model_path, tensor_rt_model_path, sample_input,
                                                 inference_type=inference_type)
        melkwargs = {
            'n_fft': N_FFT
        }
        self._transform = transforms.MFCC(sample_rate=SAMPLING_FREQUENCY,
                                          n_mfcc=N_MFCC,
                                          melkwargs=melkwargs).to(self._device)

    def get_supported_sampling_frequency(self):
        return SAMPLING_FREQUENCY

    def get_supported_duration(self):
        return DURATION

    def get_class_names(self):
        return ['T-Top', 'other']

    def __call__(self, x):
        with torch.no_grad():
            x = x.to(self._device)
            x = normalize(x)
            mfcc_features = self._transform(x).unsqueeze(0)
            scores = super(TTopKeywordSpotter, self).__call__(mfcc_features.unsqueeze(0))[0]
            probabilities = F.softmax(scores, dim=0)
            return probabilities.cpu()
