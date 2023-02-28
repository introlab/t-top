import os

import torch

from dnn_utils.dnn_model import PACKAGE_PATH, DnnModel
from dnn_utils.audio_transforms import MFCC, GPU_SUPPORTED, normalize, standardize_every_frame


DURATION = 64000
SAMPLING_FREQUENCY = 16000
N_MFCC = 80
N_FFT = 320


class VoiceDescriptorExtractorBest(DnnModel):
    def __init__(self, inference_type=None):
        torch_script_model_path = os.path.join(PACKAGE_PATH, 'models', 'voice_descriptor_extractor_best.ts.pth')
        tensor_rt_model_path = os.path.join(PACKAGE_PATH, 'models', 'voice_descriptor_extractor_best.trt.pth')
        sample_input = torch.ones((1, 1, N_MFCC, int(2 * DURATION / N_FFT) + 1))

        super(VoiceDescriptorExtractorBest, self).__init__(torch_script_model_path, tensor_rt_model_path, sample_input,
                                                       inference_type=inference_type)
        self._transform = MFCC(SAMPLING_FREQUENCY, N_FFT, N_MFCC)

    def get_supported_sampling_frequency(self):
        return SAMPLING_FREQUENCY

    def get_supported_duration(self):
        return DURATION

    def __call__(self, x):
        with torch.no_grad():
            if GPU_SUPPORTED:
                x = x.to(self._device)

            x = normalize(x)
            spectrogram = self._transform(x).unsqueeze(0)
            spectrogram = standardize_every_frame(spectrogram)
            return super(VoiceDescriptorExtractor, self).__call__(spectrogram.unsqueeze(0))[0].cpu()
