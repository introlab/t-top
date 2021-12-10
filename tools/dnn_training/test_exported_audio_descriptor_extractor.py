import argparse
import sys

import torch

try:
    from torch2trt import TRTModule

    torch2trt_found = True
except ImportError:
    torch2trt_found = False

from audio_descriptor.metrics import AudioDescriptorEvaluation
from audio_descriptor.datasets import AudioDescriptorValidationTransforms


def main():
    parser = argparse.ArgumentParser(description='Test exported audio descriptor extractor')

    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--torch_script_path', type=str, help='Choose the TorchScript path')
    parser.add_argument('--trt_path', type=str, help='Choose the TensorRT path')
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)

    parser.add_argument('--waveform_size', type=int, help='Set the waveform size', required=True)
    parser.add_argument('--n_features', type=int, help='Set n_features', required=True)
    parser.add_argument('--n_fft', type=int, help='Set n_fft', required=True)
    parser.add_argument('--audio_transform_type', choices=['mfcc', 'mel_spectrogram'],
                        help='Choose the audio transform type', required=True)

    args = parser.parse_args()

    if args.torch_script_path is not None:
        device = torch.device('cpu')
        model = torch.jit.load(args.torch_script_path)
        model = model.to(device)
    elif args.trt_path is not None:
        if not torch2trt_found:
            print('"torch2trt" is not supported.')
            sys.exit()
        else:
            device = torch.device('cuda')
            model = TRTModule()
            model.load_state_dict(torch.load(args.trt_path))
    else:
        print('"torch_script_path" or "trt_path" is required.')
        sys.exit()

    transforms = AudioDescriptorValidationTransforms(waveform_size=args.waveform_size,
                                                     n_features=args.n_features,
                                                     n_fft=args.n_fft,
                                                     audio_transform_type=args.audio_transform_type)
    evaluation = AudioDescriptorEvaluation(model, device, transforms, args.dataset_root, args.output_path)
    evaluation.evaluate()


if __name__ == '__main__':
    main()
