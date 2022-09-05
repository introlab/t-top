import argparse

from common.test import load_exported_model

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
    parser.add_argument('--audio_transform_type', choices=['mfcc', 'mel_spectrogram', 'spectrogram'],
                        help='Choose the audio transform type', required=True)

    args = parser.parse_args()

    model, device = load_exported_model(args.torch_script_path, args.trt_path)
    transforms = AudioDescriptorValidationTransforms(waveform_size=args.waveform_size,
                                                     n_features=args.n_features,
                                                     n_fft=args.n_fft,
                                                     audio_transform_type=args.audio_transform_type)
    evaluation = AudioDescriptorEvaluation(model, device, transforms, args.dataset_root, args.output_path)
    evaluation.evaluate()


if __name__ == '__main__':
    main()
