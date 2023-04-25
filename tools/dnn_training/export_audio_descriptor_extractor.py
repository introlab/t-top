"""
You need to install : https://github.com/NVIDIA-AI-IOT/torch2trt#option-2---with-plugins-experimental
"""

import argparse

from common.file_presence_checker import terminate_if_already_exported


def main():
    parser = argparse.ArgumentParser(description='Export audio descriptor extractor')
    parser.add_argument('--backbone_type', choices=['mnasnet0.5', 'mnasnet1.0',
                                                    'resnet18', 'resnet34', 'resnet50',
                                                    'open_face_inception', 'thin_resnet_34',
                                                    'ecapa_tdnn_512', 'ecapa_tdnn_1024',
                                                    'small_ecapa_tdnn_128', 'small_ecapa_tdnn_256',
                                                    'small_ecapa_tdnn_512'],
                        help='Choose the backbone type', required=True)
    parser.add_argument('--embedding_size', type=int, help='Set the embedding size', required=True)
    parser.add_argument('--pooling_layer', choices=['avg', 'vlad', 'sap'], help='Set the pooling layer', required=True)
    parser.add_argument('--waveform_size', type=int, help='Set the waveform size', required=True)
    parser.add_argument('--n_features', type=int, help='Set n_features', required=True)
    parser.add_argument('--n_fft', type=int, help='Set n_fft', required=True)

    parser.add_argument('--dataset_class_count', type=int,
                        help='Choose the dataset class count when criterion_type is "cross_entropy_loss" or '
                             '"am_softmax_loss"',
                        default=None)
    parser.add_argument('--normalized_linear', action='store_true',
                        help='Use "NormalizedLinear" instead of "nn.Linear"')
    parser.add_argument('--conv_bias', action='store_true', help='Set bias=True to Conv2d (open_face_inception only)')

    parser.add_argument('--output_dir', type=str, help='Choose the output directory', required=True)
    parser.add_argument('--torch_script_filename', type=str, help='Choose the TorchScript filename', required=True)
    parser.add_argument('--trt_filename', type=str, help='Choose the TensorRT filename', required=True)
    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', required=True)

    parser.add_argument('--trt_fp16', action='store_true', help='Choose the model checkpoint file')

    parser.add_argument('--force_export_if_exists', action='store_true')

    args = parser.parse_args()

    terminate_if_already_exported(args.output_dir, args.torch_script_filename, args.trt_filename, args.force_export_if_exists)

    import torch

    from common.model_exporter import export_model

    from train_audio_descriptor_extractor import create_model

    image_size = (args.n_features, args.waveform_size // (args.n_fft // 2) + 1)
    model = create_model(args.backbone_type, args.n_features, args.embedding_size, args.dataset_class_count,
                         args.normalized_linear, args.pooling_layer, conv_bias=args.conv_bias)
    x = torch.ones((1, 1, image_size[0], image_size[1]))
    keys_to_remove = ['_classifier._weight'] if args.dataset_class_count is None else []
    export_model(model, args.model_checkpoint, x, args.output_dir, args.torch_script_filename, args.trt_filename,
                 trt_fp16=args.trt_fp16, keys_to_remove=keys_to_remove)


if __name__ == '__main__':
    main()
