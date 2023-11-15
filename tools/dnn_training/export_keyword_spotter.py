"""
You need to install : https://github.com/NVIDIA-AI-IOT/torch2trt#option-2---with-plugins-experimental
"""

import argparse

from common.file_presence_checker import terminate_if_already_exported

from keyword_spotting.keyword_spotter import KeywordSpotter


def main():
    parser = argparse.ArgumentParser(description='Export keyword spotter')
    parser.add_argument('--dataset_type', choices=['google_speech_commands', 'ttop_keyword'],
                        help='Choose the database type', required=True)
    parser.add_argument('--mfcc_feature_count', type=int, help='Choose the MFCC feature count', required=True)

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

    model = create_model(args.dataset_type)

    x = torch.ones((1, 1, args.mfcc_feature_count, 51))
    export_model(model, args.model_checkpoint, x, args.output_dir, args.torch_script_filename, args.trt_filename,
                 trt_fp16=args.trt_fp16)


def create_model(dataset_type):
    if dataset_type == 'google_speech_commands':
        return KeywordSpotter(class_count=36, use_softmax=False)
    elif dataset_type == 'ttop_keyword':
        return KeywordSpotter(class_count=2, use_softmax=False)
    else:
        raise ValueError('Invalid database type')


if __name__ == '__main__':
    main()
