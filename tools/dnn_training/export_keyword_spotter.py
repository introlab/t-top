import argparse

import torch

from common.model_exporter import export_model

from train_keyword_spotter import create_model


def main():
    parser = argparse.ArgumentParser(description='Export keyword spotter')
    parser.add_argument('--database_type', choices=['google_speech_commands', 'ttop_keyword'],
                        help='Choose the database type', required=True)
    parser.add_argument('--mfcc_feature_count', type=int, help='Choose the MFCC feature count', required=True)

    parser.add_argument('--output_dir', type=str, help='Choose the output directory', required=True)
    parser.add_argument('--torch_script_filename', type=str, help='Choose the TorchScript filename', required=True)
    parser.add_argument('--trt_filename', type=str, help='Choose the TensorRT filename', required=True)
    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', required=True)

    parser.add_argument('--trt_fp16', action='store_true', help='Choose the model checkpoint file')

    args = parser.parse_args()

    model = create_model(args.database_type)

    x = torch.ones((1, 1, args.mfcc_feature_count, 51))
    export_model(model, args.model_checkpoint, x, args.output_dir, args.torch_script_filename, args.trt_filename,
                 trt_fp16=args.trt_fp16)


if __name__ == '__main__':
    main()
