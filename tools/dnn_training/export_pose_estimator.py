"""
You need to install : https://github.com/NVIDIA-AI-IOT/torch2trt#option-2---with-plugins-experimental
"""

import argparse

import torch

from common.model_exporter import export_model

from pose_estimation.trainers.pose_estimator_trainer import IMAGE_SIZE
from train_pose_estimator import create_model


def main():
    parser = argparse.ArgumentParser(description='Export pose estimator')
    parser.add_argument('--backbone_type', choices=['mnasnet0.5', 'mnasnet1.0', 'resnet18', 'resnet34', 'resnet50'],
                        help='Choose the backbone type', required=True)
    parser.add_argument('--upsampling_count', type=int, help='Set the upsamping layer count', required=True)

    parser.add_argument('--output_dir', type=str, help='Choose the output directory', required=True)
    parser.add_argument('--torch_script_filename', type=str, help='Choose the TorchScript filename', required=True)
    parser.add_argument('--trt_filename', type=str, help='Choose the TensorRT filename', required=True)
    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', required=True)

    parser.add_argument('--trt_fp16', action='store_true', help='Choose the model checkpoint file')

    args = parser.parse_args()

    model = create_model(args.backbone_type, args.upsampling_count)
    x = torch.ones((1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))
    export_model(model, args.model_checkpoint, x, args.output_dir, args.torch_script_filename, args.trt_filename,
                 trt_fp16=args.trt_fp16)


if __name__ == '__main__':
    main()
