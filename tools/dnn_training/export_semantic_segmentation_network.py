"""
You need to install : https://github.com/NVIDIA-AI-IOT/torch2trt#option-2---with-plugins-experimental
"""

import argparse

from common.file_presence_checker import terminate_if_already_exported


def main():
    parser = argparse.ArgumentParser(description='Export keyword spotter')
    parser.add_argument('--dataset_type', choices=['coco', 'kitchen_open_images', 'person_other_open_images'],
                        help='Choose the database type', required=True)
    parser.add_argument('--backbone_type', choices=['stdc1', 'stdc2'], help='Choose the backbone type', required=True)
    parser.add_argument('--channel_scale', type=int, help='Choose the decoder channel count scale factor',
                        required=True)

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

    from semantic_segmentation.trainers.semantic_segmentation_trainer import IMAGE_SIZE

    from train_semantic_segmentation_network import create_model

    model = create_model(args.backbone_type, args.channel_scale, args.dataset_type)

    x = torch.ones((1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))
    export_model(model, args.model_checkpoint, x, args.output_dir, args.torch_script_filename, args.trt_filename,
                 trt_fp16=args.trt_fp16)


if __name__ == '__main__':
    main()
