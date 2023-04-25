"""
You need to install : https://github.com/NVIDIA-AI-IOT/torch2trt#option-2---with-plugins-experimental
"""

import argparse

from common.file_presence_checker import terminate_if_already_exported

BACKBONE_TYPES = ['open_face', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                  'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']


def main():
    parser = argparse.ArgumentParser(description='Export face descriptor')
    parser.add_argument('--backbone_type', choices=BACKBONE_TYPES, help='Choose the backbone type', required=True)
    parser.add_argument('--embedding_size', type=int, help='Set the embedding size', required=True)

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

    from face_recognition.datasets import IMAGE_SIZE
    from train_face_descriptor_extractor import create_model

    model = create_model(args.backbone_type, args.embedding_size)
    x = torch.ones((1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))
    export_model(model, args.model_checkpoint, x, args.output_dir, args.torch_script_filename, args.trt_filename,
                 trt_fp16=args.trt_fp16, keys_to_remove=['_classifier._weight'])


if __name__ == '__main__':
    main()
