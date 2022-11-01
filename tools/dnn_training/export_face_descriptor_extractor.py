"""
You need to install : https://github.com/NVIDIA-AI-IOT/torch2trt#option-2---with-plugins-experimental
"""

import argparse

import torch

from common.model_exporter import export_model

from face_recognition.datasets import IMAGE_SIZE
from face_recognition.face_descriptor_extractor import FaceDescriptorExtractor


def main():
    parser = argparse.ArgumentParser(description='Export face descriptor')
    parser.add_argument('--embedding_size', type=int, help='Set the embedding size', required=True)

    parser.add_argument('--output_dir', type=str, help='Choose the output directory', required=True)
    parser.add_argument('--torch_script_filename', type=str, help='Choose the TorchScript filename', required=True)
    parser.add_argument('--trt_filename', type=str, help='Choose the TensorRT filename', required=True)
    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', required=True)

    parser.add_argument('--trt_fp16', action='store_true', help='Choose the model checkpoint file')

    parser.add_argument('--force_export_if_exists', action='store_true')

    args = parser.parse_args()

    model = FaceDescriptorExtractor(embedding_size=args.embedding_size)
    x = torch.ones((1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))
    export_model(model, args.model_checkpoint, x, args.output_dir, args.torch_script_filename, args.trt_filename,
                 trt_fp16=args.trt_fp16, keys_to_remove=['_classifier._weight'], force_export_if_exists=args.force_export_if_exists)


if __name__ == '__main__':
    main()
