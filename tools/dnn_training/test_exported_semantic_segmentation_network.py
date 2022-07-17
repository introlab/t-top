import argparse

import torch

from common.test import load_exported_model

from semantic_segmentation.datasets import SemanticSegmentationValidationTransforms
from semantic_segmentation.trainers.semantic_segmentation_trainer import IMAGE_SIZE, create_dataset, evaluate

from train_semantic_segmentation_network import get_class_count_from_dataset_type


def main():
    parser = argparse.ArgumentParser(description='Test exported keyword spotter')

    parser.add_argument('--dataset_type', choices=['coco', 'open_images'],
                        help='Choose the database type', required=True)
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--backbone_type', choices=['stdc1', 'stdc2'], help='Choose the backbone type', required=True)
    parser.add_argument('--channel_scale', type=int, help='Choose the decoder channel count scale factor',
                        required=True)

    parser.add_argument('--torch_script_path', type=str, help='Choose the TorchScript path')
    parser.add_argument('--trt_path', type=str, help='Choose the TensorRT path')

    args = parser.parse_args()

    model, device = load_exported_model(args.torch_script_path, args.trt_path)
    class_count = get_class_count_from_dataset_type(args.dataset_type)
    transforms = SemanticSegmentationValidationTransforms(IMAGE_SIZE, class_count)
    dataset = create_dataset(args.database_type, args.dataset_root, 'testing', transforms)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    evaluate(model, device, dataset_loader, class_count)


if __name__ == '__main__':
    main()
