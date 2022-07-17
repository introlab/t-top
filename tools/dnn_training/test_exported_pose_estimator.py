import argparse

import torch

from common.test import load_exported_model

from pose_estimation.metrics import CocoPoseEvaluation
from pose_estimation.datasets import PoseEstimationCoco
from pose_estimation.trainers.pose_estimator_trainer import create_validation_image_transform


def main():
    parser = argparse.ArgumentParser(description='Test exported pose estimator')

    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--torch_script_path', type=str, help='Choose the TorchScript path')
    parser.add_argument('--trt_path', type=str, help='Choose the TensorRT path')
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)

    args = parser.parse_args()

    model, device = load_exported_model(args.torch_script_path, args.trt_path)
    dataset = PoseEstimationCoco(args.dataset_root,
                                 train=False,
                                 data_augmentation=False,
                                 image_transforms=create_validation_image_transform())
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    coco_pose_evaluation = CocoPoseEvaluation(model, device, dataset_loader, args.output_path)
    coco_pose_evaluation.evaluate()


if __name__ == '__main__':
    main()
