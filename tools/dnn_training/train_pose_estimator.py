import argparse
import os

import torch

from pose_estimation.backbones import Mnasnet0_5, Mnasnet1_0, Resnet18, Resnet34, Resnet50
from pose_estimation.pose_estimator import PoseEstimator
from pose_estimation.trainers import PoseEstimatorTrainer


# Train a model like : https://github.com/microsoft/human-pose-estimation.pytorch
def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('--backbone_type', choices=['mnasnet0.5', 'mnasnet1.0', 'resnet18', 'resnet34', 'resnet50'],
                        help='Choose the backbone type', required=True)
    parser.add_argument('--upsampling_count', type=int, help='Set the upsamping layer count', required=True)

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--batch_size_division', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)
    parser.add_argument('--optimizer_checkpoint', type=str, help='Choose the optimizer checkpoint file', default=None)
    parser.add_argument('--scheduler_checkpoint', type=str, help='Choose the scheduler checkpoint file', default=None)

    args = parser.parse_args()

    model = create_model(args.backbone_type, args.upsampling_count)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    trainer = PoseEstimatorTrainer(device, model,
                                   epoch_count=args.epoch_count,
                                   learning_rate=args.learning_rate,
                                   dataset_root=args.dataset_root,
                                   output_path=os.path.join(args.output_path, args.backbone_type),
                                   batch_size=args.batch_size,
                                   batch_size_division=args.batch_size_division,
                                   model_checkpoint=args.model_checkpoint,
                                   optimizer_checkpoint=args.optimizer_checkpoint,
                                   scheduler_checkpoint=args.scheduler_checkpoint)
    trainer.train()


def create_model(backbone_type, upsampling_count):
    pretrained = True
    keypoint_count = 17
    if backbone_type == 'mnasnet0.5':
        return PoseEstimator(Mnasnet0_5(pretrained=pretrained),
                             keypoint_count=keypoint_count,
                             upsampling_count=upsampling_count)
    elif backbone_type == 'mnasnet1.0':
        return PoseEstimator(Mnasnet1_0(pretrained=pretrained),
                             keypoint_count=keypoint_count,
                             upsampling_count=upsampling_count)
    elif backbone_type == 'resnet18':
        return PoseEstimator(Resnet18(pretrained=pretrained),
                             keypoint_count=keypoint_count,
                             upsampling_count=upsampling_count)
    elif backbone_type == 'resnet34':
        return PoseEstimator(Resnet34(pretrained=pretrained),
                             keypoint_count=keypoint_count,
                             upsampling_count=upsampling_count)
    elif backbone_type == 'resnet50':
        return PoseEstimator(Resnet50(pretrained=pretrained),
                             keypoint_count=keypoint_count,
                             upsampling_count=upsampling_count)
    else:
        raise ValueError('Invalid backbone type')


if __name__ == '__main__':
    main()
