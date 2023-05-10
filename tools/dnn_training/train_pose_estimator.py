import argparse
import os

import torch

from common.program_arguments import save_arguments, print_arguments

from pose_estimation.pose_estimator import EfficientNetPoseEstimator
from pose_estimation.trainers import PoseEstimatorTrainer, PoseEstimatorDistillationTrainer

BACKBONE_TYPES = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4',
                  'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']


# Train a model similar to https://github.com/microsoft/human-pose-estimation.pytorch, but with residual connections.
def main():

    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('--backbone_type', choices=BACKBONE_TYPES, help='Choose the backbone type', required=True)

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--weight_decay', type=float, help='Choose the weight decay', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--batch_size_division', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)

    parser.add_argument('--heatmap_sigma', type=float, help='Choose sigma to create the heatmap', required=True)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)

    parser.add_argument('--teacher_backbone_type', choices=BACKBONE_TYPES, help='Choose the teacher backbone type',
                        default=None)
    parser.add_argument('--teacher_model_checkpoint', type=str, help='Choose the teacher model checkpoint file',
                        default=None)
    parser.add_argument('--distillation_loss_alpha', type=float, help='Choose the alpha for the distillation loss',
                        default=0.25)

    args = parser.parse_args()

    model = create_model(args.backbone_type)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    output_path = os.path.join(args.output_path, args.backbone_type + '_sig' + str(args.heatmap_sigma) +
                               '_lr' + str(args.learning_rate) + '_wd' + str(args.weight_decay) +
                               '_t' + str(args.teacher_backbone_type) + '_a' + str(args.distillation_loss_alpha))
    save_arguments(output_path, args)
    print_arguments(args)

    if args.teacher_backbone_type is not None and args.teacher_model_checkpoint is not None:
        teacher_model = create_model(args.teacher_backbone_type)
        trainer = PoseEstimatorDistillationTrainer(device, model, teacher_model,
                                                   epoch_count=args.epoch_count,
                                                   learning_rate=args.learning_rate,
                                                   weight_decay=args.weight_decay,
                                                   dataset_root=args.dataset_root,
                                                   output_path=output_path,
                                                   batch_size=args.batch_size,
                                                   batch_size_division=args.batch_size_division,
                                                   heatmap_sigma=args.heatmap_sigma,
                                                   student_model_checkpoint=args.model_checkpoint,
                                                   teacher_model_checkpoint=args.teacher_model_checkpoint,
                                                   loss_alpha=args.distillation_loss_alpha)
    elif args.teacher_backbone_type is not None or args.teacher_model_checkpoint is not None:
        raise ValueError('teacher_backbone_type and teacher_model_checkpoint must be set.')
    else:
        trainer = PoseEstimatorTrainer(device, model,
                                       epoch_count=args.epoch_count,
                                       learning_rate=args.learning_rate,
                                       weight_decay=args.weight_decay,
                                       dataset_root=args.dataset_root,
                                       output_path=output_path,
                                       batch_size=args.batch_size,
                                       batch_size_division=args.batch_size_division,
                                       heatmap_sigma=args.heatmap_sigma,
                                       model_checkpoint=args.model_checkpoint)
    trainer.train()



def create_model(backbone_type):
    return EfficientNetPoseEstimator(backbone_type, keypoint_count=17, pretrained_backbone=True)


if __name__ == '__main__':
    main()
