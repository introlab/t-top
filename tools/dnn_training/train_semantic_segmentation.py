import argparse
import os

import torch

from common.modules import load_checkpoint

from semantic_segmentation.pp_lite_seg import PpLiteSeg
from semantic_segmentation.trainers import SemanticSegmentationTrainer
from semantic_segmentation.datasets.semantic_segmentation_open_images import CLASS_COUNT

from train_backbone import create_model as create_backbone


# Train a model like : https://github.com/microsoft/human-pose-estimation.pytorch
def main():
    parser = argparse.ArgumentParser(description='Train semantic segmentation')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('--backbone_type', choices=['stdc1', 'stdc2'], help='Choose the backbone type', required=True)
    parser.add_argument('--channel_scale', type=int, help='Choose the decoder channel count scale factor',
                        required=True)

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)

    parser.add_argument('--backbone_checkpoint', type=str, help='Choose the model checkpoint file', default=None)
    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)
    parser.add_argument('--optimizer_checkpoint', type=str, help='Choose the optimizer checkpoint file', default=None)
    parser.add_argument('--scheduler_checkpoint', type=str, help='Choose the scheduler checkpoint file', default=None)

    args = parser.parse_args()

    model = create_model(args.backbone_type, args.backbone_checkpoint, args.channel_scale)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    output_path = os.path.join(args.output_path, args.backbone_type + '_s' + str(args.channel_scale))
    trainer = SemanticSegmentationTrainer(device, model,
                                          epoch_count=args.epoch_count,
                                          learning_rate=args.learning_rate,
                                          dataset_root=args.dataset_root,
                                          output_path=output_path,
                                          batch_size=args.batch_size,
                                          model_checkpoint=args.model_checkpoint,
                                          optimizer_checkpoint=args.optimizer_checkpoint,
                                          scheduler_checkpoint=args.scheduler_checkpoint)
    trainer.train()


def create_model(backbone_type, backbone_checkpoint, channel_scale):
    backbone = create_backbone(backbone_type)
    if backbone_checkpoint is not None:
        load_checkpoint(backbone, backbone_checkpoint)

    return PpLiteSeg(backbone, CLASS_COUNT, channel_scale)


if __name__ == '__main__':
    main()
