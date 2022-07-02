import argparse
import os

import torch

from backbone.stdc import Stdc1, Stdc2
from backbone.trainers import BackboneTrainer
from backbone.datasets.classification_open_images import CLASS_COUNT


# Train a model like : https://github.com/microsoft/human-pose-estimation.pytorch
def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('--model_type', choices=['stdc1', 'stdc2'], help='Choose the model type', required=True)

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)
    parser.add_argument('--criterion_type', choices=['cross_entropy_loss', 'ohem_cross_entropy_loss'],
                        help='Choose the criterion type', required=True)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)
    parser.add_argument('--optimizer_checkpoint', type=str, help='Choose the optimizer checkpoint file', default=None)
    parser.add_argument('--scheduler_checkpoint', type=str, help='Choose the scheduler checkpoint file', default=None)

    args = parser.parse_args()

    model = create_model(args.model_type)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    trainer = BackboneTrainer(device, model,
                              epoch_count=args.epoch_count,
                              learning_rate=args.learning_rate,
                              dataset_root=args.dataset_root,
                              output_path=os.path.join(args.output_path, args.model_type + '_' + args.criterion_type),
                              batch_size=args.batch_size,
                              criterion_type=args.criterion_type,
                              model_checkpoint=args.model_checkpoint,
                              optimizer_checkpoint=args.optimizer_checkpoint,
                              scheduler_checkpoint=args.scheduler_checkpoint)
    trainer.train()


def create_model(model_type):
    if model_type == 'stdc1':
        return Stdc1(class_count=CLASS_COUNT)
    elif model_type == 'stdc2':
        return Stdc1(class_count=CLASS_COUNT)
    else:
        raise ValueError('Invalid backbone type')


if __name__ == '__main__':
    main()
