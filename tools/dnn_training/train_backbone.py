import argparse
import os

import torch

from common.program_arguments import save_arguments, print_arguments

from backbone.stdc import Stdc1, Stdc2
from backbone.trainers import BackboneTrainer
from backbone.datasets.classification_image_net import CLASS_COUNT as IMAGE_NET_CLASS_COUNT
from backbone.datasets.classification_open_images import CLASS_COUNT as OPEN_IMAGES_CLASS_COUNT


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('--dataset_type', choices=['image_net', 'open_images'],
                        help='Choose the dataset type', required=True)
    parser.add_argument('--model_type', choices=['stdc1', 'stdc2'], help='Choose the model type', required=True)

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--weight_decay', type=float, help='Choose the weight decay', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)
    parser.add_argument('--criterion_type',
                        choices=['cross_entropy_loss', 'ohem_cross_entropy_loss', 'softmax_focal_loss'],
                        help='Choose the criterion type', required=True)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)

    args = parser.parse_args()

    model = create_model(args.model_type, args.dataset_type)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    output_path = os.path.join(args.output_path, args.model_type + '_' + args.criterion_type + '_' +
                               args.dataset_type + '_lr' + str(args.learning_rate) + '_wd' + str(args.weight_decay))
    save_arguments(output_path, args)
    print_arguments(args)

    trainer = BackboneTrainer(device, model,
                              dataset_type=args.dataset_type,
                              epoch_count=args.epoch_count,
                              learning_rate=args.learning_rate,
                              weight_decay=args.weight_decay,
                              dataset_root=args.dataset_root,
                              output_path=output_path,
                              batch_size=args.batch_size,
                              criterion_type=args.criterion_type,
                              model_checkpoint=args.model_checkpoint)
    trainer.train()


def create_model(model_type, dataset_type):
    if dataset_type == 'image_net':
        class_count = IMAGE_NET_CLASS_COUNT
    elif dataset_type == 'open_images':
        class_count = OPEN_IMAGES_CLASS_COUNT
    else:
        raise ValueError('Invalid dataset type')

    if model_type == 'stdc1':
        return Stdc1(class_count=class_count, dropout=0.0)
    elif model_type == 'stdc2':
        return Stdc2(class_count=class_count, dropout=0.0)
    else:
        raise ValueError('Invalid backbone type')


if __name__ == '__main__':
    main()
