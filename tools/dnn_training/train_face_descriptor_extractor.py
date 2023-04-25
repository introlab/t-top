import argparse
import os

import torch

from common.program_arguments import save_arguments, print_arguments

from face_recognition.face_descriptor_extractor import FaceDescriptorExtractor, OpenFaceBackbone, EfficientNetBackbone
from face_recognition.trainers import FaceDescriptorExtractorTrainer

BACKBONE_TYPES = ['open_face', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                  'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--vvgface2_dataset_root', type=str, help='Choose the Vggface2 root path', required=True)
    parser.add_argument('--lfw_dataset_root', type=str, help='Choose the LFW root path', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('--backbone_type', choices=BACKBONE_TYPES, help='Choose the backbone type', required=True)
    parser.add_argument('--embedding_size', type=int, help='Set the embedding size', required=True)
    parser.add_argument('--margin', type=float, help='Set the margin', required=True)

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--weight_decay', type=float, help='Choose the weight decay', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)
    parser.add_argument('--criterion_type',
                        choices=['triplet_loss', 'cross_entropy_loss', 'am_softmax_loss', 'arc_face_loss'],
                        help='Choose the criterion type', required=True)
    parser.add_argument('--dataset_class_count', type=int,
                        help='Choose the dataset class count when criterion_type is "cross_entropy_loss", '
                             '"am_softmax_loss" or "arc_face_loss"',
                        default=None)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)

    args = parser.parse_args()

    if args.criterion_type == 'triplet_loss' and args.dataset_class_count is None:
        model = create_model(args.embedding_size)
    elif args.criterion_type == 'cross_entropy_loss' and args.dataset_class_count is not None:
        model = create_model(args.embedding_size, args.dataset_class_count)
    elif args.criterion_type == 'am_softmax_loss' and args.dataset_class_count is not None:
        model = create_model(args.embedding_size, args.dataset_class_count, normalized_linear=True)
    else:
        raise ValueError('--dataset_class_count must be used with "cross_entropy_loss" or "am_softmax_loss" types')
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    output_path = os.path.join(args.output_path, 'e' + str(args.embedding_size) +
                               '_' + args.criterion_type + '_lr' + str(args.learning_rate) +
                               '_wd' + str(args.weight_decay))
    save_arguments(output_path, args)
    print_arguments(args)

    trainer = FaceDescriptorExtractorTrainer(device, model,
                                             epoch_count=args.epoch_count,
                                             learning_rate=args.learning_rate,
                                             weight_decay=args.weight_decay,
                                             criterion_type=args.criterion_type,
                                             vvgface2_dataset_root=args.vvgface2_dataset_root,
                                             lfw_dataset_root=args.lfw_dataset_root,
                                             output_path=output_path,
                                             batch_size=args.batch_size,
                                             margin=args.margin,
                                             model_checkpoint=args.model_checkpoint)
    trainer.train()


def create_model(backbone_type, embedding_size, class_count=None, normalized_linear=False):
    if backbone_type == 'open_face':
        backbone = OpenFaceBackbone()
    elif backbone_type.startswith('efficientnet_b'):
        backbone = EfficientNetBackbone(backbone_type, pretrained_backbone=True)
    else:
        raise ValueError('Invalid backbone')

    return FaceDescriptorExtractor(backbone,
                                   embedding_size=embedding_size,
                                   class_count=class_count,
                                   normalized_linear=normalized_linear)


if __name__ == '__main__':
    main()
