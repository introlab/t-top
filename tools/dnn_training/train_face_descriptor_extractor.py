import argparse
import os

import torch

from common.program_arguments import save_arguments, print_arguments

from face_recognition.face_descriptor_extractor import FaceDescriptorExtractor, OpenFaceBackbone, EfficientNetBackbone
from face_recognition.trainers import FaceDescriptorExtractorTrainer, FaceDescriptorExtractorDistillationTrainer

BACKBONE_TYPES = ['open_face', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                  'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--dataset_roots', nargs='+', type=str, help='Choose the Vggface2 root path', required=True)
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

    parser.add_argument('--teacher_backbone_type', choices=BACKBONE_TYPES, help='Choose the teacher backbone type',
                        default=None)
    parser.add_argument('--teacher_model_checkpoint', type=str, help='Choose the teacher model checkpoint file',
                        default=None)

    args = parser.parse_args()

    model = create_model_from_criterion_type(args.criterion_type, args.dataset_class_count, args.backbone_type,
                                             args.embedding_size)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    output_path = os.path.join(args.output_path, args.backbone_type + '_e' + str(args.embedding_size) +
                               '_' + args.criterion_type + '_lr' + str(args.learning_rate) +
                               '_wd' + str(args.weight_decay) + '_m' + str(args.margin) +
                               '_t' + str(args.teacher_backbone_type))
    save_arguments(output_path, args)
    print_arguments(args)

    if args.teacher_backbone_type is not None and args.teacher_model_checkpoint is not None:
        teacher_model = create_model_from_criterion_type(args.criterion_type, args.dataset_class_count,
                                                         args.teacher_backbone_type, args.embedding_size)
        trainer = FaceDescriptorExtractorDistillationTrainer(device, model, teacher_model,
                                                             epoch_count=args.epoch_count,
                                                             learning_rate=args.learning_rate,
                                                             weight_decay=args.weight_decay,
                                                             criterion_type=args.criterion_type,
                                                             dataset_roots=args.dataset_roots,
                                                             lfw_dataset_root=args.lfw_dataset_root,
                                                             output_path=output_path,
                                                             batch_size=args.batch_size,
                                                             margin=args.margin,
                                                             student_model_checkpoint=args.model_checkpoint,
                                                             teacher_model_checkpoint=args.teacher_model_checkpoint)
    elif args.teacher_backbone_type is not None or args.teacher_model_checkpoint is not None:
        raise ValueError('teacher_backbone_type and teacher_model_checkpoint must be set.')
    else:
        trainer = FaceDescriptorExtractorTrainer(device, model,
                                                 epoch_count=args.epoch_count,
                                                 learning_rate=args.learning_rate,
                                                 weight_decay=args.weight_decay,
                                                 criterion_type=args.criterion_type,
                                                 dataset_roots=args.dataset_roots,
                                                 lfw_dataset_root=args.lfw_dataset_root,
                                                 output_path=output_path,
                                                 batch_size=args.batch_size,
                                                 margin=args.margin,
                                                 model_checkpoint=args.model_checkpoint)
    trainer.train()


def create_model_from_criterion_type(criterion_type, dataset_class_count, backbone_type, embedding_size):
    if criterion_type == 'triplet_loss' and dataset_class_count is None:
        return create_model(backbone_type, embedding_size)
    elif criterion_type == 'cross_entropy_loss' and dataset_class_count is not None:
        return create_model(backbone_type, embedding_size, dataset_class_count)
    elif criterion_type == 'am_softmax_loss' or criterion_type == 'arc_face_loss' \
            and dataset_class_count is not None:
        return create_model(backbone_type, embedding_size, dataset_class_count, normalized_linear=True)
    else:
        raise ValueError('--dataset_class_count must be used with "cross_entropy_loss" or "am_softmax_loss" types')


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
