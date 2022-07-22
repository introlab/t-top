import argparse
import os

import torch

import torchvision.models

from common.program_arguments import save_arguments, print_arguments
from common.modules import load_checkpoint

from backbone.stdc import Stdc1, Stdc2
from backbone.trainers import BackboneDistillationTrainer
from backbone.datasets.classification_image_net import CLASS_COUNT


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--image_net_root', type=str, help='Choose the image net root path', required=True)
    parser.add_argument('--open_images_root', type=str, help='Choose the open images root path', default=None)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('--student_model_type', choices=['stdc1', 'stdc2'],
                        help='Choose the student model type', required=True)
    parser.add_argument('--teacher_model_type', choices=['efficientnet-b0', 'efficientnet-b2',
                                                         'efficientnet-b4', 'efficientnet-b7'],
                        help='Choose the teacher model type', required=True)

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--weight_decay', type=float, help='Choose the weight decay', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)

    parser.add_argument('--student_model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)
    parser.add_argument('--teacher_model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)

    args = parser.parse_args()

    student_model = create_student_model(args.student_model_type)
    teacher_model = create_teacher_model(args.teacher_model_type, args.teacher_model_checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    output_path = os.path.join(args.output_path, args.student_model_type + '_' + args.teacher_model_type + '_' +
                               '_lr' + str(args.learning_rate) + '_wd' + str(args.weight_decay))
    save_arguments(output_path, args)
    print_arguments(args)

    trainer = BackboneDistillationTrainer(device, student_model, teacher_model,
                                          epoch_count=args.epoch_count,
                                          learning_rate=args.learning_rate,
                                          weight_decay=args.weight_decay,
                                          image_net_root=args.image_net_root,
                                          open_images_root=args.open_images_root,
                                          output_path=output_path,
                                          batch_size=args.batch_size,
                                          student_model_checkpoint=args.student_model_checkpoint)
    trainer.train()


def create_student_model(student_model_type):
    if student_model_type == 'stdc1':
        return Stdc1(class_count=CLASS_COUNT, dropout=0.0)
    elif student_model_type == 'stdc2':
        return Stdc2(class_count=CLASS_COUNT, dropout=0.0)
    else:
        raise ValueError('Invalid student type')


def create_teacher_model(teacher_model_type, teacher_model_checkpoint):
    pretrained = teacher_model_checkpoint is None

    if teacher_model_type == 'efficientnet-b0':
        teacher_model = torchvision.models.efficientnet_b0(pretrained=pretrained)
    elif teacher_model_type == 'efficientnet-b2':
        teacher_model = torchvision.models.efficientnet_b2(pretrained=pretrained)
    elif teacher_model_type == 'efficientnet-b4':
        teacher_model = torchvision.models.efficientnet_b4(pretrained=pretrained)
    elif teacher_model_type == 'efficientnet-b7':
        teacher_model = torchvision.models.efficientnet_b7(pretrained=pretrained)
    else:
        raise ValueError('Invalid teacher type')

    if not pretrained:
        load_checkpoint(teacher_model, teacher_model_checkpoint, strict=True)

    return teacher_model


if __name__ == '__main__':
    main()
