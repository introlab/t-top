import argparse
import os

import torch

from common.program_arguments import save_arguments, print_arguments

from object_detection.datasets.object_detection_coco import CLASS_COUNT as COCO_CLASS_COUNT
from object_detection.datasets.object_detection_open_images \
    import CLASS_COUNT_WITHOUT_HUMAN_BODY_PART as OPEN_IMAGES_CLASS_COUNT
from object_detection.trainers.descriptor_yolo_v4_trainer import DescriptorYoloV4Trainer
from object_detection.descriptor_yolo_v4 import DescriptorYoloV4
from object_detection.descriptor_yolo_v4_tiny import DescriptorYoloV4Tiny
from object_detection.descriptor_yolo_v7 import DescriptorYoloV7


def main():
    parser = argparse.ArgumentParser(description='Train Descriptor Yolo v4')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--dataset_type', choices=['coco', 'open_images'], help='Choose the database type',
                        required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('--model_type', choices=['yolo_v4', 'yolo_v4_tiny', 'yolo_v7'],
                        help='Choose the model type', required=True)
    parser.add_argument('--descriptor_size', type=int, help='Choose the descriptor size', required=True)
    parser.add_argument('--class_criterion_type', choices=['cross_entropy_loss', 'bce_loss', 'sigmoid_focal_loss'],
                        help='Choose the class criterion type', required=True)

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--batch_size_division', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)

    args = parser.parse_args()

    model = create_model(args.model_type, args.descriptor_size, args.dataset_type)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    output_path = os.path.join(args.output_path, 'descriptor_yolo_v4', args.model_type)
    save_arguments(output_path, args)
    print_arguments(args)

    trainer = DescriptorYoloV4Trainer(device, model,
                                      epoch_count=args.epoch_count,
                                      learning_rate=args.learning_rate,
                                      dataset_root=args.dataset_root,
                                      dataset_type=args.dataset_type,
                                      class_criterion_type=args.class_criterion_type,
                                      output_path=output_path,
                                      batch_size=args.batch_size,
                                      batch_size_division=args.batch_size_division,
                                      model_checkpoint=args.model_checkpoint)
    trainer.train()


def create_model(model_type, descriptor_size, dataset_type):
    class_count = _get_class_count(dataset_type)
    if model_type == 'yolo_v4':
        model = DescriptorYoloV4(class_count, descriptor_size)
    elif model_type == 'yolo_v4_tiny':
        model = DescriptorYoloV4Tiny(class_count, descriptor_size)
    elif model_type == 'yolo_v7':
        model = DescriptorYoloV7(class_count, descriptor_size)
    else:
        raise ValueError('Invalid model type')

    return model


def _get_class_count(dataset_type):
    if dataset_type == 'coco':
        return COCO_CLASS_COUNT
    elif dataset_type == 'open_images':
        return OPEN_IMAGES_CLASS_COUNT
    elif dataset_type == 'objects365':
        return 365
    else:
        raise ValueError('Invalid dataset type')


if __name__ == '__main__':
    main()
