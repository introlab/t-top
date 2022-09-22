import argparse
import os

import torch

from common.program_arguments import save_arguments, print_arguments

from face_recognition.face_descriptor_extractor import FaceDescriptorExtractor
from face_recognition.trainers import FaceDescriptorExtractorTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--vvgface2_dataset_root', type=str, help='Choose the Vggface2 root path', required=True)
    parser.add_argument('--lfw_dataset_root', type=str, help='Choose the LFW root path', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('--embedding_size', type=int, help='Set the embedding size', required=True)
    parser.add_argument('--margin', type=float, help='Set the margin', required=True)

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--weight_decay', type=float, help='Choose the weight decay', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)
    parser.add_argument('--criterion_type', choices=['triplet_loss', 'cross_entropy_loss', 'am_softmax_loss'],
                        help='Choose the criterion type', required=True)
    parser.add_argument('--dataset_class_count', type=int,
                        help='Choose the dataset class count when criterion_type is "cross_entropy_loss" or '
                             '"am_softmax_loss"',
                        default=None)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)

    args = parser.parse_args()

    if args.criterion_type == 'triplet_loss' and args.dataset_class_count is None:
        model = create_model(args.embedding_size)
    elif args.criterion_type == 'cross_entropy_loss' and args.dataset_class_count is not None:
        model = create_model(args.embedding_size, args.dataset_class_count)
    elif args.criterion_type == 'am_softmax_loss' and args.dataset_class_count is not None:
        model = create_model(args.embedding_size, args.dataset_class_count, am_softmax_linear=True)
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


def create_model(embedding_size, class_count=None, am_softmax_linear=False):
    return FaceDescriptorExtractor(embedding_size=embedding_size,
                                   class_count=class_count,
                                   am_softmax_linear=am_softmax_linear)


if __name__ == '__main__':
    main()
