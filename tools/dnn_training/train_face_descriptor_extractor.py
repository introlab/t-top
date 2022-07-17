import argparse

import torch

from common.program_arguments import save_arguments, print_arguments

from face_recognition.face_descriptor_extractor import FaceDescriptorExtractor
from face_recognition.trainers import FaceDescriptorExtractorTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('--embedding_size', type=int, help='Set the embedding size', required=True)
    parser.add_argument('--margin', type=float, help='Set the margin', required=True)

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)

    args = parser.parse_args()
    save_arguments(args.output_path, args)
    print_arguments(args)

    model = create_model(args.embedding_size)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    trainer = FaceDescriptorExtractorTrainer(device, model,
                                             epoch_count=args.epoch_count,
                                             learning_rate=args.learning_rate,
                                             dataset_root=args.dataset_root,
                                             output_path=args.output_path,
                                             batch_size=args.batch_size,
                                             margin=args.margin,
                                             model_checkpoint=args.model_checkpoint)
    trainer.train()


def create_model(embedding_size):
    return FaceDescriptorExtractor(embedding_size=embedding_size)


if __name__ == '__main__':
    main()
