import argparse
import os

import torch

from common.program_arguments import save_arguments, print_arguments

from export_keyword_spotter import create_model
from keyword_spotting.trainers import KeywordSpotterTrainer


# Train a model like : https://github.com/microsoft/human-pose-estimation.pytorch
def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('--dataset_type', choices=['google_speech_commands', 'ttop_keyword'],
                        help='Choose the database type', required=True)
    parser.add_argument('--mfcc_feature_count', type=int, help='Choose the MFCC feature count', required=True)

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--weight_decay', type=float, help='Choose the weight decay', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--batch_size_division', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)

    args = parser.parse_args()

    model = create_model(args.dataset_type)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    output_path = os.path.join(args.output_path, args.dataset_type,
                               'mfcc_feature_count_' + str(args.mfcc_feature_count))
    save_arguments(output_path, args)
    print_arguments(args)

    trainer = KeywordSpotterTrainer(device, model,
                                    dataset_type=args.dataset_type,
                                    mfcc_feature_count=args.mfcc_feature_count,
                                    epoch_count=args.epoch_count,
                                    learning_rate=args.learning_rate,
                                    weight_decay=args.weight_decay,
                                    dataset_root=args.dataset_root,
                                    output_path=output_path,
                                    batch_size=args.batch_size,
                                    batch_size_division=args.batch_size_division,
                                    model_checkpoint=args.model_checkpoint)
    trainer.train()


if __name__ == '__main__':
    main()
