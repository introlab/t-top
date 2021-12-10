import argparse
import os

import torch

from keyword_spotting.keyword_spotter import KeywordSpotter
from keyword_spotting.trainers import KeywordSpotterTrainer


# Train a model like : https://github.com/microsoft/human-pose-estimation.pytorch
def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('--database_type', choices=['google_speech_commands', 'ttop_keyword'],
                        help='Choose the database type', required=True)
    parser.add_argument('--mfcc_feature_count', type=int, help='Choose the MFCC feature count', required=True)

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--batch_size_division', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)
    parser.add_argument('--optimizer_checkpoint', type=str, help='Choose the optimizer checkpoint file', default=None)
    parser.add_argument('--scheduler_checkpoint', type=str, help='Choose the scheduler checkpoint file', default=None)

    args = parser.parse_args()

    model = create_model(args.database_type)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    output_path = os.path.join(args.output_path, args.database_type,
                               'mfcc_feature_count_' + str(args.mfcc_feature_count))
    trainer = KeywordSpotterTrainer(device, model,
                                    dataset_type=args.database_type,
                                    mfcc_feature_count=args.mfcc_feature_count,
                                    epoch_count=args.epoch_count,
                                    learning_rate=args.learning_rate,
                                    dataset_root=args.dataset_root,
                                    output_path=output_path,
                                    batch_size=args.batch_size,
                                    batch_size_division=args.batch_size_division,
                                    model_checkpoint=args.model_checkpoint,
                                    optimizer_checkpoint=args.optimizer_checkpoint,
                                    scheduler_checkpoint=args.scheduler_checkpoint)
    trainer.train()


def create_model(database_type):
    if database_type == 'google_speech_commands':
        return KeywordSpotter(class_count=36, use_softmax=False)
    elif database_type == 'ttop_keyword':
        return KeywordSpotter(class_count=2, use_softmax=False)
    else:
        raise ValueError('Invalid database type')


if __name__ == '__main__':
    main()
