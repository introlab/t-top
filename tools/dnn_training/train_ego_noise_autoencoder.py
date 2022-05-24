import argparse
import os

import torch

from ego_noise.ego_noise_autoencoder import PcaEgoNoiseAutoEncoderModel, TwoLayerNoiseAutoEncoderModel
from ego_noise.trainers import EgoNoiseAutoencoderTrainer


# Train a model like : https://github.com/microsoft/human-pose-estimation.pytorch
def main():
    parser = argparse.ArgumentParser(description='Train Ego Noise Autoencoder')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path (LibriSpeech)', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('--model_type', choices=['pca', 'two_layer'], help='Choose the model type', required=True)
    parser.add_argument('--embedding_size', type=int, help='Set the margin', required=True)
    parser.add_argument('--sample_rate', type=int, help='Set the sample rate', required=True)
    parser.add_argument('--n_fft', type=int, help='Set the N FFT', required=True)

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)
    parser.add_argument('--optimizer_checkpoint', type=str, help='Choose the optimizer checkpoint file', default=None)
    parser.add_argument('--scheduler_checkpoint', type=str, help='Choose the scheduler checkpoint file', default=None)

    args = parser.parse_args()

    model = create_model(args.model_type, args.n_fft, args.embedding_size)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    output_path = os.path.join(args.output_path, 'ego_noise_' + args.model_type + '_' + str(args.embedding_size))
    trainer = EgoNoiseAutoencoderTrainer(device, model,
                                         epoch_count=args.epoch_count,
                                         learning_rate=args.learning_rate,
                                         dataset_root=args.dataset_root,
                                         output_path=output_path,
                                         batch_size=args.batch_size,
                                         sample_rate=args.sample_rate,
                                         n_fft=args.n_fft,
                                         model_checkpoint=args.model_checkpoint,
                                         optimizer_checkpoint=args.optimizer_checkpoint,
                                         scheduler_checkpoint=args.scheduler_checkpoint)
    trainer.train()


def create_model(model_type, n_fft, embedding_size):
    input_size = n_fft // 2 + 1
    if model_type == 'pca':
        return PcaEgoNoiseAutoEncoderModel(input_size, embedding_size)
    elif model_type == 'two_layer':
        return TwoLayerNoiseAutoEncoderModel(input_size, embedding_size)
    else:
        raise ValueError('Invalid model type')


if __name__ == '__main__':
    main()
