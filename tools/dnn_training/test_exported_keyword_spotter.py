import argparse

import torch

from common.test import load_exported_model

from keyword_spotting.trainers.keyword_spotter_trainer import create_dataset, get_noise_root, evaluate
from keyword_spotting.datasets import KeywordSpottingTrainingTransforms


def main():
    parser = argparse.ArgumentParser(description='Test exported keyword spotter')

    parser.add_argument('--dataset_type', choices=['google_speech_commands', 'ttop_keyword'],
                        help='Choose the database type', required=True)
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--mfcc_feature_count', type=int, help='Choose the MFCC feature count', required=True)

    parser.add_argument('--torch_script_path', type=str, help='Choose the TorchScript path')
    parser.add_argument('--trt_path', type=str, help='Choose the TensorRT path')

    args = parser.parse_args()

    model, device = load_exported_model(args.torch_script_path, args.trt_path)
    transforms = KeywordSpottingTrainingTransforms(get_noise_root(args.dataset_type, args.dataset_root),
                                                   n_mfcc=args.mfcc_feature_count)
    dataset = create_dataset(args.database_type, args.dataset_root, 'testing', transforms)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    evaluate(model, device, dataset_loader)


if __name__ == '__main__':
    main()
