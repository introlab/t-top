import argparse
import sys

import torch

try:
    from torch2trt import TRTModule

    torch2trt_found = True
except ImportError:
    torch2trt_found = False

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

    if args.torch_script_path is not None:
        device = torch.device('cpu')
        model = torch.jit.load(args.torch_script_path)
        model = model.to(device)
    elif args.trt_path is not None:
        if not torch2trt_found:
            print('"torch2trt" is not supported.')
            sys.exit()
        else:
            device = torch.device('cuda')
            model = TRTModule()
            model.load_state_dict(torch.load(args.trt_path))
    else:
        print('"torch_script_path" or "trt_path" is required.')
        sys.exit()

    transforms = KeywordSpottingTrainingTransforms(get_noise_root(args.dataset_type, args.dataset_root),
                                                   n_mfcc=args.mfcc_feature_count)
    dataset = create_dataset(args.database_type, args.dataset_root, 'testing', transforms)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    evaluate(model, device, dataset_loader)


if __name__ == '__main__':
    main()
