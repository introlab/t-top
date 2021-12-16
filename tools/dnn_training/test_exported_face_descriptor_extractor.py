import argparse
import sys

import torch

try:
    from torch2trt import TRTModule

    torch2trt_found = True
except ImportError:
    torch2trt_found = False

from face_recognition.metrics import LfwEvaluation
from face_recognition.trainers.face_descriptor_extractor_trainer import create_validation_image_transform


def main():
    parser = argparse.ArgumentParser(description='Test exported face descriptor extractor')

    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--torch_script_path', type=str, help='Choose the TorchScript path')
    parser.add_argument('--trt_path', type=str, help='Choose the TensorRT path')
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)

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

    lfw_evaluation = LfwEvaluation(model, device, create_validation_image_transform(),
                                   args.dataset_root, args.output_path)
    lfw_evaluation.evaluate()


if __name__ == '__main__':
    main()
