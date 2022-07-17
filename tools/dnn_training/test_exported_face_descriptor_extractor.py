import argparse

from common.test import load_exported_model

from face_recognition.metrics import LfwEvaluation
from face_recognition.trainers.face_descriptor_extractor_trainer import create_validation_image_transform


def main():
    parser = argparse.ArgumentParser(description='Test exported face descriptor extractor')

    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--torch_script_path', type=str, help='Choose the TorchScript path')
    parser.add_argument('--trt_path', type=str, help='Choose the TensorRT path')
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)

    args = parser.parse_args()

    model, device = load_exported_model(args.torch_script_path, args.trt_path)
    lfw_evaluation = LfwEvaluation(model, device, create_validation_image_transform(),
                                   args.dataset_root, args.output_path)
    lfw_evaluation.evaluate()


if __name__ == '__main__':
    main()
