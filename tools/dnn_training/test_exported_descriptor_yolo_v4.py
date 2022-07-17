import argparse
import os

import torch

from common.test import load_exported_model

from object_detection.datasets import ObjectDetectionCoco, CocoDetectionValidationTransforms
from object_detection.datasets.object_detection_open_images \
    import CLASS_COUNT_WITHOUT_HUMAN_BODY_PART as OPEN_IMAGES_CLASS_COUNT
from object_detection.metrics import CocoObjectEvaluation, YoloObjectDetectionEvaluation
from object_detection.descriptor_yolo_v4 import IMAGE_SIZE as DESCRIPTOR_YOLO_V4_IMAGE_SIZE
from object_detection.descriptor_yolo_v4_tiny import IMAGE_SIZE as DESCRIPTOR_YOLO_V4_TINY_IMAGE_SIZE


def main():
    parser = argparse.ArgumentParser(description='Test exported descriptor yolo v4')

    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--dataset_type', choices=['coco', 'open_images'], help='Choose the database type',
                        required=True)
    parser.add_argument('--torch_script_path', type=str, help='Choose the TorchScript path')
    parser.add_argument('--trt_path', type=str, help='Choose the TensorRT path')
    parser.add_argument('--model_type', choices=['yolo_v4', 'yolo_v4_tiny'],
                        help='Choose the model type', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)

    args = parser.parse_args()

    model, device = load_exported_model(args.torch_script_path, args.trt_path)

    image_size = get_image_size_from_model_type(args.model_type)
    dataset = ObjectDetectionCoco(os.path.join(args.dataset_root, 'coco/val2017'),
                                  os.path.join(args.dataset_root, 'coco/instances_val2017.json'),
                                  transforms=CocoDetectionValidationTransforms(image_size))
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    if args.dataset_type == 'coco':
        coco_pose_evaluation = CocoObjectEvaluation(model, device, dataset_loader, args.output_path)
        coco_pose_evaluation.evaluate()
    elif args.dataset_type == 'open_images':
        evaluation = YoloObjectDetectionEvaluation(model, device, dataset_loader, OPEN_IMAGES_CLASS_COUNT)
        evaluation.evaluate()
    else:
        raise ValueError('Invalid dataset type')


def get_image_size_from_model_type(model_type):
    if model_type == 'yolo_v4':
        return DESCRIPTOR_YOLO_V4_IMAGE_SIZE
    elif model_type == 'yolo_v4_tiny':
        return DESCRIPTOR_YOLO_V4_TINY_IMAGE_SIZE
    else:
        raise ValueError('Invalid model type')


if __name__ == '__main__':
    main()
