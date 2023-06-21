import argparse
import os

import torch

from common.modules import load_checkpoint

from object_detection.datasets.yolo_collate import yolo_collate
from object_detection.datasets.coco_detection_transforms import CocoDetectionValidationTransforms
from object_detection.datasets.object_detection_coco import ObjectDetectionCoco
from object_detection.metrics import CocoObjectEvaluation
from object_detection.modules.test_converted_yolo import create_model


def main():
    parser = argparse.ArgumentParser(description='Test the specified converted model')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--model_type', choices=['yolo_v4', 'yolo_v4_tiny', 'yolo_v7', 'yolo_v7_tiny'],
                        help='Choose the mode', required=True)
    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', required=True)
    parser.add_argument('--coco_root', type=str, help='Choose the image file', required=True)
    parser.add_argument('--batch_size', type=int, help='Choose the batch size', default=4)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    model = create_model(args.model_type, class_probs=False)
    load_checkpoint(model, args.model_checkpoint)
    model.eval()

    dataset = ObjectDetectionCoco(
        os.path.join(args.coco_root, 'val2017'),
        os.path.join(args.coco_root, 'instances_val2017.json'),
        transforms=CocoDetectionValidationTransforms(model.get_image_size(), one_hot_class=True))
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              collate_fn=yolo_collate,
                                              shuffle=False,
                                              num_workers=1)

    os.makedirs(args.output_path, exist_ok=True)
    evaluation = CocoObjectEvaluation(model.to(device), device, data_loader, args.output_path)
    evaluation.evaluate()


if __name__ == '__main__':
    main()
