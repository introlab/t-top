import argparse
import os


import numpy as np

import torch

from tqdm import tqdm

from common.metrics import RocDistancesThresholdsEvaluation
from common.modules import load_checkpoint

from object_detection.criterions.yolo_v4_loss import calculate_iou
from object_detection.datasets import CocoDetectionValidationTransforms, ObjectDetectionCoco
from object_detection.datasets import Objects365DetectionValidationTransforms, ObjectDetectionObjects365, \
    COCO_OBJECTS365_CLASS_INDEXES
from object_detection.descriptor_yolo_v7 import DescriptorYoloV7
from object_detection.datasets.object_detection_coco import CLASS_COUNT
from object_detection.filter_yolo_predictions import group_predictions, filter_yolo_predictions


COMPARABLE_CONFIDENCE_THRESHOLD = 0.01
NOT_COMPARABLE_CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
NOT_COMPARABLE_IOU_THRESHOLD = 0.5


class CocoDescriptorEvaluation(RocDistancesThresholdsEvaluation):
    def __init__(self, embeddings_class_pairs, interval, output_path):
        super(CocoDescriptorEvaluation, self).__init__(output_path, thresholds=np.arange(0, 2, 0.0001))
        self._embeddings = torch.stack([p[0] for p in embeddings_class_pairs], dim=0).half()
        self._classes = torch.stack([p[1] for p in embeddings_class_pairs], dim=0).to(torch.int16)
        self._interval = interval

        if self._embeddings.device.type == 'cuda':
            self._embeddings = self._embeddings.half()

    def _calculate_distances(self):
        N = self._embeddings.size(0)
        distances = torch.zeros(self._calculate_pair_count(N),
                                dtype=self._embeddings.dtype,
                                device=self._embeddings.device)

        k = 0
        for i in range(N):
            others = self._embeddings[i + 1::self._interval]
            distances[k:k + others.size(0)] = (self._embeddings[i].repeat(others.size(0), 1) - others).pow(2).sum(dim=1).sqrt()
            k += others.size(0)

        torch.cuda.empty_cache()
        return distances[::self._interval]

    def _get_is_same_person_target(self):
        N = self._classes.size(0)
        is_same_person_target = torch.zeros(self._calculate_pair_count(N),
                                            dtype=torch.bool,
                                            device=self._classes.device)

        k = 0
        for i in range(N):
            others = self._classes[i + 1::self._interval]
            is_same_person_target[k:k + others.size(0)] = self._classes[i]  == others
            k += others.size(0)

        torch.cuda.empty_cache()
        return is_same_person_target[::self._interval]

    def _calculate_pair_count(self, N):
        c = 0
        for i in range(N):
            c += self._embeddings[i + 1::self._interval].size(0)

        return c


def main():
    parser = argparse.ArgumentParser(description='Test the specified descriptor yolo model')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--embedding_size', type=int, help='Choose the embedding size', required=True)
    parser.add_argument('--checkpoint', type=str, help='Choose the checkpoint file path', required=True)
    parser.add_argument('--dataset_root', type=str, help='Choose the coco root path', required=True)
    parser.add_argument('--dataset_type', type=str, choices=['coco', 'objects365'], help='Choose the coco root path',
                        required=True)
    parser.add_argument('--comparable', action='store_true', help='Enable comparable results')
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    model = DescriptorYoloV7(CLASS_COUNT, embedding_size=args.embedding_size, class_probs=False)
    load_checkpoint(model, args.checkpoint)

    if args.dataset_type == 'coco':
        transforms = CocoDetectionValidationTransforms(model.get_image_size(), one_hot_class=False)
        dataset = ObjectDetectionCoco(os.path.join(args.dataset_root, 'val2017'),
                                      os.path.join(args.dataset_root, 'instances_val2017.json'),
                                      transforms)
        interval = 2 if args.comparable else 1
    elif args.dataset_type == 'objects365':
        transforms = Objects365DetectionValidationTransforms(model.get_image_size(), one_hot_class=False)
        dataset = ObjectDetectionObjects365(os.path.join(args.dataset_root),
                                            split='validation',
                                            transforms=transforms,
                                            ignored_classes=COCO_OBJECTS365_CLASS_INDEXES)
        interval = 1000 if args.comparable else 30
    else:
        raise ValueError(f'Invalid dataset ({args.dataset_type})')

    os.makedirs(args.output_path, exist_ok=True)


    evaluate(model, args.embedding_size, dataset, device, args.comparable, interval, args.output_path)


def evaluate(model, embedding_size, dataset, device, comparable, interval, output_path):
    model = model.to(device)
    model.eval()

    embeddings_class_pairs = []

    bbox_count = 0
    with torch.no_grad():
        for image, target, metadata in tqdm(dataset):
            target['bbox'] = target['bbox'].to(device)
            target['class'] = target['class'].to(device)

            bbox_count += target['bbox'].size(0)
            embeddings_class_pairs.extend(
                compute_embedding(model, embedding_size, image.to(device), target, comparable))

        torch.cuda.empty_cache()

        print(f'{len(embeddings_class_pairs)} boxes out of {bbox_count} detected')
        coco_descriptor_evaluation = CocoDescriptorEvaluation(embeddings_class_pairs, interval, output_path)
        coco_descriptor_evaluation.evaluate()


def compute_embedding(model, embedding_size, image_tensor, target, comparable):
    predictions = model(image_tensor.unsqueeze(0))
    predictions = group_predictions(predictions)[0]
    C = predictions.size(1)
    predictions = filter_yolo_predictions(predictions,
                                          confidence_threshold=COMPARABLE_CONFIDENCE_THRESHOLD if comparable else NOT_COMPARABLE_CONFIDENCE_THRESHOLD,
                                          nms_threshold=NMS_THRESHOLD)

    if len(predictions) == 0:
        print('Warning: No predictions found')
        predicted_boxes = torch.zeros(1, C).to(image_tensor.device)
    else:
        predicted_boxes = torch.stack(predictions, dim=0)

    embeddings_class_pairs = []

    for i in range(target['bbox'].size(0)):
        target_box = target['bbox'][i]
        target_class = target['class'][i]

        ious = calculate_iou(predicted_boxes[:, :4], target_box.repeat(len(predicted_boxes), 1))
        best_index = ious.argmax()
        best_predicted_box = predicted_boxes[best_index]

        if comparable or ious[best_index] > NOT_COMPARABLE_IOU_THRESHOLD:
            embeddings_class_pairs.append((best_predicted_box[-embedding_size:], target_class))

    return embeddings_class_pairs


if __name__ == '__main__':
    main()
