import argparse
import json
import os

from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms
import torchvision.transforms.functional as TF

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from train_pose_estimator import create_model as create_pose_estimator_model, BACKBONE_TYPES as POSE_BACKBONE_TYPES

from common.modules import load_checkpoint

from object_detection.datasets import ObjectDetectionCoco, CocoDetectionValidationTransforms
from object_detection.filter_yolo_predictions import group_predictions, filter_yolo_predictions_by_classes
from object_detection.modules.yolo_layer import X_INDEX, Y_INDEX, W_INDEX, H_INDEX, CONFIDENCE_INDEX
from object_detection.modules.yolo_layer import CLASSES_INDEX
from object_detection.modules.test_converted_yolo import create_model as create_yolo_model

from pose_estimation.trainers.pose_estimator_trainer import IMAGE_SIZE as POSE_ESTIMATOR_IMAGE_SIZE
from pose_estimation.datasets.pose_estimation_coco import COCO_PERSON_CATEGORY_ID
from pose_estimation.pose_estimator import get_coordinates


PERSON_CLASS_INDEX = 0
BBOX_SCALE = 1.25


class ObjectDetectionFolder(Dataset):
    def __init__(self, image_root_path, transforms=None):
        self._image_root_path = image_root_path
        self._image_filenames = os.listdir(self._image_root_path)
        self._image_filenames.sort()

        if transforms is None:
            raise ValueError('Invalid transforms')
        self._transforms = transforms

    def __getitem__(self, index):
        image = Image.open(os.path.join(self._image_root_path, self._image_filenames[index])).convert('RGB')
        image_id = int(os.path.splitext(self._image_filenames[index])[0])

        initial_width, initial_height = image.size

        target = None
        image, target, transforms_metadata = self._transforms(image, target)
        metadata = {
            'image_id': image_id,
            'initial_width': initial_width,
            'initial_height': initial_height,
            'scale': transforms_metadata['scale'],
            'offset_x': transforms_metadata['offset_x'],
            'offset_y': transforms_metadata['offset_y']
        }
        return image, target, metadata

    def __len__(self):
        return len(self._image_filenames)


# TODO Refactor to reduce code duplication
class CocoPoseEvaluationWithYolo():
    def __init__(self, yolo_model, pose_estimator_model, device, dataset_root, dataset_split, output_path,
                 confidence_threshold=0.01, nms_threshold=0.5, presence_threshold=0.0):
        self._device = device
        self._yolo_model = yolo_model.to(device)
        self._pose_estimator_model = pose_estimator_model.to(device)

        transforms = CocoDetectionValidationTransforms(yolo_model.get_image_size(), one_hot_class=False)

        if dataset_split == 'validation':
            self._image_root_path = os.path.join(dataset_root, 'val2017')
            self._dataset = ObjectDetectionCoco(
                    self._image_root_path,
                    os.path.join(dataset_root, 'instances_val2017.json'),
                    transforms=transforms)

            self._annotation_file_path = os.path.join(dataset_root, 'person_keypoints_val2017.json')
        elif dataset_split == 'test':
            self._image_root_path = os.path.join(dataset_root, 'test2017')
            self._dataset = ObjectDetectionFolder(
                self._image_root_path,
                transforms=transforms)

            self._annotation_file_path = None

        os.makedirs(output_path, exist_ok=True)
        self._result_file_path = os.path.join(output_path, '{}_results.json'.format(dataset_split))

        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self._presence_threshold = presence_threshold

        self._pose_estimator_normalization = \
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def evaluate(self):
        self._yolo_model.eval()
        self._pose_estimator_model.eval()

        with open(self._result_file_path, 'w') as result_file:
            json.dump(self._get_results(), result_file)

        return self._evaluate_coco()

    def _get_results(self):
        with torch.no_grad():
            results = []
            for image, _, metadata in tqdm(self._dataset):
                yolo_predictions = self._yolo_model.forward(image.unsqueeze(0).to(self._device))
                yolo_predictions = group_predictions(yolo_predictions)[0]
                yolo_predictions = filter_yolo_predictions_by_classes(yolo_predictions,
                                                                      confidence_threshold=self._confidence_threshold,
                                                                      nms_threshold=self._nms_threshold)

                results.extend(self._get_image_results(yolo_predictions, metadata))

            return results

    def _get_image_results(self, yolo_predictions, metadata):
        image_id = metadata['image_id']
        scale = metadata['scale']
        offset_x = metadata['offset_x']
        offset_y = metadata['offset_y']
        initial_width = metadata['initial_width']
        initial_height = metadata['initial_height']

        results = []
        for yolo_prediction in yolo_predictions:
            class_probs = yolo_prediction[CLASSES_INDEX:]
            class_index = torch.argmax(class_probs, dim=0).item()
            confidence = yolo_prediction[CONFIDENCE_INDEX].item()
            if class_index != PERSON_CLASS_INDEX or confidence < self._confidence_threshold:
                continue

            center_x = ((yolo_prediction[X_INDEX] - offset_x) / scale).item()
            center_y = ((yolo_prediction[Y_INDEX] - offset_y) / scale).item()
            width = (yolo_prediction[W_INDEX] / scale).item() * BBOX_SCALE
            height = (yolo_prediction[H_INDEX] / scale).item() * BBOX_SCALE

            x0 = np.clip(int(center_x - width / 2), 0, initial_width)
            y0 = np.clip(int(center_y - height / 2), 0, initial_height)
            x1 = np.clip(int(center_x + width / 2), 0, initial_width)
            y1 = np.clip(int(center_y + height / 2), 0, initial_height)
            if x0 >= x1 or y0 >= y1:
                continue

            heatmap_prediction = self._get_heatmap_prediction(image_id, x0, y0, x1, y1)
            results.append(self._get_result(image_id, x0, y0, x1, y1, confidence, heatmap_prediction))

        return results

    def _get_heatmap_prediction(self, image_id, x0, y0, x1, y1):
        file = '{:012d}.jpg'.format(image_id)
        image_tensor = TF.to_tensor(Image.open(os.path.join(self._image_root_path, file)).convert('RGB'))
        image_tensor = image_tensor[:, y0:y1, x0:x1].to(self._device)

        image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=POSE_ESTIMATOR_IMAGE_SIZE, mode='bilinear')
        image_tensor = self._pose_estimator_normalization(image_tensor.squeeze(0))

        heatmap_prediction = self._pose_estimator_model(image_tensor.unsqueeze(0))[0]
        return heatmap_prediction

    def _get_result(self, image_id, x0, y0, x1, y1, confidence, heatmap_prediction):
        predicted_coordinates, presence_prediction = get_coordinates(heatmap_prediction.unsqueeze(0))

        heatmap_width = heatmap_prediction.size()[2]
        heatmap_height = heatmap_prediction.size()[1]

        original_width = x1 - x0
        original_height = y1 - y0
        keypoints = []
        for i in range(predicted_coordinates.size()[1]):
            if presence_prediction[0, i] < self._presence_threshold:
                keypoints.append(0)
                keypoints.append(0)
            else:
                x = predicted_coordinates[0, i, 0].item() / heatmap_width * original_width + x0
                y = predicted_coordinates[0, i, 1].item() / heatmap_height * original_height + y0
                keypoints.append(round(x))  # x
                keypoints.append(round(y))  # y

            keypoints.append(1)

        return {
            'image_id': image_id,
            'category_id': COCO_PERSON_CATEGORY_ID,
            'keypoints': keypoints,
            'score': presence_prediction.mean().item() * confidence
        }

    def _evaluate_coco(self):
        if self._annotation_file_path is None:
            return [], []

        coco_gt = COCO(self._annotation_file_path)
        coco_dt = coco_gt.loadRes(self._result_file_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        return coco_eval.stats, names


def main():
    parser = argparse.ArgumentParser(description='Test pose estimator with detected person')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--dataset_split', choices=['validation', 'test'], required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)

    parser.add_argument('--yolo_model_type', choices=['yolo_v4', 'yolo_v4_tiny', 'yolo_v7', 'yolo_v7_tiny'],
                        help='Choose the model type', required=True)
    parser.add_argument('--yolo_model_checkpoint', type=str, help='Choose the model checkpoint file for YOLO',
                        required=True)

    parser.add_argument('--pose_backbone_type', choices=POSE_BACKBONE_TYPES,
                        help='Choose the backbone type', required=True)
    parser.add_argument('--pose_model_checkpoint', type=str, help='Choose the model checkpoint file for the pose',
                        required=True)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    yolo_model = create_yolo_model(args.yolo_model_type, class_probs=True)
    load_checkpoint(yolo_model, args.yolo_model_checkpoint)

    pose_estimator_model = create_pose_estimator_model(args.pose_backbone_type)
    load_checkpoint(pose_estimator_model, args.pose_model_checkpoint)

    evaluation = CocoPoseEvaluationWithYolo(yolo_model, pose_estimator_model, device,
                                            args.dataset_root, args.dataset_split, args.output_path)
    evaluation.evaluate()


if __name__ == '__main__':
    main()
