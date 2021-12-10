import os
import json

import torch

from tqdm import tqdm

from object_detection.datasets.coco_detection_transforms import CLASS_INDEX_TO_CATEGORY_ID_MAPPING
from object_detection.modules.yolo_layer import X_INDEX, Y_INDEX, W_INDEX, H_INDEX, CONFIDENCE_INDEX, CLASSES_INDEX
from object_detection.filter_yolo_predictions import group_predictions, filter_yolo_predictions


class CocoObjectEvaluation:
    def __init__(self, model, device, dataset_loader, output_path, confidence_threshold=0.005, nms_threshold=0.45):
        self._model = model
        self._device = device
        self._dataset_loader = dataset_loader
        self._dataset = dataset_loader.dataset
        self._results_file_path = os.path.join(output_path, 'results.json')

        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self._class_count = len(CLASS_INDEX_TO_CATEGORY_ID_MAPPING)

    def evaluate(self):
        with open(self._results_file_path, 'w') as file:
            json.dump(self._get_results(), file)

        return self._dataset.evaluate(self._results_file_path)

    def _get_results(self):
        results = []
        for image, target, metadata in tqdm(self._dataset_loader):
            predictions = self._model(image.to(self._device))
            predictions = group_predictions(predictions)

            for n in range(image.size()[0]):
                image_id = metadata['image_id'][n].item()
                scale = metadata['scale'][n].item()
                results.extend(self._get_result(image_id, scale, predictions[n]))

        return results

    def _get_result(self, image_id, scale, predictions):
        predictions = filter_yolo_predictions(predictions, self._confidence_threshold, self._nms_threshold)
        if len(predictions) == 0:
            return []

        predictions = torch.stack(predictions)

        sorted_index = torch.argsort(predictions[:, CONFIDENCE_INDEX], descending=True)
        sorted_predictions = predictions[sorted_index]

        result = []

        for i in range(len(sorted_predictions)):
            center_x = sorted_predictions[i][X_INDEX].item() / scale
            center_y = sorted_predictions[i][Y_INDEX].item() / scale
            w = sorted_predictions[i][W_INDEX].item() / scale
            h = sorted_predictions[i][H_INDEX].item() / scale

            x = center_x - w / 2
            y = center_y - h / 2

            class_index = torch.argmax(sorted_predictions[i][CLASSES_INDEX:CLASSES_INDEX + self._class_count]).item()
            category_id = CLASS_INDEX_TO_CATEGORY_ID_MAPPING[class_index]

            result.append({
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [round(x), round(y), round(w), round(h)],
                'score': sorted_predictions[i][CONFIDENCE_INDEX].item(),
                'segmentation': []
            })

        return result
