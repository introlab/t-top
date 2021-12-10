import os
import json

from tqdm import tqdm

from pose_estimation.datasets.pose_estimation_coco import COCO_PERSON_CATEGORY_ID
from pose_estimation.pose_estimator import get_coordinates


class CocoPoseEvaluation:
    def __init__(self, model, device, dataset_loader, output_path, presence_threshold=0.0):
        self._model = model
        self._device = device
        self._dataset_loader = dataset_loader
        self._dataset = dataset_loader.dataset
        self._result_file_path = os.path.join(output_path, 'results.json')

        self._presence_threshold = presence_threshold

    def evaluate(self):
        with open(self._result_file_path, 'w') as result_file:
            json.dump(self._get_results(), result_file)

        return self._dataset.evaluate(self._result_file_path)

    def _get_results(self):
        results = []
        for image, _, metadata in tqdm(self._dataset_loader):
            heatmap_prediction = self._model(image.to(self._device))

            for n in range(heatmap_prediction.size()[0]):
                results.append(self._get_result(metadata['annotation_id'][n].item(),
                                                metadata['image_id'][n].item(),
                                                heatmap_prediction[n]))

        return results

    def _get_result(self, annotation_id, image_id, heatmap_prediction):
        predicted_coordinates, presence_prediction = get_coordinates(heatmap_prediction.unsqueeze(0))

        heatmap_width = heatmap_prediction.size()[2]
        heatmap_height = heatmap_prediction.size()[1]

        keypoints = []
        for i in range(predicted_coordinates.size()[1]):
            keypoints.append(predicted_coordinates[0, i, 0].item())  # x
            keypoints.append(predicted_coordinates[0, i, 1].item())  # y
            keypoints.append(1)

        keypoints = self._dataset.transform_keypoints(annotation_id, keypoints, heatmap_width, heatmap_height)
        for i in range(predicted_coordinates.size()[1]):
            if presence_prediction[0, i] < self._presence_threshold:
                keypoints[3 * i] = 0
                keypoints[3 * i + 1] = 0

        return {
            'image_id': image_id,
            'category_id': COCO_PERSON_CATEGORY_ID,
            'keypoints': keypoints,
            'score': presence_prediction.mean().item()
        }
