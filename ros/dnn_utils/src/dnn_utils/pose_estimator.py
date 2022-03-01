import os
import sys
import time

import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from dnn_utils.dnn_model import PACKAGE_PATH, DnnModel

sys.path.append(os.path.join(PACKAGE_PATH, '..', '..', 'tools', 'dnn_training'))
from pose_estimation.pose_estimator import get_coordinates


IMAGE_SIZE = (256, 192)

class Stopwatch:
    def __init__(self, prefix):
        self._prefix = 'pose_estimator.py' + prefix

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, *args):
        print(self._prefix + ' - elapsed time', time.time() - self._start)


class PoseEstimator(DnnModel):
    def __init__(self, inference_type=None):
        torch_script_model_path = os.path.join(PACKAGE_PATH, 'models', 'pose_estimator.ts.pth')
        tensor_rt_model_path = os.path.join(PACKAGE_PATH, 'models', 'pose_estimator.trt.pth')
        sample_input = torch.ones((1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))

        super(PoseEstimator, self).__init__(torch_script_model_path, tensor_rt_model_path, sample_input,
                                            inference_type=inference_type)
        self._normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_supported_image_size(self):
        return IMAGE_SIZE

    def get_keypoint_names(self):
        return ['nose',
                'left eye',
                'right eye',
                'left ear',
                'right ear',
                'left shoulder',
                'right shoulder',
                'left elbow',
                'right elbow',
                'left wrist',
                'right wrist',
                'left hip',
                'right hip',
                'left knee',
                'right knee',
                'left ankle',
                'right ankle']

    def get_skeleton_pairs(self):
        return [[0, 1],
                [0, 2],
                [1, 2],
                [1, 3],
                [2, 4],
                [3, 5],
                [4, 6],
                [5, 6],
                [5, 7],
                [5, 11],
                [6, 8],
                [6, 12],
                [7, 9],
                [8, 10],
                [11, 12],
                [13, 11],
                [14, 12],
                [15, 13],
                [16, 14]]

    def __call__(self, image_tensor):
        with torch.no_grad():
            height = image_tensor.size(1)
            width = image_tensor.size(2)
            with Stopwatch('transforms'):
                image_tensor = F.interpolate(image_tensor.to(self._device).unsqueeze(0), size=IMAGE_SIZE, mode='bilinear')
                image_tensor = self._normalization(image_tensor.squeeze(0))
            with Stopwatch('dnn'):
                pose_heatmaps = super(PoseEstimator, self).__call__(image_tensor.unsqueeze(0))

            with Stopwatch('get_coordinates'):
                heatmap_coordinates, presence = get_coordinates(pose_heatmaps)
                heatmap_coordinates = heatmap_coordinates.cpu()
                presence = presence.cpu()

            with Stopwatch('for'):
                scaled_coordinates = np.zeros((heatmap_coordinates.size()[1], 2))
                for i in range(heatmap_coordinates.size()[1]):
                    scaled_coordinates[i, 0] = heatmap_coordinates[0, i, 0] / pose_heatmaps.size()[3] * width
                    scaled_coordinates[i, 1] = heatmap_coordinates[0, i, 1] / pose_heatmaps.size()[2] * height

            return scaled_coordinates, presence[0]
