import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from dnn_utils.dnn_model import PACKAGE_PATH, DnnModel

sys.path.append(os.path.join(PACKAGE_PATH, '..', '..', '..', 'tools', 'dnn_training'))
from pose_estimation.pose_estimator import get_coordinates


IMAGE_SIZE = (256, 192)
PRESENCE_SCALE = 4.0


class PoseEstimator(DnnModel):
    def __init__(self, inference_type=None):
        torch_script_model_path = os.path.join(PACKAGE_PATH, 'models', 'pose_estimator_efficientnet_b0.ts.pth')
        tensor_rt_model_path = os.path.join(PACKAGE_PATH, 'models', 'pose_estimator_efficientnet_b0.trt.pth')
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
            image_tensor = F.interpolate(image_tensor.to(self._device).unsqueeze(0), size=IMAGE_SIZE, mode='bilinear')
            image_tensor = self._normalization(image_tensor.squeeze(0))

            pose_heatmaps = super(PoseEstimator, self).__call__(image_tensor.unsqueeze(0))
            heatmap_coordinates, presence = get_coordinates(pose_heatmaps)

            scaled_coordinates = torch.zeros(heatmap_coordinates.size()[1], 2, device=self._device)
            scaled_coordinates[:, 0] = heatmap_coordinates[0, :, 0] / pose_heatmaps.size()[3] * width
            scaled_coordinates[:, 1] = heatmap_coordinates[0, :, 1] / pose_heatmaps.size()[2] * height

            return scaled_coordinates.cpu().numpy(), presence.cpu().numpy()[0] * PRESENCE_SCALE
