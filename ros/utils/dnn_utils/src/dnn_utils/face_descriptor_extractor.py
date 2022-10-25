import os
import sys

import numpy as np

import cv2

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from dnn_utils.dnn_model import PACKAGE_PATH, DnnModel

sys.path.append(os.path.join(PACKAGE_PATH, '..', '..', '..', 'tools', 'dnn_training'))
from face_recognition.datasets.align_faces import get_landmarks_from_pose, cv2_transform_to_theta


IMAGE_SIZE = (128, 96)


class FaceDescriptorExtractor(DnnModel):
    def __init__(self, inference_type=None):
        torch_script_model_path = os.path.join(PACKAGE_PATH, 'models', 'face_descriptor_extractor.ts.pth')
        tensor_rt_model_path = os.path.join(PACKAGE_PATH, 'models', 'face_descriptor_extractor.trt.pth')
        sample_input = torch.ones((1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))

        super(FaceDescriptorExtractor, self).__init__(torch_script_model_path, tensor_rt_model_path, sample_input,
                                                      inference_type=inference_type)
        self._normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_supported_image_size(self):
        return IMAGE_SIZE

    def __call__(self, image_tensor, pose_coordinates, pose_presence):
        landmarks, theoretical_landmark = get_landmarks_from_pose(pose_coordinates, pose_presence)
        transform = cv2.getAffineTransform(landmarks.astype(np.float32),
                                           (theoretical_landmark * np.array((IMAGE_SIZE[1], IMAGE_SIZE[0]))).astype(np.float32))
        try:
            theta = cv2_transform_to_theta(transform, image_tensor.size(1), image_tensor.size(2), IMAGE_SIZE[0], IMAGE_SIZE[1])
        except np.linalg.LinAlgError:
            raise ValueError('Invalid face pose')

        with torch.no_grad():
            grid = F.affine_grid(theta, torch.Size((1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))).to(self._device)
            aligned_image = F.grid_sample(image_tensor.unsqueeze(0).to(self._device), grid, mode='nearest').squeeze(0)
            cv2_aligned_image = (255 * aligned_image.permute(1, 2, 0)).to(torch.uint8).cpu().numpy()

            normalized_aligned_image = self._normalization(aligned_image)
            return super(FaceDescriptorExtractor, self).__call__(normalized_aligned_image.unsqueeze(0))[0].cpu(), cv2_aligned_image
