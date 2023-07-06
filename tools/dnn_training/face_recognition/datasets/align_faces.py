import os
import argparse
import math
import shutil

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm

from common.modules import load_checkpoint

from pose_estimation.pose_estimator import get_coordinates
from pose_estimation.trainers.pose_estimator_trainer import IMAGE_SIZE as POSE_ESTIMATOR_IMAGE_SIZE

from train_pose_estimator import create_model, BACKBONE_TYPES

ALIGNED_IMAGE_SIZE = (128, 96)


class FolderFaceAligner:
    def __init__(self, pose_estimator_model, device, presence_threshold, ignore_presence_threshold_for_nose_eyes):
        self._device = device
        self._pose_estimator_model = pose_estimator_model.to(device)
        self._pose_estimator_image_transform = transforms.Compose([
            transforms.Resize(POSE_ESTIMATOR_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._presence_threshold = presence_threshold
        self._ignore_presence_threshold_for_nose_eyes = ignore_presence_threshold_for_nose_eyes

    def align(self, input_path, output_path):
        person_names = [o for o in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, o))]
        for person_name in tqdm(person_names):
            self._align_person_images(input_path, output_path, person_name)

    def _align_person_images(self, input_path, output_path, person_name):
        os.makedirs(os.path.join(output_path, person_name), exist_ok=True)

        image_filenames = [o for o in os.listdir(os.path.join(input_path, person_name))]
        for image_filename in image_filenames:
            try:
                self._align_person_image(input_path, output_path, person_name, image_filename)
            except (ValueError, np.linalg.LinAlgError):
                print('Warning: the alignment is impossible ({})'.format(image_filename))
                if self._ignore_presence_threshold_for_nose_eyes:
                    shutil.copyfile(os.path.join(input_path, person_name, image_filename),
                                    os.path.join(output_path, person_name, image_filename))

    def _align_person_image(self, input_path, output_path, person_name, image_filename):
        output_size = (ALIGNED_IMAGE_SIZE[1], ALIGNED_IMAGE_SIZE[0])
        landmarks, theoretical_landmark, _ = self._get_landmarks(input_path, person_name, image_filename)

        cv2_image = cv2.imread(os.path.join(input_path, person_name, image_filename))
        cv2_transform = cv2.getAffineTransform(landmarks.astype(np.float32),
                                               (theoretical_landmark * np.array(output_size)).astype(np.float32))
        height = cv2_image.shape[0]
        width = cv2_image.shape[1]
        theta = cv2_transform_to_theta(cv2_transform, height, width, ALIGNED_IMAGE_SIZE[0], ALIGNED_IMAGE_SIZE[1])
        grid = F.affine_grid(theta, torch.Size((1, 3, ALIGNED_IMAGE_SIZE[0], ALIGNED_IMAGE_SIZE[1])))

        torch_image = torch.from_numpy(cv2_image).permute(2, 0, 1).unsqueeze(0).float()
        torch_aligned_image = F.grid_sample(torch_image, grid, mode='bilinear').squeeze(0)
        cv2_aligned_image = torch_aligned_image.permute(1, 2, 0).numpy()
        cv2.imwrite(os.path.join(output_path, person_name, image_filename), cv2_aligned_image)

    def _get_landmarks(self, input_path, person_name, image_filename):
        with torch.no_grad():
            image = Image.open(os.path.join(input_path, person_name, image_filename)).convert('RGB')
            pose_estimator_image = self._pose_estimator_image_transform(image)
            pose_heatmaps = self._pose_estimator_model(pose_estimator_image.unsqueeze(0).to(self._device))
            heatmap_coordinates, presence = get_coordinates(pose_heatmaps)

            scaled_coordinates = np.zeros((heatmap_coordinates.size()[1], 2))

            for i in range(heatmap_coordinates.size()[1]):
                scaled_coordinates[i, 0] = heatmap_coordinates[0, i, 0] / pose_heatmaps.size()[3] * image.width
                scaled_coordinates[i, 1] = heatmap_coordinates[0, i, 1] / pose_heatmaps.size()[2] * image.height

        return get_landmarks_from_pose(scaled_coordinates, presence[0].cpu().numpy(),
                                       self._presence_threshold, self._ignore_presence_threshold_for_nose_eyes)


def get_landmarks_from_pose(pose, presence, presence_threshold, ignore_presence_threshold_for_nose_eyes=False):
    if (presence[0:5] > presence_threshold).all():
        eyes_center = (pose[1] + pose[2]) / 2
        ears_center = (pose[3] + pose[4]) / 2

        landmarks = np.zeros((3, 2))
        landmarks[0] = 2 * pose[0] - eyes_center + ears_center - eyes_center
        landmarks[1] = pose[3] - np.array([0, ears_center[1] - eyes_center[1]])
        landmarks[2] = pose[4] - np.array([0, ears_center[1] - eyes_center[1]])

        theoretical_landmarks = np.array([[0.5, 0.75],
                                         [0.9, 0.25],
                                         [0.1, 0.25]])
        return landmarks, theoretical_landmarks, 5
    elif (presence[0:3] > presence_threshold).all() and presence[3] > presence_threshold:
        landmarks = np.zeros((3, 2))
        landmarks[0] = pose[0]
        landmarks[1] = np.array([pose[3, 0], pose[1, 1]])
        landmarks[2] = pose[2]

        eye_ear_x_diff = landmarks[1, 0] - landmarks[2, 0]
        eye_nose_x_diff =  landmarks[0, 0] - landmarks[2, 0]

        theoretical_landmarks = np.array([[0.25 + 0.6 * eye_nose_x_diff / eye_ear_x_diff, 0.45],
                                         [0.85, 0.25],
                                         [0.25, 0.25]])

        return landmarks, theoretical_landmarks, 4

    elif (presence[0:3] > presence_threshold).all() and presence[4] > presence_threshold:
        landmarks = np.zeros((3, 2))
        landmarks[0] = pose[0]
        landmarks[1] = pose[1]
        landmarks[2] = np.array([pose[4, 0], pose[1, 1]])

        eye_ear_x_diff = landmarks[1, 0] - landmarks[2, 0]
        eye_nose_x_diff = landmarks[0, 0] - landmarks[2, 0]

        theoretical_landmarks = np.array([[0.15 + 0.6 * eye_nose_x_diff / eye_ear_x_diff, 0.45],
                                         [0.75, 0.25],
                                         [0.15, 0.25]])

        return landmarks, theoretical_landmarks, 4

    elif (presence[0:3] > presence_threshold).all() or ignore_presence_threshold_for_nose_eyes:
        eyes_x_diff = pose[1, 0] - pose[2, 0]
        eye_nose_x_diff =  pose[0, 0] - pose[2, 0]

        theoretical_landmarks = np.array([[0.35 + 0.3 * eye_nose_x_diff / eyes_x_diff, 0.5],
                                         [0.7, 0.35],
                                         [0.3, 0.35]])

        return pose[0:3], theoretical_landmarks, 3
    else:
        raise ValueError('The alignment is not possible')


def cv2_transform_to_theta(transform, source_height, source_width, destination_height, destination_width):
    transform = np.vstack([transform, [[0, 0, 1]]])
    A = inv_3x3(_get_normalization_matrix(source_height, source_width))
    B = _get_normalization_matrix(destination_height, destination_width)

    normalized_transform = np.dot(B, np.dot(transform, A))
    theta = inv_3x3(normalized_transform)

    return torch.from_numpy(theta[:2, :]).unsqueeze(0).float()


def _get_normalization_matrix(height, width, eps=1e-14):
    return np.array([[2.0 / (width - 1 + eps), 0.0, -1.0], [0.0, 2.0 / (height - 1 + eps), -1.0], [0.0, 0.0, 1.0]])


# Faster than np.linalg.inv for 3x3 matrix (https://github.com/numpy/numpy/issues/17166)
def inv_3x3(x):
    det = det_3x3(x)
    if abs(det) < 1e-6 or not math.isfinite(det):
        raise np.linalg.LinAlgError()

    adj = np.zeros_like(x)
    x00, x01, x02, x10, x11, x12, x20, x21, x22 = x.ravel()

    adj[0, 0] = x11 * x22 - x12 * x21
    adj[0, 1] = x02 * x21 - x01 * x22
    adj[0, 2] = x01 * x12 - x02 * x11
    adj[1, 0] = x12 * x20 - x10 * x22
    adj[1, 1] = x00 * x22 - x02 * x20
    adj[1, 2] = x02 * x10 - x00 * x12
    adj[2, 0] = x10 * x21 - x11 * x20
    adj[2, 1] = x01 * x20 - x00 * x21
    adj[2, 2] = x00 * x11 - x01 * x10

    return adj / det


# Faster than np.linalg.det for 3x3 matrix (https://github.com/numpy/numpy/issues/17166)
def det_3x3(x):
    assert len(x.shape) == 2 and x.shape[0] == 3 and x.shape[1] == 3

    x00, x01, x02, x10, x11, x12, x20, x21, x22 = x.ravel()
    return x00 * (x11 * x22 - x21 * x12) - \
           x01 * (x10 * x22 - x20 * x12) + \
           x02 * (x10 * x21 - x20 * x11)


def main():
    parser = argparse.ArgumentParser(description='Align faces')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--pose_estimator_backbone_type', choices=BACKBONE_TYPES,
                        help='Choose the pose estimator backbone type', required=True)
    parser.add_argument('--pose_estimator_model_checkpoint', type=str,
                        help='Choose the pose estimator model checkpoint file', required=True)
    parser.add_argument('--presence_threshold', type=float, help='Choose the presence threshold', required=True)
    parser.add_argument('--ignore_presence_threshold_for_nose_eyes', action='store_true',
                        help='Ignore the presence threshold for nose and eyes keypoint')

    parser.add_argument('--input', type=str, help='Choose the input path', required=True)
    parser.add_argument('--output', type=str, help='Choose the output path', required=True)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    pose_estimator_model = create_model(args.pose_estimator_backbone_type)
    load_checkpoint(pose_estimator_model, args.pose_estimator_model_checkpoint)
    pose_estimator_model.eval()

    aligner = FolderFaceAligner(pose_estimator_model, device,
                                args.presence_threshold, args.ignore_presence_threshold_for_nose_eyes)
    aligner.align(args.input, args.output)


if __name__ == '__main__':
    main()
