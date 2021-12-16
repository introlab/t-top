import os
import argparse

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from common.modules import load_checkpoint

from pose_estimation.pose_estimator import get_coordinates
from pose_estimation.trainers.pose_estimator_trainer import IMAGE_SIZE as POSE_ESTIMATOR_IMAGE_SIZE

from train_pose_estimator import create_model

ALIGNED_IMAGE_SIZE = (128, 96)
PRESENCE_THRESHOLD = 0.4


class FolderFaceAligner:
    def __init__(self, pose_estimator_model):
        self._pose_estimator_model = pose_estimator_model
        self._pose_estimator_image_transform = transforms.Compose([
            transforms.Resize(POSE_ESTIMATOR_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def align_lfw(self, input_path, output_path):
        person_names = [o for o in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, o))]
        for person_name in person_names:
            print('Processing {} images'.format(person_name))
            self._align_person_images(input_path, output_path, person_name)

    def _align_person_images(self, input_path, output_path, person_name):
        os.makedirs(os.path.join(output_path, person_name), exist_ok=True)

        image_filenames = [o for o in os.listdir(os.path.join(input_path, person_name))]
        for image_filename in image_filenames:
            try:
                self._align_person_image(input_path, output_path, person_name, image_filename)
            except (ValueError, np.linalg.LinAlgError):
                print('Warning: the alignment is impossible ({})'.format(image_filename))

    def _align_person_image(self, input_path, output_path, person_name, image_filename):
        output_size = (ALIGNED_IMAGE_SIZE[1], ALIGNED_IMAGE_SIZE[0])
        landmarks, theoretical_landmark = self._get_landmarks(input_path, person_name, image_filename)

        cv2_image = cv2.imread(os.path.join(input_path, person_name, image_filename))
        cv2_transform = cv2.getAffineTransform(landmarks.astype(np.float32),
                                               (theoretical_landmark * np.array(output_size)).astype(np.float32))
        height = cv2_image.shape[0]
        width = cv2_image.shape[1]
        theta = cv2_transform_to_theta(cv2_transform, height, width, ALIGNED_IMAGE_SIZE[0], ALIGNED_IMAGE_SIZE[1])
        grid = F.affine_grid(theta, torch.Size((1, 3, ALIGNED_IMAGE_SIZE[0], ALIGNED_IMAGE_SIZE[1])))

        torch_image = torch.from_numpy(cv2_image).permute(2, 0, 1).unsqueeze(0).float()
        torch_aligned_image = F.grid_sample(torch_image, grid, mode='nearest').squeeze(0)
        cv2_aligned_image = torch_aligned_image.permute(1, 2, 0).numpy()
        cv2.imwrite(os.path.join(output_path, person_name, image_filename), cv2_aligned_image)

    def _get_landmarks(self, input_path, person_name, image_filename):
        image = Image.open(os.path.join(input_path, person_name, image_filename)).convert('RGB')
        pose_estimator_image = self._pose_estimator_image_transform(image)
        pose_heatmaps = self._pose_estimator_model(pose_estimator_image.unsqueeze(0))
        heatmap_coordinates, presence = get_coordinates(pose_heatmaps)

        scaled_coordinates = np.zeros((heatmap_coordinates.size()[1], 2))

        for i in range(heatmap_coordinates.size()[1]):
            scaled_coordinates[i, 0] = heatmap_coordinates[0, i, 0] / pose_heatmaps.size()[3] * image.width
            scaled_coordinates[i, 1] = heatmap_coordinates[0, i, 1] / pose_heatmaps.size()[2] * image.height

        return get_landmarks_from_pose(scaled_coordinates, presence[0])


def get_landmarks_from_pose(pose, presence):
    if (presence[0:5] > PRESENCE_THRESHOLD).all():
        eyes_center = (pose[1] + pose[2]) / 2
        hears_center = (pose[3] + pose[4]) / 2

        landmarks = np.zeros((3, 2))
        landmarks[0] = 2 * pose[0] - eyes_center + hears_center - eyes_center
        landmarks[1] = pose[3] - np.array([0, hears_center[1] - eyes_center[1]])
        landmarks[2] = pose[4] - np.array([0, hears_center[1] - eyes_center[1]])

        theoretical_landmark = np.array([[0.5, 0.75],
                                         [0.9, 0.25],
                                         [0.1, 0.25]])
        return landmarks, theoretical_landmark
    elif (presence[0:3] > PRESENCE_THRESHOLD).all() and presence[3] > PRESENCE_THRESHOLD:
        landmarks = np.zeros((3, 2))
        landmarks[0] = pose[0]
        landmarks[1] = np.array([pose[3, 0], pose[1, 1]])
        landmarks[2] = pose[2]

        theoretical_landmark = np.array([[0.25, 0.5],
                                         [0.9, 0.25],
                                         [0.25, 0.25]])

        return landmarks, theoretical_landmark

    elif (presence[0:3] > PRESENCE_THRESHOLD).all() and presence[4] > PRESENCE_THRESHOLD:
        landmarks = np.zeros((3, 2))
        landmarks[0] = pose[0]
        landmarks[1] = pose[1]
        landmarks[2] = np.array([pose[4, 0], pose[1, 1]])

        theoretical_landmark = np.array([[0.75, 0.5],
                                         [0.75, 0.25],
                                         [0.1, 0.25]])

        return landmarks, theoretical_landmark

    elif (presence[0:3] > PRESENCE_THRESHOLD).all():
        theoretical_landmark = np.array([[0.5, 0.5144414],
                                         [0.75, 0.25],
                                         [0.25, 0.25]])

        return pose[0:3], theoretical_landmark
    else:
        raise ValueError('The aligment is not possible')


def cv2_transform_to_theta(transform, source_height, source_width, destination_height, destination_width):
    transform = np.vstack([transform, [[0, 0, 1]]])
    A = np.linalg.inv(_get_normalization_matrix(source_height, source_width))
    B = _get_normalization_matrix(destination_height, destination_width)

    normalized_transform = np.dot(B, np.dot(transform, A))
    theta = np.linalg.inv(normalized_transform)

    return torch.from_numpy(theta[:2, :]).unsqueeze(0).float()


def _get_normalization_matrix(height, width, eps=1e-14):
    return np.array([[2.0 / (width - 1 + eps), 0.0, -1.0], [0.0, 2.0 / (height - 1 + eps), -1.0], [0.0, 0.0, 1.0]])


def main():
    parser = argparse.ArgumentParser(description='Align LFW faces')
    parser.add_argument('--pose_estimator_backbone_type',
                        choices=['mnasnet0.5', 'mnasnet1.0', 'resnet18', 'resnet34', 'resnet50'],
                        help='Choose the pose estimator backbone type', required=True)
    parser.add_argument('--pose_estimator_upsampling_count', type=int,
                        help='Set the pose estimator upsamping layer count', required=True)
    parser.add_argument('--pose_estimator_model_checkpoint', type=str,
                        help='Choose the pose estimator model checkpoint file', required=True)

    parser.add_argument('--input', type=str, help='Choose the input path', required=True)
    parser.add_argument('--output', type=str, help='Choose the output path', required=True)

    args = parser.parse_args()

    pose_estimator_model = create_model(args.pose_estimator_backbone_type, args.pose_estimator_upsampling_count)
    load_checkpoint(pose_estimator_model, args.pose_estimator_model_checkpoint)
    pose_estimator_model.eval()

    aligner = FolderFaceAligner(pose_estimator_model)
    aligner.align_lfw(args.input, args.output)


if __name__ == '__main__':
    main()
