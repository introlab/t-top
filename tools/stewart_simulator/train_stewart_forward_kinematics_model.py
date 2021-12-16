import os

import numpy as np
from scipy.spatial.transform import Rotation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from state.stewart_forward_kinematics import StewartForwardKinematicsModel

PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))

VALIDATION_SPLIT_RATIO = 0.1
EPOCH_COUNT = 200
BATCH_SIZE = 128


class StewartForwardKinematicsDataset(Dataset):
    def __init__(self):
        dataset_path = os.path.join(PROJECT_PATH, 'stewart_forward_kinematics_dataset')

        servo_angles = np.loadtxt(os.path.join(dataset_path, 'servo_angles.txt'), dtype=np.float32)
        self._servo_angles = torch.from_numpy(servo_angles)

        poses = np.loadtxt(os.path.join(dataset_path, 'poses.txt'), dtype=np.float32)
        self._poses = torch.from_numpy(poses)

    def __len__(self):
        return self._servo_angles.size(0)

    def __getitem__(self, index):
        return self._servo_angles[index], self._poses[index]


def main():
    dataset = StewartForwardKinematicsDataset()
    validation_dataset_size = int(len(dataset) * VALIDATION_SPLIT_RATIO)
    training_dataset_size = len(dataset) - validation_dataset_size
    training_dataset, validation_dataset = random_split(dataset, [training_dataset_size, validation_dataset_size])

    training_dataset_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataset_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = StewartForwardKinematicsModel()

    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCH_COUNT)

    best_validation_position_loss = float('inf')
    best_validation_orientation_loss = float('inf')

    for epoch in range(EPOCH_COUNT):

        model.train()
        training_loss = 0.0
        for servo_angles, poses in training_dataset_loader:
            outputs = model(servo_angles)
            loss = criterion(outputs, poses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        model.eval()
        validation_position_loss = 0.0
        validation_orientation_loss = 0.0
        for servo_angles, poses in validation_dataset_loader:
            outputs = model(servo_angles)
            validation_position_loss += criterion(outputs[:, :3], poses[:, :3]).item()
            validation_orientation_loss += criterion(outputs[:, 3:], poses[:, 3:]).item()

        print(f'Epoch: {epoch + 1}/{EPOCH_COUNT}')
        print(f'training_loss={training_loss / len(training_dataset)}')
        print(f'validation_position_loss={validation_position_loss / len(validation_dataset)}')
        print(f'validation_orientation_loss={validation_orientation_loss / len(validation_dataset)}')
        print()

        if validation_position_loss < best_validation_position_loss:
            best_validation_position_loss = validation_position_loss
            best_validation_orientation_loss = validation_orientation_loss
            torch.save(model.state_dict(), os.path.join(PROJECT_PATH, 'stewart_forward_kinematics_model.pth'))

        scheduler.step()

    print(f'best_validation_position_loss={best_validation_position_loss / len(validation_dataset)}')
    print(f'best_validation_orientation_loss={best_validation_orientation_loss / len(validation_dataset)}')

    model.load_state_dict(torch.load(os.path.join(PROJECT_PATH, 'stewart_forward_kinematics_model.pth')))
    print_covariance_matrix(model, validation_dataset)


def print_covariance_matrix(model, dataset):
    observations = np.zeros((len(dataset), 6))

    for i in range(len(dataset)):
        servo_angles, pose = dataset[i]
        position = pose[:3].detach().numpy()
        orientation = Rotation.from_quat(pose[3:].detach().numpy()).as_euler('xyz')

        estimated_pose = model(servo_angles.unsqueeze(0))[0]
        estimated_position = estimated_pose[:3].detach().numpy()
        estimated_orientation = Rotation.from_quat(estimated_pose[3:].detach().numpy()).as_euler('xyz')

        observations[i, :3] = estimated_position - position
        observations[i, 3:] = estimated_orientation - orientation

    print('Covariance matrix:')
    print(np.cov(observations, rowvar=False))

if __name__ == '__main__':
    main()
