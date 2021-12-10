import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


class StewartForwardKinematics:  # TODO use an analytic forward kinematics model
    def __init__(self, inverse_kinematics, initial_z):
        self._inverse_kinematics = inverse_kinematics
        self._initial_z = initial_z

        self._model = StewartForwardKinematicsModel()

        project_path = os.path.dirname(os.path.realpath(__file__))
        state_dict = torch.load(os.path.join(project_path, '..', 'stewart_forward_kinematics_model.pth'))
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def calculate_pose(self, servo_angles):
        pose = self._model(torch.tensor([servo_angles]))

        return np.array([pose[0, 0].item(), pose[0, 1].item(), pose[0, 2].item()]), \
               Rotation.from_quat([pose[0, 3].item(), pose[0, 4].item(), pose[0, 5].item(), pose[0, 6].item()])

    def export_layers(self):
        return self._model.position_model.layers, self._model.orientation_model.layers


class StewartForwardKinematicsModel(nn.Module):
    def __init__(self):
        super(StewartForwardKinematicsModel, self).__init__()

        self.position_model = StewartForwardKinematicsPositionModel()
        self.orientation_model = StewartForwardKinematicsOrientationModel()

    def forward(self, servo_angles):
        position = self.position_model(servo_angles)
        orientation = self.orientation_model(servo_angles)

        return torch.cat([position, orientation], dim=1)


class StewartForwardKinematicsPositionModel(nn.Module):
    def __init__(self):
        super(StewartForwardKinematicsPositionModel, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features=6, out_features=64))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Linear(in_features=64, out_features=32))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Linear(in_features=32, out_features=16))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Linear(in_features=16, out_features=3))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class StewartForwardKinematicsOrientationModel(nn.Module):
    def __init__(self):
        super(StewartForwardKinematicsOrientationModel, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features=6, out_features=64))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Linear(in_features=64, out_features=32))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Linear(in_features=32, out_features=16))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Linear(in_features=16, out_features=4))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = F.normalize(x, dim=1, p=2.0)

        return x
