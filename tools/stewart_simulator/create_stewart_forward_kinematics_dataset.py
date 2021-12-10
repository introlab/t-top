import os
import random

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from configuration.global_configuration import GlobalConfiguration
from state.stewart_configuration_preprocessor import StewartConfigurationPreprocessor
from state.stewart_inverse_kinematics import StewartInverseKinematics
from state.stewart_state import TopState, BottomState

COUNT = 1000000
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))


class Sampler:
    def __init__(self):
        configuration = GlobalConfiguration(os.path.join(PROJECT_PATH, 'configuration.json'))
        self.top_state = TopState(configuration.top_configuration)
        self.bottom_state = BottomState(configuration.bottom_configuration)

        self._configuration_preprocessor = StewartConfigurationPreprocessor(configuration,
                                                                            self.top_state.get_initial_anchors(),
                                                                            self.bottom_state.get_initial_anchors())

        self._inverse_kinematics = \
            StewartInverseKinematics(configuration,
                                     self.top_state.get_initial_anchors(),
                                     self._configuration_preprocessor.bottom_linear_actuator_anchors,
                                     self._configuration_preprocessor.bottom_horn_orientation_angles)

        self._initial_z = self._configuration_preprocessor.initial_top_z

    def sample(self):
        while True:
            try:
                xyz_offset_abs = 0.05
                rot_offset_abs = 0.5
                x = random.uniform(-xyz_offset_abs, xyz_offset_abs)
                y = random.uniform(-xyz_offset_abs, xyz_offset_abs)
                z = self._initial_z + random.uniform(-xyz_offset_abs, xyz_offset_abs)

                orientation = Rotation.from_euler('xyz', [random.uniform(-rot_offset_abs, rot_offset_abs),
                                                          random.uniform(-rot_offset_abs, rot_offset_abs),
                                                          random.uniform(-rot_offset_abs, rot_offset_abs)])
                quaternion = orientation.as_quat()
                servo_angles = self._inverse_kinematics.calculate_servo_angles(np.array([x, y, z]), orientation)

                pose = np.array([x, y, z, quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
                return np.array(servo_angles), pose
            except:
                pass


def main():
    sampler = Sampler()

    servo_angles = np.zeros((COUNT, 6))
    poses = np.zeros((COUNT, 7))

    for i in tqdm(range(COUNT)):
        servo_angles[i], poses[i] = sampler.sample()

    output_path = os.path.join(PROJECT_PATH, 'stewart_forward_kinematics_dataset')
    os.makedirs(output_path, exist_ok=True)

    np.savetxt(os.path.join(output_path, 'servo_angles.txt'), servo_angles)
    np.savetxt(os.path.join(output_path, 'poses.txt'), poses)


if __name__ == '__main__':
    main()
