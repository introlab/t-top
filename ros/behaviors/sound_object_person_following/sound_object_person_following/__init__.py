import os
import json

CAMERA_3D_CALIBRATION_FILE_PATH = os.path.join(os.environ['HOME'], '.ros', 't-top', 'sound_object_person_following', 'camera_3d_calibration.json')


class Camera3dCalibration:
    def __init__(self, center_x, center_y, width, height):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.half_width = width / 2
        self.half_height = height / 2

    @staticmethod
    def load_json_file():
        if not os.path.exists(CAMERA_3D_CALIBRATION_FILE_PATH):
            raise FileNotFoundError('camera_3d_calibration.json is missing.')

        with open(CAMERA_3D_CALIBRATION_FILE_PATH, 'r') as file:
            data = json.load(file)
            return Camera3dCalibration(data['center_x'], data['center_y'], data['width'], data['height'])

    def save_json_file(self):
        os.makedirs(os.path.dirname(CAMERA_3D_CALIBRATION_FILE_PATH), exist_ok=True)
        with open(CAMERA_3D_CALIBRATION_FILE_PATH, 'w') as file:
            data = {
                'center_x': self.center_x,
                'center_y': self.center_y,
                'width': self.width,
                'height': self.height
            }
            json.dump(data, file)
