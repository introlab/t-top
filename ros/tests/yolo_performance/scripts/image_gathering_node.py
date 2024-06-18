#!/usr/bin/env python3

import os
import uuid

import cv2

import rclpy
import rclpy.node

from cv_bridge import CvBridge

from sensor_msgs.msg import Image


MAX_WINDOW_SIZE = 600
ENTER_KEY_CODE = 13


class ImageGatheringNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('image_gathering_node')

        self._output_path = os.path.expanduser(self.declare_parameter('output_path', '').get_parameter_value().string_value)
        self._image_count = self.declare_parameter('image_count', 100).get_parameter_value().integer_value
        self._setups = self.declare_parameter('setups', []).get_parameter_value().string_array_value
        self._classes = self.declare_parameter('classes', []).get_parameter_value().string_array_value

        self._current_image_index = 0
        self._current_class_index = 0
        self._current_setup_index = 0
        self._is_waiting = True
        self._print_current_class()

        self._create_output_directories()

        self._cv_bridge = CvBridge()
        self._image_sub = self.create_subscription(Image, 'image_raw', self._image_cb, 1)

    def _create_output_directories(self):
        for s in self._setups:
            for c in self._classes:
                directory_path = os.path.join(self._output_path, s, c)
                os.makedirs(directory_path, exist_ok=True)

    def _image_cb(self, color_image_msg):
        color_image = self._cv_bridge.imgmsg_to_cv2(color_image_msg, 'bgr8')
        pressed_key = self._display_image(color_image)

        if self._is_waiting and pressed_key == ENTER_KEY_CODE:
            self._is_waiting = False
            print('Gathering images')
        elif not self._is_waiting and self._current_image_index < self._image_count:
            self._save_image(color_image)
            self._current_image_index += 1
        elif not self._is_waiting and self._current_class_index < len(self._classes):
            self._current_image_index = 0
            self._current_setup_index += 1
            if self._current_setup_index >= len(self._setups):
                self._current_setup_index = 0
                self._current_class_index += 1
            self._is_waiting = True

            if self._current_class_index < len(self._classes):
                self._print_current_class()
            else:
                print('The gathering is finished.')
                rclpy.shutdown()

    def _display_image(self, color_image):
        scale = min(MAX_WINDOW_SIZE / color_image.shape[0], MAX_WINDOW_SIZE / color_image.shape[1])
        cv2.imshow("camera", cv2.resize(color_image, (int(scale * color_image.shape[1]), int(scale * color_image.shape[0]))))
        return cv2.waitKey(1)

    def _save_image(self, color_image):
        path = os.path.join(self._output_path,
                            self._setups[self._current_setup_index],
                            self._classes[self._current_class_index],
                            f'{uuid.uuid4()}.jpg')
        cv2.imwrite(path, color_image)

    def _print_current_class(self):
        print()
        print(f'Waiting for {self._classes[self._current_class_index]} - {self._setups[self._current_setup_index]}')
        print(f'Make sure only the object and at most one person are visible.')
        print(f'Press enter to continue')

    def run(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    image_gathering_node = ImageGatheringNode()

    try:
        image_gathering_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        image_gathering_node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
