#!/usr/bin/env python3

import numpy as np
import cv2

import torch

import rospy
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from video_analyzer.msg import VideoAnalysis

from dnn_utils import DescriptorYoloV4, YoloV4, PoseEstimator, FaceDescriptorExtractor
import hbba_lite


PERSON_POSE_KEYPOINT_COLORS = [(0, 255, 0),
                               (255, 0, 0),
                               (0, 0, 255),
                               (255, 0, 0),
                               (0, 0, 255),
                               (255, 0, 0),
                               (0, 0, 255),
                               (255, 0, 0),
                               (0, 0, 255),
                               (255, 0, 0),
                               (0, 0, 255),
                               (255, 0, 0),
                               (0, 0, 255),
                               (255, 0, 0),
                               (0, 0, 255),
                               (255, 0, 0),
                               (0, 0, 255)]


class ObjectAnalysis:
    def __init__(self, center_x, center_y, width, height,
                 object_class, object_confidence,
                 object_descriptor=None, object_image=None,
                 pose_analysis=None, face_analysis=None):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.object_class = object_class
        self.object_confidence = object_confidence
        self.object_descriptor = object_descriptor
        self.object_image = object_image

        self.pose_analysis = pose_analysis
        self.face_analysis = face_analysis

    @staticmethod
    def from_yoloV4_prediction(prediction, object_class_names):
        return ObjectAnalysis(prediction.center_x, prediction.center_y, prediction.width, prediction.height,
                              object_class_names[prediction.class_index], prediction.confidence,
                              prediction.descriptor)

class PoseAnalysis:
    def __init__(self, pose_coordinates, pose_confidence, pose_image):
        self.pose_coordinates = pose_coordinates
        self.pose_confidence = pose_confidence
        self.pose_image = pose_image


class FaceAnalysis:
    def __init__(self, descriptor, face_image=None):
        self.descriptor = descriptor
        self.face_image = face_image


class VideoAnalyzerNode:
    def __init__(self):
        self._use_descriptor_yolo_v4 = rospy.get_param('~use_descriptor_yolo_v4')
        self._confidence_threshold = rospy.get_param('~confidence_threshold')
        self._nms_threshold = rospy.get_param('~nms_threshold')
        self._person_probability_threshold = rospy.get_param('~person_probability_threshold')
        self._pose_confidence_threshold = rospy.get_param('~pose_confidence_threshold')
        self._inference_type = rospy.get_param('~inference_type', None)

        self._pose_enabled = rospy.get_param('~pose_enabled', True)
        self._face_descriptor_enabled = rospy.get_param('~face_descriptor_enabled', True)
        self._cropped_image_enabled = rospy.get_param('~cropped_image_enabled', True)

        if self._face_descriptor_enabled and not self._pose_enabled:
            raise ValueError('The pose estimation must be enabled when the face descriptor extraction is enabled.')

        if self._use_descriptor_yolo_v4:
            self._object_detector = DescriptorYoloV4(confidence_threshold=self._confidence_threshold,
                                                     nms_threshold=self._nms_threshold, inference_type=self._inference_type)
        else:
            self._object_detector = YoloV4(confidence_threshold=self._confidence_threshold,
                                           nms_threshold=self._nms_threshold, inference_type=self._inference_type)
        self._object_class_names = self._object_detector.get_class_names()

        if self._pose_enabled:
            self._pose_estimator = PoseEstimator(inference_type=self._inference_type)
            self._skeleton_pairs = self._pose_estimator.get_skeleton_pairs()
        if self._face_descriptor_enabled:
            self._face_descriptor_extractor = FaceDescriptorExtractor(inference_type=self._inference_type)

        self._person_class_index = self._object_class_names.index('person')

        self._video_analysis_pub = rospy.Publisher('video_analysis', VideoAnalysis, queue_size=10)
        self._analysed_image_pub = hbba_lite.OnOffHbbaPublisher('analysed_image', Image, queue_size=10)

        self._cv_bridge = CvBridge()

    def _analyse(self, color_image):
        color_image_tensor = self._convert_color_image_to_tensor(color_image)
        predictions = self._object_detector(color_image_tensor)

        object_analyses = []
        for prediction in predictions:
            object_analysis = ObjectAnalysis.from_yoloV4_prediction(prediction, self._object_class_names)

            x0, y0, x1, y1 = self._get_bbox(prediction, color_image.shape[1], color_image.shape[0])
            object_color_image = color_image[y0:y1, x0:x1, :]
            object_color_image_tensor = color_image_tensor[:, y0:y1, x0:x1]

            if self._cropped_image_enabled:
                object_analysis.object_image = object_color_image
            if self._pose_enabled and prediction.class_index == self._person_class_index and \
                    prediction.class_probabilities[prediction.class_index] > self._person_probability_threshold:
                object_analysis.pose_analysis, object_analysis.face_analysis = \
                    self._analyse_person(object_color_image, object_color_image_tensor, x0, y0)

            object_analyses.append(object_analysis)

        return object_analyses

    def _convert_color_image_to_tensor(self, color_image):
        return torch.from_numpy(color_image).to(self._object_detector.device()).permute(2, 0, 1).float() / 255

    def _analyse_person(self, cv_color_image, color_image_tensor, x0, y0):
        pose_coordinates, pose_confidence = self._pose_estimator(color_image_tensor)

        face_analysis = None
        if self._face_descriptor_enabled:
            try:
                face_descriptor, face_image = self._face_descriptor_extractor(color_image_tensor,
                                                                              pose_coordinates, pose_confidence)
            except ValueError:
                face_descriptor = torch.tensor([])
                face_image = None

            face_analysis = FaceAnalysis(face_descriptor.tolist())
            if self._cropped_image_enabled:
                face_analysis.face_image = face_image

        pose_image = None
        if self._cropped_image_enabled:
            pose_image = cv_color_image.copy()
            self._draw_person_pose(pose_image, pose_coordinates, pose_confidence)

        pose_coordinates[:, 0] += x0
        pose_coordinates[:, 1] += y0
        pose_analysis = PoseAnalysis(pose_coordinates.tolist(), pose_confidence.tolist(), pose_image)

        return pose_analysis, face_analysis

    def _publish_analysed_image(self, color_image, header, object_analyses):
        if self._analysed_image_pub.is_filtering_all_messages:
            return

        for object_analysis in object_analyses:
            self._draw_object_analysis(color_image, object_analysis)

        msg = self._cv_bridge.cv2_to_imgmsg(color_image, 'rgb8')
        msg.header = header
        self._analysed_image_pub.publish(msg)

    def _draw_object_analysis(self, image, object_analysis):
        x0, y0, x1, y1 = self._get_bbox(object_analysis, image.shape[1], image.shape[0])
        color = (255, 0, 0)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=4)

        if object_analysis.pose_analysis is not None:
            self._draw_person_pose(image,
                                   object_analysis.pose_analysis.pose_coordinates,
                                   object_analysis.pose_analysis.pose_confidence)

    def _get_bbox(self, prediction, width, height):
        x0 = np.clip(int(prediction.center_x - prediction.width / 2), 0, width)
        y0 = np.clip(int(prediction.center_y - prediction.height / 2), 0, height)
        x1 = np.clip(int(prediction.center_x + prediction.width / 2), 0, width)
        y1 = np.clip(int(prediction.center_y + prediction.height / 2), 0, height)
        return x0, y0, x1, y1

    def _draw_person_pose(self, image, pose_coordinates, pose_confidence):
        for i in range(len(pose_confidence)):
            if pose_confidence[i] >= self._pose_confidence_threshold:
                x = int(pose_coordinates[i][0])
                y = int(pose_coordinates[i][1])
                cv2.circle(image, (x, y), 10, PERSON_POSE_KEYPOINT_COLORS[i], thickness=cv2.FILLED)

        for pair in self._skeleton_pairs:
            if pose_confidence[pair[0]] >= self._pose_confidence_threshold and \
                    pose_confidence[pair[1]] >= self._pose_confidence_threshold:
                x0 = int(pose_coordinates[pair[0]][0])
                y0 = int(pose_coordinates[pair[0]][1])
                x1 = int(pose_coordinates[pair[1]][0])
                y1 = int(pose_coordinates[pair[1]][1])
                cv2.line(image, (x0, y0), (x1, y1), (255, 255, 0), thickness=4)

    def run(self):
        rospy.spin()


def convert_pose_coordinates_2d(pose_coordinates, image_width, image_height):
    points = []
    for pose_coordinate in pose_coordinates:
        points.append(Point(x=pose_coordinate[0] / image_width, y=pose_coordinate[1] / image_height))
    return points
