#!/usr/bin/env python3

import traceback

import numpy as np
import cv2

import torch

import rospy
import message_filters
from cv_bridge import CvBridge

from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from video_analyzer.msg import VideoAnalysis, VideoAnalysisObject

from dnn_utils import DescriptorYoloV4, YoloV4, PoseEstimator, FaceDescriptorExtractor
import hbba_lite

MM_TO_M = 0.001


class VideoAnalyzerNode:
    def __init__(self):
        self._use_descriptor_yolo_v4 = rospy.get_param('~use_descriptor_yolo_v4')
        self._confidence_threshold = rospy.get_param('~confidence_threshold')
        self._nms_threshold = rospy.get_param('~nms_threshold')
        self._person_probability_threshold = rospy.get_param('~person_probability_threshold')
        self._pose_confidence_threshold = rospy.get_param('~pose_confidence_threshold')
        self._inference_type = rospy.get_param('~inference_type', None)
        self._depth_mean_offset = rospy.get_param('~depth_mean_offset', 1)

        if self._use_descriptor_yolo_v4:
            self._object_detector = DescriptorYoloV4(confidence_threshold=self._confidence_threshold,
                                                     nms_threshold=self._nms_threshold, inference_type=self._inference_type)
        else:
            self._object_detector = YoloV4(confidence_threshold=self._confidence_threshold,
                                           nms_threshold=self._nms_threshold, inference_type=self._inference_type)
        self._object_class_names = self._object_detector.get_class_names()

        self._pose_estimator = PoseEstimator(inference_type=self._inference_type)
        self._face_descriptor_extractor = FaceDescriptorExtractor(inference_type=self._inference_type)

        self._person_class_index = self._object_class_names.index('person')

        self._analysed_image_pub = rospy.Publisher('analysed_image', Image, queue_size=10)
        self._video_analysis_pub = rospy.Publisher('video_analysis', VideoAnalysis, queue_size=10)
        self._video_analysis_seq = 0

        self._cv_bridge = CvBridge()

        self._analysed_image_hbba_filter_state = hbba_lite.OnOffHbbaFilterState('analysed_image/filter_state')
        color_image_sub = message_filters.Subscriber('color_image_raw', Image)
        depth_image_sub = message_filters.Subscriber('depth_image_raw', Image)
        depth_camera_info_sub = message_filters.Subscriber('depth_camera_info', CameraInfo)
        self._image_ts = hbba_lite.ThrottlingHbbaApproximateTimeSynchronizer([color_image_sub, depth_image_sub, depth_camera_info_sub],
                                                                             5, 0.03, self._image_cb, 'image_raw/filter_state')

    def _image_cb(self, color_image_msg, depth_image_msg, depth_camera_info):
        if depth_image_msg.encoding != '16UC1':
            rospy.logerr('Invalid depth image encoding')

        try:
            color_image = self._cv_bridge.imgmsg_to_cv2(color_image_msg, 'rgb8')
            depth_image = self._cv_bridge.imgmsg_to_cv2(depth_image_msg, '16UC1')
            self._analyse(color_image, depth_image, depth_camera_info, color_image_msg.header)
        except Exception as e:
            rospy.logerr('Image analysis error: {} \n {}'.format(e, traceback.format_exc()))

    def _analyse(self, color_image, depth_image, depth_camera_info, header):
        color_image_tensor = self._convert_color_image_to_tensor(color_image)
        predictions = self._object_detector(color_image_tensor)

        person_predictions = []
        object_images = []
        object_corners = []
        for prediction in predictions:
            x0, y0, x1, y1 = self._get_bbox(prediction, color_image.shape[1], color_image.shape[0])
            object_color_image = color_image[y0:y1, x0:x1, :]
            object_color_image_tensor = color_image_tensor[:, y0:y1, x0:x1]
            object_images.append(object_color_image)
            object_corner = Point()
            object_corner.x = x0
            object_corner.y = y0
            object_corners.append(object_corner)
            if prediction.class_index == self._person_class_index and \
                    prediction.class_probabilities[prediction.class_index] > self._person_probability_threshold:
                person_predictions.append(self._analyse_person(object_color_image, object_color_image_tensor, x0, y0))
            else:
                person_predictions.append(None)

        self._publish_video_analysis(predictions, object_images, person_predictions, depth_image, depth_camera_info, header, object_corners)

        if not self._analysed_image_hbba_filter_state.is_filtering_all_messages:
            self._publish_analysed_image(color_image, header, predictions, person_predictions)

    def _convert_color_image_to_tensor(self, color_image):
        return torch.from_numpy(color_image).to(self._object_detector.device()).permute(2, 0, 1).float() / 255

    def _analyse_person(self, cv_color_image, color_image_tensor, x0, y0):
        pose_coordinates, pose_confidence = self._pose_estimator(color_image_tensor)
        pose_image = cv_color_image.copy()
        self._draw_person_pose(pose_image, pose_coordinates, pose_confidence)

        try:
            face_descriptor, face_image = self._face_descriptor_extractor(color_image_tensor, pose_coordinates, pose_confidence)
        except ValueError:
            face_descriptor = torch.tensor([])
            face_image = np.zeros((1, 1, 3), np.uint8)

        pose_coordinates[:, 0] += x0
        pose_coordinates[:, 1] += y0

        return pose_coordinates.tolist(), pose_confidence.tolist(), pose_image, face_descriptor.tolist(), face_image

    def _publish_video_analysis(self, predictions, object_images, person_predictions, depth_image, depth_camera_info, header, object_corners):
        msg = VideoAnalysis()
        msg.header.seq = header.seq
        msg.header.stamp = header.stamp
        msg.header.frame_id = depth_camera_info.header.frame_id

        for i in range(len(predictions)):
            o = self._convert_prediction_to_video_analysis_object(predictions[i], object_images[i], person_predictions[i],
                                                                  depth_image, depth_camera_info, object_corners[i])
            msg.objects.append(o)

        self._video_analysis_pub.publish(msg)

    def _convert_prediction_to_video_analysis_object(self, prediction, object_image, person_prediction,
                                                     depth_image, depth_camera_info, object_corner):
        o = VideoAnalysisObject()
        o.center = self._project_2d_to_3d(prediction.center_x, prediction.center_y,
                                          depth_image, depth_camera_info)

        o.object_class = self._object_class_names[prediction.class_index]
        o.object_corner = object_corner
        o.object_confidence = prediction.confidence
        o.object_image = self._cv_bridge.cv2_to_imgmsg(object_image, encoding='rgb8')

        if self._use_descriptor_yolo_v4:
            o.object_descriptor = prediction.descriptor

        if person_prediction is not None:
            pose_coordinates, pose_confidence, pose_image, face_descriptor, face_image = person_prediction
            o.person_pose = self._convert_pose_coordinates(pose_coordinates, depth_image, depth_camera_info)
            o.person_pose_confidence = pose_confidence
            o.person_pose_image = self._cv_bridge.cv2_to_imgmsg(pose_image, encoding='rgb8')

            o.face_descriptor = face_descriptor
            o.face_image = self._cv_bridge.cv2_to_imgmsg(face_image, encoding='rgb8')

        return o

    def _convert_pose_coordinates(self, pose_coordinates, depth_image, depth_camera_info):
        points = []
        for pose_coordinate in pose_coordinates:
            point = self._project_2d_to_3d(pose_coordinate[0], pose_coordinate[1], depth_image, depth_camera_info)
            points.append(point)
        return points

    def _project_2d_to_3d(self, x, y, depth_image, depth_camera_info):
        x0 = int(x - self._depth_mean_offset)
        y0 = int(y - self._depth_mean_offset)
        x1 = int(x + self._depth_mean_offset)
        y1 = int(y + self._depth_mean_offset)
        depth = depth_image[y0:y1, x0:x1].mean()
        K = depth_camera_info.K

        point = Point()
        point.x = (x - K[2]) * depth / K[0] * MM_TO_M
        point.y = (y - K[5]) * depth / K[4] * MM_TO_M
        point.z = depth * MM_TO_M
        return point

    def _publish_analysed_image(self, color_image, header, predictions, person_predictions):
        for prediction in predictions:
            self._draw_prediction(color_image, prediction)

        for prediction in person_predictions:
            if prediction == None:
                continue
            self._draw_person_pose(color_image, prediction[0], prediction[1])

        msg = self._cv_bridge.cv2_to_imgmsg(color_image, 'rgb8')
        msg.header = header
        self._analysed_image_pub.publish(msg)

    def _draw_prediction(self, image, prediction):
        x0, y0, x1, y1 = self._get_bbox(prediction, image.shape[1], image.shape[0])
        color = (255, 0, 0)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=4)

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
                cv2.circle(image, (x, y), 10, (0, 255, 0), thickness=cv2.FILLED)

        for pair in self._pose_estimator.get_skeleton_pairs():
            if pose_confidence[pair[0]] >= self._pose_confidence_threshold and \
                    pose_confidence[pair[1]] >= self._pose_confidence_threshold:
                x0 = int(pose_coordinates[pair[0]][0])
                y0 = int(pose_coordinates[pair[0]][1])
                x1 = int(pose_coordinates[pair[1]][0])
                y1 = int(pose_coordinates[pair[1]][1])
                cv2.line(image, (x0, y0), (x1, y1), (255, 255, 0), thickness=4)

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('video_analyzer_node')
    video_analyzer_node = VideoAnalyzerNode()
    video_analyzer_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
