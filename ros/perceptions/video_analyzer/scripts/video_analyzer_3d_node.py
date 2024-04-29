#!/usr/bin/env python3

import traceback
from datetime import datetime
import rospy
import message_filters

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from video_analyzer.msg import VideoAnalysis, VideoAnalysisObject

import hbba_lite

from video_analyzer.lib_video_analyzer_node import VideoAnalyzerNode, convert_pose_coordinates_2d


MM_TO_M = 0.001


class VideoAnalyzer3dNode(VideoAnalyzerNode):
    def __init__(self):
        super().__init__()
        self._depth_mean_offset = rospy.get_param('~depth_mean_offset', 1)

        color_image_sub = message_filters.Subscriber('color_image_raw', Image)
        depth_image_sub = message_filters.Subscriber('depth_image_raw', Image)
        depth_camera_info_sub = message_filters.Subscriber('depth_camera_info', CameraInfo)
        self._image_ts = hbba_lite.ThrottlingHbbaApproximateTimeSynchronizer([color_image_sub, depth_image_sub, depth_camera_info_sub],
                                                                             1, 0.03, self._image_cb, 'image_raw/filter_state')

    def _image_cb(self, color_image_msg, depth_image_msg, depth_camera_info):
        if depth_image_msg.encoding != '16UC1':
            rospy.logerr('Invalid depth image encoding')

        try:
            start_time = datetime.now()
            color_image = self._cv_bridge.imgmsg_to_cv2(color_image_msg, 'rgb8')
            depth_image = self._cv_bridge.imgmsg_to_cv2(depth_image_msg, '16UC1')

            object_analyses, semantic_segmentation = self._analyse(color_image)

            video_analysis_msg = self._analysis_to_msg(object_analyses, semantic_segmentation,
                                                       color_image_msg.header, color_image,
                                                       depth_image, depth_camera_info)
            video_analysis_msg.processing_time_s = (datetime.now() - start_time).total_seconds()
            self._video_analysis_pub.publish(video_analysis_msg)
            self._publish_analysed_image(color_image, color_image_msg.header, object_analyses)
        except Exception as e:
            rospy.logerr(f'Image analysis error: {e} \n {traceback.format_exc()}')

    def _analysis_to_msg(self, object_analyses, semantic_segmentation, header, color_image, depth_image, depth_camera_info):
        image_height, image_width, _ = color_image.shape

        msg = VideoAnalysis()
        msg.header.seq = header.seq
        msg.header.stamp = header.stamp
        msg.header.frame_id = depth_camera_info.header.frame_id
        msg.contains_3d_positions = True

        for object_analysis in object_analyses:
            o = self._object_analysis_to_msg(object_analysis, image_height, image_width, depth_image, depth_camera_info)
            msg.objects.append(o)

        if semantic_segmentation is not None:
            msg.semantic_segmentation.append(self._semantic_segmentation_to_msg(semantic_segmentation))

        return msg

    def _object_analysis_to_msg(self, object_analysis, image_height, image_width, depth_image, depth_camera_info):
        o = VideoAnalysisObject()
        o.center_2d = Point(x=object_analysis.center_x / image_width, y=object_analysis.center_y / image_height)
        o.center_3d = self._project_2d_to_3d(object_analysis.center_x, object_analysis.center_y,
                                             depth_image, depth_camera_info)
        o.width_2d = object_analysis.width / image_width
        o.height_2d = object_analysis.height / image_height
        o.object_class = object_analysis.object_class
        o.object_confidence = object_analysis.object_confidence
        o.object_class_probability = object_analysis.object_class_probability
        if object_analysis.object_image is not None:
            o.object_image = self._cv_bridge.cv2_to_imgmsg(object_analysis.object_image, encoding='rgb8')
        o.object_descriptor = object_analysis.object_descriptor

        if object_analysis.pose_analysis is not None:
            o.person_pose_2d = convert_pose_coordinates_2d(object_analysis.pose_analysis.pose_coordinates,
                                                           image_width, image_height)
            o.person_pose_3d = self._convert_pose_coordinates_3d(object_analysis.pose_analysis.pose_coordinates,
                                                                 depth_image, depth_camera_info)

            o.person_pose_confidence = object_analysis.pose_analysis.pose_confidence
            if object_analysis.pose_analysis.pose_image is not None:
                o.person_pose_image = self._cv_bridge.cv2_to_imgmsg(object_analysis.pose_analysis.pose_image, encoding='rgb8')

        if object_analysis.face_analysis is not None:
            o.face_descriptor = object_analysis.face_analysis.descriptor
            o.face_alignment_keypoint_count = object_analysis.face_analysis.alignment_keypoint_count
            o.face_sharpness_score = object_analysis.face_analysis.sharpness_score
            if object_analysis.face_analysis.face_image is not None:
                o.face_image = self._cv_bridge.cv2_to_imgmsg(object_analysis.face_analysis.face_image, encoding='rgb8')

        return o

    def _convert_pose_coordinates_3d(self, pose_coordinates, depth_image, depth_camera_info):
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
        depth_pixels = depth_image[y0:y1, x0:x1]
        depth = depth_pixels[depth_pixels != 0].mean()
        K = depth_camera_info.K

        point = Point()
        point.x = (x - K[2]) * depth / K[0] * MM_TO_M
        point.y = (y - K[5]) * depth / K[4] * MM_TO_M
        point.z = depth * MM_TO_M
        return point


def main():
    rospy.init_node('video_analyzer_3d_node')
    video_analyzer_node = VideoAnalyzer3dNode()
    video_analyzer_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
