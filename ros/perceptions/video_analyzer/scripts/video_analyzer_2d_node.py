#!/usr/bin/env python3

import traceback
from datetime import datetime
import rospy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from video_analyzer.msg import VideoAnalysis, VideoAnalysisObject

import hbba_lite

from video_analyzer.lib_video_analyzer_node import VideoAnalyzerNode, convert_pose_coordinates_2d


class VideoAnalyzer2dNode(VideoAnalyzerNode):
    def __init__(self):
        super().__init__()
        self._image_sub = hbba_lite.ThrottlingHbbaSubscriber('image_raw', Image, self._image_cb, queue_size=1)

    def _image_cb(self, color_image_msg):
        try:
            start_time = datetime.now()
            color_image = self._cv_bridge.imgmsg_to_cv2(color_image_msg, 'rgb8')
            object_analyses, semantic_segmentation = self._analyse(color_image)

            video_analysis_msg = self._analysis_to_msg(object_analyses, semantic_segmentation,
                                                       color_image_msg.header, color_image)
            video_analysis_msg.processing_time_s = (datetime.now() - start_time).total_seconds()
            self._video_analysis_pub.publish(video_analysis_msg)
            self._publish_analysed_image(color_image, color_image_msg.header, object_analyses)
        except Exception as e:
            rospy.logerr(f'Image analysis error: {e} \n {traceback.format_exc()}')

    def _analysis_to_msg(self, object_analyses, semantic_segmentation, header, color_image):
        image_height, image_width, _ = color_image.shape

        msg = VideoAnalysis()
        msg.header = header
        msg.contains_3d_positions = False

        for object_analysis in object_analyses:
            msg.objects.append(self._object_analysis_to_msg(object_analysis, image_height, image_width))

        if semantic_segmentation is not None:
            msg.semantic_segmentation.append(self._semantic_segmentation_to_msg(semantic_segmentation))

        return msg

    def _object_analysis_to_msg(self, object_analysis, image_height, image_width):
        o = VideoAnalysisObject()
        o.center_2d = Point(x=object_analysis.center_x / image_width, y=object_analysis.center_y / image_height)
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


def main():
    rospy.init_node('video_analyzer_2d_node')
    video_analyzer_node = VideoAnalyzer2dNode()
    video_analyzer_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
