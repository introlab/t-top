#!/usr/bin/env python3

import rospy

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from video_analyzer.msg import VideoAnalysis

class VideoAnalysisMarkersNode:
    def __init__(self):
        self._video_analysis_sub = rospy.Subscriber('video_analysis', VideoAnalysis, self._video_analysis_cb, queue_size=10)
        self._video_analysis_markers_pub = rospy.Publisher('video_analysis_markers', MarkerArray, queue_size=10)

    def _delete_markers(self):
        markerArray = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        markerArray.markers.append(marker)
        self._video_analysis_markers_pub.publish(markerArray)

    def _create_marker(self, header, tag, type, pose, ID, scale, rgb, points=[], name=''):
        marker = Marker()
        marker.header.stamp = header.stamp
        marker.header.frame_id = header.frame_id
        marker.ns = tag
        marker.id = ID
        marker.type = type
        marker.action = Marker.ADD
        marker.pose.position = pose
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.a = 1.0
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        marker.points = points
        marker.text = name

        return marker

    def _video_analysis_cb(self, msg):
        if not msg.contains_3d_positions:
            rospy.logerr('The video analysis must contain 3d positions.')
            return

        self._delete_markers()

        markerArray = MarkerArray()

        objID = 0
        personID = 0
        nameID = 0
        color = [1, 0, 0]
        for obj in msg.objects:
            if obj.person_pose_3d == []:
                color = [1, 0, 0]
                objMarker = self._create_marker(msg.header, 'objects', Marker.SPHERE, obj.center_3d, objID, 0.1, color)
                markerArray.markers.append(objMarker)

                objID += 1

            else:
                color = [0, 1, 0]
                personMarker = self._create_marker(msg.header, 'people', Marker.SPHERE_LIST, obj.person_pose_3d[0], personID, 0.05, color, obj.person_pose_3d)
                markerArray.markers.append(personMarker)

                personID += 1

            nameMarker = self._create_marker(msg.header, 'names', Marker.TEXT_VIEW_FACING, obj.center_3d, nameID, 0.1, color, name=obj.object_class)
            markerArray.markers.append(nameMarker)

            nameID += 1

        self._video_analysis_markers_pub.publish(markerArray)

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('video_analysis_markers_node')
    video_analysis_markers_node = VideoAnalysisMarkersNode()
    video_analysis_markers_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
