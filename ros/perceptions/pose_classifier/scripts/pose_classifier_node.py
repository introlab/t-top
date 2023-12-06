#!/usr/bin/env python3

import numpy as np

import rospy
from video_analyzer.msg import VideoAnalysis

from pose_classifier.msg import PoseClassification, PoseClassifications


NOSE_INDEX = 0
LEFT_EYE_INDEX = 1
RIGHT_EYE_INDEX = 2
LEFT_EAR_INDEX = 3
RIGHT_EAR_INDEX = 4

LEFT_SHOULDER_INDEX = 5
RIGHT_SHOULDER_INDEX = 6
LEFT_ELBOW_INDEX = 7
RIGHT_ELBOW_INDEX = 8
LEFT_WRIST_INDEX = 9
RIGHT_WRIST_INDEX = 10

LEFT_HIP_INDEX = 11
RIGHT_HIP_INDEX = 12
LEFT_KNEE_INDEX = 13
RIGHT_KNEE_INDEX = 14
LEFT_ANKLE_INDEX = 15
RIGHT_ANKLE_INDEX = 16

LEFT_RIGHT_PAIR_INDEXES = [
    [LEFT_EYE_INDEX, RIGHT_EYE_INDEX],
    [LEFT_EAR_INDEX, RIGHT_EAR_INDEX],
    [LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX],
    [LEFT_HIP_INDEX, RIGHT_HIP_INDEX],
]

# Head ear-eye diffs
DOWN_EAR_EYE_DIFF = -1.0
NORMAL_EAR_EYE_DIFF = 0.0
UP_EAR_EYE_DIFF = 1.0

# Arm Angles
DOWN_SHOULDER_ELBOW_ANGLE_DEG = 90
HORIZONTAL_SHOULDER_ELBOW_ANGLE_DEG = 20
UP_SHOULDER_ELBOW_ANGLE_DEG = -45

STRAIGHT_ELBOW_WRIST_ANGLE_DEG = 0
UP_RIGHT_ANGLE_ELBOW_WRIST_ANGLE_DEG = -90
DOWN_RIGHT_ANGLE_ELBOW_WRIST_ANGLE_DEG = 90


def vector_angle(v1, v2, eps=1e-9):
    unit_v1 = v1 / (np.linalg.norm(v1) + eps)
    unit_v2 = v2 / (np.linalg.norm(v2) + eps)
    abs_angle = np.arccos(np.clip(np.dot(unit_v1, unit_v2), a_min=-1.0, a_max=1.0))

    v1_3d = np.array([v1[0], v1[1], 0.0])
    v2_3d = np.array([v2[0], v2[1], 0.0])

    c = np.cross(v1_3d, v2_3d)

    return np.sign(c[2]) * abs_angle


class PoseClassifierNode:
    def __init__(self):
        self._pose_confidence_threshold = rospy.get_param('~pose_confidence_threshold', 0.4)

        self._pose_classification_pub = rospy.Publisher('pose_classification', PoseClassifications, queue_size=1)
        self._video_analysis_sub = rospy.Subscriber('video_analysis', VideoAnalysis, self._video_analysis_cb, queue_size=1)

    def _video_analysis_cb(self, video_analysis_msg):
        pose_classifications_msg = PoseClassifications()
        pose_classifications_msg.header = video_analysis_msg.header

        for object in video_analysis_msg.objects:
            if len(object.person_pose_2d) == 0 or len(object.person_pose_confidence) == 0:
                continue

            pose = self._classify_pose(object.person_pose_2d, object.person_pose_confidence)
            pose.object = object
            pose_classifications_msg.poses.append(pose)

        self._pose_classification_pub.publish(pose_classifications_msg)

    def _classify_pose(self, person_pose_2d, person_pose_confidence):
        msg = PoseClassification()
        msg.is_facing_camera = self._detect_facing_camera(person_pose_2d, person_pose_confidence)

        if msg.is_facing_camera:
            msg.head_vertical_class = self._classify_head_vertically(person_pose_2d, person_pose_confidence)
            msg.head_horizontal_class = self._classify_head_horizontally(person_pose_confidence)
            msg.left_arm_class = self._classify_arm(person_pose_2d, person_pose_confidence,
                                                    RIGHT_SHOULDER_INDEX, LEFT_SHOULDER_INDEX,
                                                    LEFT_ELBOW_INDEX, LEFT_WRIST_INDEX)
            msg.right_arm_class = self._classify_arm(person_pose_2d, person_pose_confidence,
                                                     LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX,
                                                     RIGHT_ELBOW_INDEX, RIGHT_WRIST_INDEX, sign=-1)

        return msg

    def _detect_facing_camera(self, person_pose_2d, person_pose_confidence):
        is_facing_camera_count = 0
        is_not_facing_camera_count = 0

        for pair in LEFT_RIGHT_PAIR_INDEXES:
            if (person_pose_confidence[pair[0]] < self._pose_confidence_threshold or
                person_pose_confidence[pair[1]] < self._pose_confidence_threshold):
                continue

            if person_pose_2d[pair[0]].x >= person_pose_2d[pair[1]].x:
                is_facing_camera_count += 1
            else:
                is_not_facing_camera_count += 1

        return is_facing_camera_count >= is_not_facing_camera_count


    def _classify_head_vertically(self, person_pose_2d, person_pose_confidence):
        if (person_pose_confidence[LEFT_EYE_INDEX] < self._pose_confidence_threshold or
            person_pose_confidence[RIGHT_EYE_INDEX] < self._pose_confidence_threshold):
            return ''

        left_eye_x = person_pose_2d[LEFT_EYE_INDEX].x
        left_eye_y = person_pose_2d[LEFT_EYE_INDEX].y
        right_eye_x = person_pose_2d[RIGHT_EYE_INDEX].x
        right_eye_y = person_pose_2d[RIGHT_EYE_INDEX].y
        eye_mean_y = (left_eye_y + right_eye_y) / 2.0
        eye_distance = np.linalg.norm(np.array([left_eye_x - right_eye_x, left_eye_y - right_eye_y]))

        is_left_ear_visible = person_pose_confidence[LEFT_EAR_INDEX] > self._pose_confidence_threshold
        is_right_ear_visible = person_pose_confidence[RIGHT_EAR_INDEX] > self._pose_confidence_threshold
        if is_left_ear_visible and is_right_ear_visible:
            ear_mean_y = (person_pose_2d[LEFT_EAR_INDEX].y + person_pose_2d[RIGHT_EAR_INDEX].y) / 2.0
        elif is_left_ear_visible:
            ear_mean_y = person_pose_2d[LEFT_EAR_INDEX].y
        elif is_right_ear_visible:
            ear_mean_y = person_pose_2d[RIGHT_EAR_INDEX].y
        else:
            return ''

        diff = (ear_mean_y - eye_mean_y) / eye_distance
        ear_eye_diffs = np.array([DOWN_EAR_EYE_DIFF, NORMAL_EAR_EYE_DIFF, UP_EAR_EYE_DIFF])
        index = np.abs(ear_eye_diffs - diff).argmin()
        if index == 0:
            return 'down'
        elif index == 1:
            return 'straight'
        elif index == 2:
            return 'up'
        else:
            return ''

    def _classify_head_horizontally(self, person_pose_confidence):
        is_nose_visible = person_pose_confidence[NOSE_INDEX] > self._pose_confidence_threshold
        is_left_ear_visible = person_pose_confidence[LEFT_EAR_INDEX] > self._pose_confidence_threshold
        is_right_ear_visible = person_pose_confidence[RIGHT_EAR_INDEX] > self._pose_confidence_threshold

        if is_nose_visible and not is_left_ear_visible and is_right_ear_visible:
            return 'left'
        elif is_nose_visible and is_left_ear_visible and not is_right_ear_visible:
            return 'right'
        else:
            return 'straight'

    def _classify_arm(self, person_pose_2d, person_pose_confidence,
                      other_shoulder_index, shoulder_index, elbow_index, wrist_index,
                      sign=1):
        if (person_pose_confidence[other_shoulder_index] < self._pose_confidence_threshold or
            person_pose_confidence[shoulder_index] < self._pose_confidence_threshold or
            person_pose_confidence[elbow_index] < self._pose_confidence_threshold or
            person_pose_confidence[wrist_index] < self._pose_confidence_threshold):
            return ''

        right_left_shoulder_vector = np.array([
            person_pose_2d[shoulder_index].x - person_pose_2d[other_shoulder_index].x,
            person_pose_2d[shoulder_index].y - person_pose_2d[other_shoulder_index].y
        ])
        shoulder_elbow_vector = np.array([
            person_pose_2d[elbow_index].x - person_pose_2d[shoulder_index].x,
            person_pose_2d[elbow_index].y - person_pose_2d[shoulder_index].y
        ])
        elbow_wrist_vector = np.array([
            person_pose_2d[wrist_index].x - person_pose_2d[elbow_index].x,
            person_pose_2d[wrist_index].y - person_pose_2d[elbow_index].y
        ])

        shoulder_elbow_angle = sign * vector_angle(right_left_shoulder_vector, shoulder_elbow_vector)
        elbow_wrist_angle = sign * vector_angle(shoulder_elbow_vector, elbow_wrist_vector)

        return self._classify_arm_angles(shoulder_elbow_angle, elbow_wrist_angle)

    def _classify_arm_angles(self, shoulder_elbow_angle_rad, elbow_wrist_angle_rad):
        shoulder_elbow_angle_deg = np.rad2deg(shoulder_elbow_angle_rad)
        elbow_wrist_angle_deg = np.rad2deg(elbow_wrist_angle_rad)

        shoulder_elbow_target_angles = np.array([DOWN_SHOULDER_ELBOW_ANGLE_DEG,
                                                 HORIZONTAL_SHOULDER_ELBOW_ANGLE_DEG,
                                                 UP_SHOULDER_ELBOW_ANGLE_DEG])
        shoulder_elbow_target_angle_index = np.abs(shoulder_elbow_target_angles - shoulder_elbow_angle_deg).argmin()

        elbow_wrist_target_angles = np.array([STRAIGHT_ELBOW_WRIST_ANGLE_DEG,
                                              UP_RIGHT_ANGLE_ELBOW_WRIST_ANGLE_DEG,
                                              DOWN_RIGHT_ANGLE_ELBOW_WRIST_ANGLE_DEG])
        elbow_wrist_target_angle_index = np.abs(elbow_wrist_target_angles - elbow_wrist_angle_deg).argmin()

        if shoulder_elbow_target_angle_index == 0 and elbow_wrist_target_angle_index == 0:
            return 'down_straight'
        elif shoulder_elbow_target_angle_index == 0 and elbow_wrist_target_angle_index == 1:
            return 'down_right_angle_external'
        elif shoulder_elbow_target_angle_index == 0 and elbow_wrist_target_angle_index == 2:
            return 'down_right_angle_internal'
        elif shoulder_elbow_target_angle_index == 1 and elbow_wrist_target_angle_index == 0:
            return 'horizontal_straight'
        elif shoulder_elbow_target_angle_index == 1 and elbow_wrist_target_angle_index == 1:
            return 'horizontal_right_angle_up'
        elif shoulder_elbow_target_angle_index == 1 and elbow_wrist_target_angle_index == 2:
            return 'horizontal_right_angle_down'
        elif shoulder_elbow_target_angle_index == 2:
            return 'up'
        else:
            return ''

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('pose_classifier_node')
    pose_classifier_node = PoseClassifierNode()
    pose_classifier_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
