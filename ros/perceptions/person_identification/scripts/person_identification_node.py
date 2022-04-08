#!/usr/bin/env python3

import threading

import numpy as np

import rospy
import tf
from geometry_msgs.msg import PointStamped, Point, Vector3
from video_analyzer.msg import VideoAnalysis
from audio_analyzer.msg import AudioAnalysis
from person_identification.msg import PersonNames, PersonName

import person_identification


PERSON_POSE_NOSE_INDEX = 0


def calculate_distance(a, b):
    return np.linalg.norm(a - b)


def calculate_angle(a, b):
    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))


class FaceDescriptorData:
    def __init__(self, descriptor, position, direction):
        self.descriptor = descriptor
        self.position = position
        self.direction = direction


class VoiceDescriptorData:
    def __init__(self, descriptor, direction):
        self.descriptor = descriptor
        self.direction = direction


class PersonIdentificationNode:
    def __init__(self):
        self._face_descriptor_threshold = rospy.get_param('~face_descriptor_threshold')
        self._voice_descriptor_threshold = rospy.get_param('~voice_descriptor_threshold')
        self._face_voice_descriptor_threshold = rospy.get_param('~face_voice_descriptor_threshold')
        self._nose_confidence_threshold = rospy.get_param('~nose_confidence_threshold')
        self._direction_frame_id = rospy.get_param('~direction_frame_id')
        self._direction_angle_threshold_rad = rospy.get_param('~direction_angle_threshold_rad')
        self._ignore_direction_z = rospy.Rate(rospy.get_param('~ignore_direction_z'))
        self._rate = rospy.Rate(rospy.get_param('~search_frequency'))

        self._face_descriptors_by_name = {}
        self._voice_descriptors_by_name = {}
        self._face_voice_descriptors_by_name = {}
        self._load_descriptors()

        self._descriptors_lock = threading.Lock()
        self._face_descriptor_data = []
        self._voice_descriptor_data = None
        self._person_name_pub = rospy.Publisher('person_names', PersonNames, queue_size=5)

        self._tf_listener = tf.TransformListener()
        self._video_analysis_sub = rospy.Subscriber('video_analysis', VideoAnalysis, self._video_analysis_cb, queue_size=1)
        self.audio_analysis = rospy.Subscriber('audio_analysis', AudioAnalysis, self._audio_analysis_cb, queue_size=1)

    def _load_descriptors(self):
        people = person_identification.load_people()
        for name, descriptors in people.items():
            if 'face' in descriptors:
                self._face_descriptors_by_name[name] = np.array(descriptors['face'])
            if 'voice' in descriptors:
                self._voice_descriptors_by_name[name] = np.array(descriptors['voice'])
            if 'face' in descriptors and 'voice' in descriptors:
                self._face_voice_descriptors_by_name[name] = np.array(descriptors['face'] + descriptors['voice'])

    def _video_analysis_cb(self, msg):
        with self._descriptors_lock:
            for object in msg.objects:
                if len(object.face_descriptor) == 0 or len(object.person_pose) == 0 or len(object.person_pose_confidence) == 0 \
                        or object.person_pose_confidence[PERSON_POSE_NOSE_INDEX] < self._nose_confidence_threshold:
                    continue

                position, direction = self._get_face_position_and_direction(object.person_pose[PERSON_POSE_NOSE_INDEX], msg.header)
                if np.isfinite(direction).all():
                    self._face_descriptor_data.append(FaceDescriptorData(np.array(object.face_descriptor), position, direction))

    def _get_face_position_and_direction(self, point, header):
        temp_in_point = PointStamped()
        temp_in_point.header = header
        temp_in_point.point.x = point.x
        temp_in_point.point.y = point.y
        temp_in_point.point.z = point.z

        odas_point = self._tf_listener.transformPoint(self._direction_frame_id, temp_in_point)

        position = np.array([odas_point.point.x, odas_point.point.y, odas_point.point.z])
        direction = position.copy()
        if self._ignore_direction_z:
            direction[2] = 0

        direction /= np.linalg.norm(direction)
        return position, direction

    def _audio_analysis_cb(self, msg):
        if msg.header.frame_id != self._direction_frame_id:
            rospy.logerr(f'Invalid frame id ({msg.header.frame_id} != {self._direction_frame_id})')
            return

        if len(msg.voice_descriptor) == 0:
            return

        voice_direction = np.array([msg.direction_x, msg.direction_y, msg.direction_z])
        if self._ignore_direction_z:
            voice_direction[2] = 0

        voice_direction /= np.linalg.norm(voice_direction)
        if np.isfinite(voice_direction).all():
            with self._descriptors_lock:
                self._voice_descriptor_data = VoiceDescriptorData(np.array(msg.voice_descriptor), voice_direction)

    def run(self):
        while not rospy.is_shutdown():
            with self._descriptors_lock:
                names = []
                names.extend(self._find_face_voice_descriptor_name())
                names.extend(self._find_voice_descriptor_name())
                names.extend(self._find_face_descriptor_names())

                self._face_descriptor_data.clear()
                self._voice_descriptor_data = None

            self._publish_names(names)
            self._rate.sleep()

    def _find_face_voice_descriptor_name(self):
        if self._voice_descriptor_data is None or len(self._face_descriptor_data) == 0 or \
                len(self._face_voice_descriptors_by_name) == 0:
            return []

        face_descriptor_index_angle_pairs = [(i, calculate_angle(x.direction, self._voice_descriptor_data.direction))
                                             for i, x in enumerate(self._face_descriptor_data)]
        face_descriptor_index, angle = min(face_descriptor_index_angle_pairs, key=lambda x: x[1])
        if angle > self._direction_angle_threshold_rad:
            return []

        face_descriptor_data = self._face_descriptor_data[face_descriptor_index]
        descriptor = np.concatenate([face_descriptor_data.descriptor, self._voice_descriptor_data.descriptor])

        name_distance_pairs = [(x[0], calculate_distance(x[1], descriptor))
                               for x in self._face_voice_descriptors_by_name.items()]
        name, distance = min(name_distance_pairs, key=lambda x: x[1])
        if distance <= self._face_voice_descriptor_threshold:
            self._voice_descriptor_data = None
            del self._face_descriptor_data[face_descriptor_index]
            return [self._create_person_name(name,
                                             position=face_descriptor_data.position,
                                             direction=face_descriptor_data.direction)]
        else:
            return []

    def _find_face_descriptor_names(self):
        if len(self._face_descriptors_by_name) == 0:
            return []

        names = []
        for face_descriptor_data in self._face_descriptor_data:
            name_distance_pairs = [(x[0], calculate_distance(x[1], face_descriptor_data.descriptor))
                                   for x in self._face_descriptors_by_name.items()]
            name, distance = min(name_distance_pairs, key=lambda x: x[1])

            if distance <= self._face_descriptor_threshold:
                names.append(self._create_person_name(name,
                                                      position=face_descriptor_data.position,
                                                      direction=face_descriptor_data.direction))

        return names # filter names

    def _find_voice_descriptor_name(self):
        if self._voice_descriptor_data is None or len(self._voice_descriptors_by_name) == 0:
            return []

        name_distance_pairs = [(x[0], calculate_distance(x[1], self._voice_descriptor_data.descriptor))
                               for x in self._voice_descriptors_by_name.items()]
        name, distance = min(name_distance_pairs, key=lambda x: x[1])

        if distance <= self._voice_descriptor_threshold:
            return [self._create_person_name(name,
                                             direction=self._voice_descriptor_data.direction)]
        else:
            return []

    def _create_person_name(self, name, position=None, direction=None):
        person_name = PersonName()
        person_name.name = name
        person_name.frame_id = self._direction_frame_id
        if position is not None:
            person_name.position.append(Point(x=position[0], y=position[1], z=position[2]))
        if direction is not None:
            person_name.direction.append(Vector3(x=direction[0], y=direction[1], z=direction[2]))

        return person_name

    def _publish_names(self, names):
        msg = PersonNames()
        msg.names = self._filter_names(names)
        self._person_name_pub.publish(msg)

    def _filter_names(self, names):
        inserted_names = set()
        filtered_names = []

        for name in reversed(names):
            if name.name not in inserted_names:
                filtered_names.append(name)
                inserted_names.add(name.name)

        return filtered_names


def main():
    rospy.init_node('person_identification_node')
    person_identification_node = PersonIdentificationNode()
    person_identification_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
