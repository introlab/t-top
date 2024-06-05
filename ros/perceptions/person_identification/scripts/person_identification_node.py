#!/usr/bin/env python3

import numpy as np

import rclpy
import rclpy.node
import rclpy.time

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs

from geometry_msgs.msg import PointStamped, Point, Vector3
from perception_msgs.msg import VideoAnalysis, AudioAnalysis, PersonNames, PersonName

import person_identification


PERSON_POSE_NOSE_INDEX = 0


def calculate_distance(a, b):
    return np.linalg.norm(a - b)


def calculate_angle(a, b):
    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))


class FaceDescriptorData:
    def __init__(self, descriptor, position_2d, position_3d, direction):
        self.descriptor = descriptor
        self.position_2d = position_2d
        self.position_3d = position_3d
        self.direction = direction


class VoiceDescriptorData:
    def __init__(self, descriptor, direction):
        self.descriptor = descriptor
        self.direction = direction


class PersonIdentificationNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('person_identification_node')

        self._face_sharpness_score_threshold = self.declare_parameter('face_sharpness_score_threshold', 0.5).get_parameter_value().double_value
        self._face_descriptor_threshold = self.declare_parameter('face_descriptor_threshold', 0.7).get_parameter_value().double_value
        self._voice_descriptor_threshold = self.declare_parameter('voice_descriptor_threshold', 1.266).get_parameter_value().double_value
        self._face_voice_descriptor_threshold = self.declare_parameter('face_voice_descriptor_threshold', 1.5092).get_parameter_value().double_value
        self._nose_confidence_threshold = self.declare_parameter('nose_confidence_threshold', 0.4).get_parameter_value().double_value
        self._direction_frame_id = self.declare_parameter('direction_frame_id', 'odas').get_parameter_value().string_value
        self._direction_angle_threshold_rad = self.declare_parameter('direction_angle_threshold_rad', 0.15).get_parameter_value().double_value
        self._ignore_direction_z = self.declare_parameter('ignore_direction_z', True).get_parameter_value().bool_value
        self._search_frequency = self.declare_parameter('search_frequency', 2.0).get_parameter_value().double_value

        self._face_descriptors_by_name = {}
        self._voice_descriptors_by_name = {}
        self._face_voice_descriptors_by_name = {}
        self._load_descriptors()

        self._face_descriptor_data = []
        self._voice_descriptor_data = None
        self._person_name_pub = self.create_publisher(PersonNames, 'person_names', 5)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._video_analysis_sub = self.create_subscription(VideoAnalysis, 'video_analysis', self._video_analysis_cb, 10)
        self._audio_analysis = self.create_subscription(AudioAnalysis, 'audio_analysis', self._audio_analysis_cb, 10)

        self._timer = self.create_timer(1 / self._search_frequency, self._timer_callback)

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
        if not msg.contains_3d_positions:
            self.get_logger().error('The video analysis must contain 3d positions.')
            return

        for object in msg.objects:
            if len(object.face_descriptor) == 0 or len(object.person_pose_2d) == 0 or len(object.person_pose_3d) == 0 \
                    or len(object.person_pose_confidence) == 0 \
                    or object.person_pose_confidence[PERSON_POSE_NOSE_INDEX] < self._nose_confidence_threshold \
                    or object.face_sharpness_score < self._face_sharpness_score_threshold:
                continue

            try:
                position_2d = object.person_pose_2d[PERSON_POSE_NOSE_INDEX]
                position_3d, direction = self._get_face_position_3d_and_direction(object.person_pose_3d[PERSON_POSE_NOSE_INDEX], msg.header)
                if np.isfinite(direction).all():
                    self._face_descriptor_data.append(FaceDescriptorData(np.array(object.face_descriptor),
                                                                         position_2d,
                                                                         position_3d,
                                                                         direction))
            except TransformException as ex:
                self.get_logger().warn(f'Could not transform: {ex}')

    def _get_face_position_3d_and_direction(self, point, header):
        temp_in_point = PointStamped()
        temp_in_point.header = header
        temp_in_point.point.x = point.x
        temp_in_point.point.y = point.y
        temp_in_point.point.z = point.z

        transform = self._tf_buffer.lookup_transform(self._direction_frame_id, header.frame_id, rclpy.time.Time.from_msg(header.stamp))
        odas_point = tf2_geometry_msgs.do_transform_point(temp_in_point, transform)

        position = np.array([odas_point.point.x, odas_point.point.y, odas_point.point.z])
        direction = position.copy()
        if self._ignore_direction_z:
            direction[2] = 0

        direction /= np.linalg.norm(direction)
        return position, direction

    def _audio_analysis_cb(self, msg):
        if msg.header.frame_id != self._direction_frame_id:
            self.get_logger().error(f'Invalid frame id ({msg.header.frame_id} != {self._direction_frame_id})')
            return

        if len(msg.voice_descriptor) == 0:
            return

        voice_direction = np.array([msg.direction_x, msg.direction_y, msg.direction_z])
        if self._ignore_direction_z:
            voice_direction[2] = 0

        voice_direction /= np.linalg.norm(voice_direction)
        if np.isfinite(voice_direction).all():
            self._voice_descriptor_data = VoiceDescriptorData(np.array(msg.voice_descriptor), voice_direction)

    def _timer_callback(self):
        names = []
        names.extend(self._find_face_voice_descriptor_name())
        names.extend(self._find_voice_descriptor_name())
        names.extend(self._find_face_descriptor_names())

        self._face_descriptor_data.clear()
        self._voice_descriptor_data = None

        self._publish_names(names)

    def run(self):
        rclpy.spin(self)

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
                                             'face_and_voice',
                                             position_2d=face_descriptor_data.position_2d,
                                             position_3d=face_descriptor_data.position_3d,
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
                                                      'face',
                                                      position_2d=face_descriptor_data.position_2d,
                                                      position_3d=face_descriptor_data.position_3d,
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
                                             'voice',
                                             direction=self._voice_descriptor_data.direction)]
        else:
            return []

    def _create_person_name(self, name, detection_type, position_2d=None, position_3d=None, direction=None):
        person_name = PersonName()
        person_name.name = name
        person_name.detection_type = detection_type
        person_name.frame_id = self._direction_frame_id
        if position_2d is not None:
            person_name.position_2d.append(position_2d)
        if position_3d is not None:
            person_name.position_3d.append(Point(x=position_3d[0], y=position_3d[1], z=position_3d[2]))
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
    rclpy.init()
    person_identification_node = PersonIdentificationNode()

    try:
        person_identification_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        person_identification_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
