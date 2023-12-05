#!/usr/bin/env python3

from abc import ABC, abstractmethod
import math
import threading

import numpy as np
from scipy.optimize import dual_annealing

import rospy
from odas_ros.msg import OdasSstArrayStamped
from video_analyzer.msg import VideoAnalysis

from t_top import MovementCommands, vector_to_angles, HEAD_POSE_PITCH_INDEX, HEAD_ZERO_Z

from sound_object_person_following import Camera3dCalibration


PERSON_CLASS = 'person'


class Target:
    def __init__(self, yaw=None, image_x=None, image_y=None):
        if yaw is not None and image_x is not None:
            raise ValueError('yaw and image_x cannot be set together')

        self.yaw = yaw
        self.image_x = image_x
        self.image_y = image_y

    def from_sound_following(yaw):
        return Target(yaw=yaw)

    def from_object_person_following(image_x, image_y):
        return Target(image_x=image_x, image_y=image_y)


class Follower(ABC):
    @property
    @abstractmethod
    def target(self):
        pass


class SoundFollower(Follower):
    def __init__(self, movement_commands):
        self._movement_commands = movement_commands

        self._min_sst_activity = rospy.get_param('~min_sst_activity')
        self._min_valid_sst_pitch = rospy.get_param('~min_valid_sst_pitch')
        self._max_valid_sst_pitch = rospy.get_param('~max_valid_sst_pitch')
        self._direction_frame_id = rospy.get_param('~direction_frame_id')

        self._target_lock = threading.Lock()
        self._target = None

        self._sst_sub = rospy.Subscriber('sst', OdasSstArrayStamped, self._sst_cb, queue_size=1)

    def _sst_cb(self, sst):
        if self._movement_commands.is_filtering_all_messages:
            return
        if len(sst.sources) > 1:
            rospy.logerr(f'Invalid sst (len(sst.sources)={len(sst.sources)})')
            return
        if sst.header.frame_id != self._direction_frame_id:
            rospy.logerr(f'Invalid direction frame id ({sst.header.frame_id} != {self._direction_frame_id})')
            return
        if len(sst.sources) == 0 or sst.sources[0].activity < self._min_sst_activity:
            return

        yaw, pitch = vector_to_angles(sst.sources[0])
        if pitch < self._min_valid_sst_pitch or pitch > self._max_valid_sst_pitch:
            return

        with self._target_lock:
            self._target = Target.from_sound_following(yaw)

    @property
    def target(self):
         with self._target_lock:
            return self._target


class ObjectPersonFollower(Follower):
    def __init__(self, camera_3d_calibration, movement_commands):
        self._camera_3d_calibration = camera_3d_calibration
        self._movement_commands = movement_commands

        self._object_classes = set(rospy.get_param('~object_classes'))
        self._padding = rospy.get_param('~padding')
        self._target_lambda = rospy.get_param('~target_lambda')
        _verify_padding(self._camera_3d_calibration, self._padding)

        self._target_lock = threading.Lock()
        self._target = None

        self._video_analysis_sub = rospy.Subscriber('video_analysis', VideoAnalysis, self._video_analysis_cb, queue_size=1)

    def _video_analysis_cb(self, msg):
        if self._movement_commands.is_filtering_all_messages:
            return

        person = self._find_biggest_person(msg.objects)
        if person is None:
            target = None
        else:
            target = self._find_target(msg, person)

        with self._target_lock:
            self._target = target

    def _find_biggest_person(self, objects):
        person_area_pairs = [(o, o.width_2d * o.height_2d) for o in objects if o.object_class == PERSON_CLASS]
        if len(person_area_pairs) == 0:
            return None

        return max(person_area_pairs, key= lambda x: x[1])[0]

    @abstractmethod
    def _find_target(self, msg, person, eps=1e-6):
        pass

    def _find_target_range(self, person):
        half_person_width = person.width_2d / 2
        half_person_height = person.height_2d / 2

        x_offset = half_person_width - self._camera_3d_calibration.half_width + self._padding
        min_x = person.center_2d.x + x_offset
        max_x = person.center_2d.x - x_offset
        if min_x > max_x:
            min_x = person.center_2d.x
            max_x = min_x

        y_offset = half_person_height - self._camera_3d_calibration.half_height + self._padding
        min_y = person.center_2d.y + y_offset
        max_y = person.center_2d.y - y_offset
        if min_y > max_y:
            min_y = max_y

        return min_x, max_x, min_y, max_y

    @property
    def target(self):
         with self._target_lock:
            return self._target


def _verify_padding(camera_3d_calibration, padding):
        if camera_3d_calibration.half_width < padding or camera_3d_calibration.half_height < padding:
            raise ValueError('The padding is too big.')


class BoundingBoxObjectPersonFollower(ObjectPersonFollower):
    def _find_target(self, msg, person, eps=1e-6):
        min_x, max_x, min_y, max_y = self._find_target_range(person)
        bounding_boxes = self._get_object_bounding_boxes(self._filter_objects(msg.objects))

        def cost(variables):
            target_x, target_y = variables

            person_center_loss = (person.center_2d.x - target_x)**2 + (person.center_2d.y - target_y)**2

            if len(bounding_boxes) > 0:
                wx0 = target_x - self._camera_3d_calibration.half_width
                wy0 = target_y - self._camera_3d_calibration.half_height
                wx1 = target_x + self._camera_3d_calibration.half_width
                wy1 = target_y + self._camera_3d_calibration.half_height

                intersection_w = np.minimum(bounding_boxes[:, 2], wx1) - np.maximum(bounding_boxes[:, 0], wx0)
                intersection_h = np.minimum(bounding_boxes[:, 3], wy1) - np.maximum(bounding_boxes[:, 1], wy0)
                intersection_w = np.clip(intersection_w, a_min=0., a_max=None)
                intersection_h = np.clip(intersection_h, a_min=0., a_max=None)
                intersection = (intersection_w * intersection_h).sum()
            else:
                intersection = 0

            return self._target_lambda * person_center_loss - intersection

        bounds = [(min_x, max_x + eps), (min_y, max_y + eps)]
        result = dual_annealing(cost, bounds, maxiter=50)

        return Target.from_object_person_following(result.x[0], result.x[1])

    def _filter_objects(self, objects):
        if self._object_classes == set(['all']):
            return objects
        else:
            return list(filter(lambda o: o.object_class in self._object_classes, objects))

    def _get_object_bounding_boxes(self, objects):
        bounding_boxes = []
        for o in objects:
            half_width = o.width_2d / 2 + self._padding
            half_height = o.height_2d / 2 + self._padding

            top_lelf_x = o.center_2d.x - half_width
            top_lelf_y = o.center_2d.y - half_height
            bottom_right_x = o.center_2d.x + half_width
            bottom_right_y = o.center_2d.y + half_height

            bounding_boxes.append([top_lelf_x, top_lelf_y, bottom_right_x, bottom_right_y])
        return np.array(bounding_boxes)


class SemanticSegmentationObjectPersonFollower(ObjectPersonFollower):
    def _find_target(self, msg, person, eps=1e-6):
        if len(msg.semantic_segmentation) == 0:
            rospy.logerr('The video analysis must have semantic segmentation.')
            return

        min_x, max_x, min_y, max_y = self._find_target_range(person)
        mask = self._get_mask_from_semantic_segmentation(msg.semantic_segmentation[0])

        window_width = self._camera_3d_calibration.half_width - self._padding
        window_height = self._camera_3d_calibration.half_height - self._padding

        def cost(variables):
            target_x, target_y = variables

            person_center_loss = (person.center_2d.x - target_x)**2 + (person.center_2d.y - target_y)**2

            h, w = msg.semantic_segmentation[0].shape
            x0 = np.clip(math.floor((target_x - window_width) * w), 0, w)
            y0 = np.clip(math.floor((target_y - window_height) * h), 0, h)
            x1 = np.clip(math.floor((target_x + window_width) * w), 0, w)
            y1 = np.clip(math.floor((target_y + window_height) * h), 0, h)
            intersection = mask[y0:y1, x0:x1].sum()

            return self._target_lambda * person_center_loss - intersection

        bounds = [(min_x, max_x + eps), (min_y, max_y + eps)]
        result = dual_annealing(cost, bounds, maxiter=50)

        return Target.from_object_person_following(result.x[0], result.x[1])

    def _get_mask_from_semantic_segmentation(self, semantic_segmentation):
        class_index_by_name = {i: name for i, name in enumerate(semantic_segmentation.class_names)}
        class_indexes = np.array(semantic_segmentation.class_indexes).reshape(semantic_segmentation.height,
                                                                              semantic_segmentation.width)

        if self._object_classes == set(['all']):
            background_index = class_index_by_name['background']
            return class_indexes != background_index
        else:
            desired_class_indexes = {class_index_by_name[name] for name in self._object_classes}
            return np.isin(class_indexes, desired_class_indexes)


class SoundObjectPersonFollowingNode:
    def __init__(self):
        self._simulation = rospy.get_param('~simulation')
        self._rate = rospy.Rate(rospy.get_param('~control_frequency'))
        self._torso_control_alpha = rospy.get_param('~torso_control_alpha')
        self._torso_control_p_gain = rospy.get_param('~torso_control_p_gain')
        self._head_control_p_gain = rospy.get_param('~head_control_p_gain')
        self._min_head_pitch = rospy.get_param('~min_head_pitch_rad')
        self._max_head_pitch = rospy.get_param('~max_head_pitch_rad')
        self._object_person_follower_type = rospy.get_param('~object_person_follower_type')
        self._camera_3d_calibration = Camera3dCalibration.load_json_file()

        self._movement_commands = MovementCommands(self._simulation, namespace='sound_object_person_following')
        self._sound_follower = SoundFollower(self._movement_commands)
        if self._object_person_follower_type == 'bounding_box':
            self._object_person_follower = BoundingBoxObjectPersonFollower(self._camera_3d_calibration, self._movement_commands)
        elif self._object_person_follower_type == 'semantic_segmentation':
            self._object_person_follower = SemanticSegmentationObjectPersonFollower(self._camera_3d_calibration, self._movement_commands)
        else:
            raise ValueError(f'Invalid object_person_follower_type (object_person_follower_type={object_person_follower_type})')

    def run(self):
        while not rospy.is_shutdown():
            self._rate.sleep()
            if self._movement_commands.is_filtering_all_messages:
                continue

            target = self._object_person_follower.target
            if target is None:
                target = self._sound_follower.target

            if target is not None:
                self._update_from_target(target)

    def _update_from_target(self, target):
        if target.yaw is not None:
            self._update_torso_from_yaw(target.yaw)
        elif target.image_x is not None:
            self._update_torso_from_image_x(target.image_x)

        if target.image_y is not None:
            self._update_head_from_image_y(target.image_y)

    def _update_torso_from_yaw(self, target_torso_yaw):
        distance = target_torso_yaw - self._movement_commands.current_torso_pose
        if distance < -math.pi:
            distance = 2 * math.pi + distance
        elif distance > math.pi:
            distance = -(2 * math.pi - distance)

        pose = self._movement_commands.current_torso_pose + self._torso_control_alpha * distance
        self._movement_commands.move_torso(pose)

    def _update_torso_from_image_x(self, current_image_x):
        current_yaw = self._movement_commands.current_torso_pose
        yaw = current_yaw + self._torso_control_p_gain * (self._camera_3d_calibration.center_x - current_image_x)
        self._movement_commands.move_torso(yaw)

    def _update_head_from_image_y(self, current_image_y):
        current_pitch = self._movement_commands.current_head_pose[HEAD_POSE_PITCH_INDEX]
        pitch = current_pitch + self._head_control_p_gain * (current_image_y - self._camera_3d_calibration.center_y)
        pitch = max(self._min_head_pitch, min(pitch, self._max_head_pitch))
        self._movement_commands.move_head([0, 0, HEAD_ZERO_Z, 0, pitch, 0])


def main():
    rospy.init_node('sound_object_person_following_node')
    explore_node = SoundObjectPersonFollowingNode()
    explore_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
