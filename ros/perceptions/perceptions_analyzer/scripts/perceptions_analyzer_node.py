#!/usr/bin/env python3

import time
import json
from datetime import datetime
from collections import defaultdict

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from perception_msgs.msg import (
    AudioAnalysis,
    VideoAnalysis,
    PersonNames,
    ContextInput,
    IdentifiedPerson,
    DetectedObjects,
    DetectedAudio,
)
from perception_msgs.srv import PerceiveObjects
from ament_index_python.packages import get_package_share_directory


class PerceptionsAnalyzer(Node):
    def __init__(self):
        super().__init__("perceptions_analyzer_service")

        self.clothing_description = None
        self.mood = None
        self.person_identity = None
        self.object = None

        self.currently_visible_objects = set()
        self.currently_visible_objects_detection_status_historic = {}
        self.currently_visible_objects_timestamp_historic = {}

        self.currently_identified_persons = set()
        self.currently_identified_persons_detection_status_historic = {}
        self.currently_identified_persons_timestamp_historic = {}

        self.audio_event_active = defaultdict(bool)
        self.audio_event_start_time = defaultdict(float)

        self.DETECTION_TIMEOUT = 120
        self.MIN_CONFIDENCE = 0.7
        self.n_thresh_detection = 20
        self.n_thresh_disparition = 40

        self.object_pub = self.create_publisher(
            DetectedObjects, "/perception/detected_objects", 10
        )
        self.identity_pub = self.create_publisher(
            IdentifiedPerson, "/perception/identified_person", 10
        )
        self.detected_audio_pub = self.create_publisher(
            DetectedAudio, "/perception/detected_audioClass", 10
        )
        self.current_objects_pub = self.create_publisher(
            ContextInput, "/perception/current_objects", 10
        )

        self.create_subscription(
            VideoAnalysis, "/camera_3d/video_analysis", self.video_analyzer_callback, 10
        )
        self.create_subscription(
            PersonNames, "/person_names", self.person_identification_callback, 10
        )
        self.create_subscription(
            AudioAnalysis, "audio_analysis", self.audio_analyzer_callback, 10
        )

        self._current_local_weather_service = self.create_service(
            PerceiveObjects,
            "perception/detected_objects",
            self._handle_detected_objects,
        )
        self._classes_file = (
            get_package_share_directory("perceptions_analyzer")
            + "/classes/classes.json",
        )

        f = open(self._classes_file[0], "r")

        classes = json.load(f)
        f.close()

        self.video_ignored_classes = [
            x.lower() for x in classes["video_ignored_classes"]
        ]
        self.acknowledged_classes = [
            x.lower() for x in classes["audio_acknowledged_classes"]
        ]

    def video_analyzer_callback(self, msg: VideoAnalysis):
        current_time = time.time()

        detected_objects_this_frame = set()

        for obj in msg.objects:
            if (
                obj.object_confidence > self.MIN_CONFIDENCE
                and obj.object_class.lower() not in self.video_ignored_classes
            ):
                detected_objects_this_frame.add(obj.object_class)
                history = (
                    self.currently_visible_objects_detection_status_historic.setdefault(
                        obj.object_class, []
                    )
                )
                timestamps = (
                    self.currently_visible_objects_timestamp_historic.setdefault(
                        obj.object_class, []
                    )
                )

                history.append(True)
                timestamps.append(current_time)

                if obj.object_class not in self.currently_visible_objects:
                    if len(history) >= self.n_thresh_detection and all(
                        history[-self.n_thresh_detection :]
                    ):
                        self.get_logger().info(f"Objet detected: {obj.object_class}")
                        self.currently_visible_objects.add(obj.object_class)
                        self.object_pub.publish(
                            DetectedObjects(
                                header=Header(
                                    stamp=self.get_clock().now().to_msg(), frame_id=""
                                ),
                                object_name=obj.object_class,
                                is_visible=True,
                            )
                        )

                        self.current_objects_pub.publish(
                            ContextInput(
                                header=Header(
                                    stamp=self.get_clock().now().to_msg(), frame_id=""
                                ),
                                text="",
                                role="",
                                objects=self.currently_visible_objects,
                                revive_conversation=False,
                            )
                        )

        for obj in list(self.currently_visible_objects):
            if obj not in detected_objects_this_frame:
                self.currently_visible_objects_detection_status_historic[obj].append(
                    False
                )
                self.currently_visible_objects_timestamp_historic[obj].append(
                    current_time
                )

                history = self.currently_visible_objects_detection_status_historic[obj]
                if len(history) > self.n_thresh_disparition and not any(
                    history[-self.n_thresh_disparition :]
                ):
                    self.get_logger().info(f"Objet gone: {obj}")
                    self.object_pub.publish(
                        DetectedObjects(
                            header=Header(
                                stamp=self.get_clock().now().to_msg(), frame_id=""
                            ),
                            object_name=obj,
                            is_visible=False,
                        )
                    )
                    del self.currently_visible_objects_detection_status_historic[obj]
                    del self.currently_visible_objects_timestamp_historic[obj]
                    self.currently_visible_objects.remove(obj)

                    self.current_objects_pub.publish(
                        ContextInput(
                            header=Header(
                                stamp=self.get_clock().now().to_msg(), frame_id=""
                            ),
                            text="",
                            role="",
                            objects=self.currently_visible_objects,
                            revive_conversation=False,
                        )
                    )

    def person_identification_callback(self, msg: PersonNames):
        current_time = time.time()

        detected_persons_this_frame = set()

        for person in msg.names:
            name = person.name
            detected_persons_this_frame.add(name)
            history = (
                self.currently_identified_persons_detection_status_historic.setdefault(
                    name, []
                )
            )
            timestamps = (
                self.currently_identified_persons_timestamp_historic.setdefault(
                    name, []
                )
            )

            history.append(True)
            timestamps.append(current_time)

            if (
                len(history) >= self.n_thresh_detection
                and all(history[-self.n_thresh_detection :])
                and name not in self.currently_identified_persons
            ):
                self.get_logger().warn(f"Personne détectée: {name}")
                self.currently_identified_persons.add(name)
                self.identity_pub.publish(
                    IdentifiedPerson(
                        person_name=name,
                        is_visible=True,
                    )
                )

        for name in list(self.currently_identified_persons):
            if name not in detected_persons_this_frame:
                history = self.currently_identified_persons_detection_status_historic[
                    name
                ]
                history.append(False)
                self.currently_identified_persons_timestamp_historic[name].append(
                    current_time
                )

                if len(history) > self.n_thresh_disparition and not any(
                    history[-self.n_thresh_disparition :]
                ):
                    self.get_logger().warn(f"Personne disparue: {name}")
                    self.identity_pub.publish(
                        IdentifiedPerson(
                            person_name=name,
                            is_visible=False,
                        )
                    )
                    del self.currently_identified_persons_detection_status_historic[
                        name
                    ]
                    del self.currently_identified_persons_timestamp_historic[name]
                    self.currently_identified_persons.remove(name)

    def audio_analyzer_callback(self, msg: AudioAnalysis):
        detected_now = set(
            audio_class.audio_class.lower() for audio_class in msg.audio_classes
        )

        for audio_class in detected_now:
            if (
                audio_class in self.audio_acknowledged_classes
                and not self.audio_event_active[audio_class]
            ):
                self.audio_event_active[audio_class] = True
                self.get_logger().warn(f"Son actif: {audio_class}")
                self.detected_audio_pub.publish(
                    DetectedAudio(
                        header=msg.header,
                        audio_class_name=audio_class,
                        is_active=True,
                    )
                )

        for audio_class in list(self.audio_event_active):
            if audio_class not in detected_now and self.audio_event_active[audio_class]:
                self.audio_event_active[audio_class] = False
                self.get_logger().warn(f"Son inactif: {audio_class}")
                self.detected_audio_pub.publish(
                    DetectedAudio(
                        header=msg.header,
                        audio_class_name=audio_class,
                        is_active=False,
                    )
                )

    def _handle_detected_objects(self, request, response):
        self.get_logger().info(f"Fetching current_objects")
        try:
            response.ok = True
            response.objects = self.currently_visible_objects

        except Exception as e:
            self.get_logger().error(
                f"An error occured while retrieving the perceived objects: {e}"
            )
            response.ok = False

        return response


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionsAnalyzer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
