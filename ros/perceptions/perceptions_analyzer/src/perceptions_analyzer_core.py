#!/usr/bin/env python3

import time
from datetime import datetime
from collections import defaultdict

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from perceptions_analyzer.msg import IdentifiedPerson, DetectedObjects, DetectedAudio
from audio_analyzer.msg import AudioAnalysis
from video_analyzer.msg import VideoAnalysis
from person_identification.msg import PersonNames


class PerceptionsAnalyzer(Node):
    def __init__(self):
        super().__init__('perceptions_analyzer_service')

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

        self.object_pub = self.create_publisher(DetectedObjects, '/perception/detected_objects', 10)
        self.identity_pub = self.create_publisher(IdentifiedPerson, '/perception/identified_person', 10)
        self.detectedAudio_pub = self.create_publisher(DetectedAudio, '/perception/detected_audioClass', 10)

        self.create_subscription(VideoAnalysis, '/camera_3d/video_analysis', self.videoAnalyzerCallback, 10)
        self.create_subscription(PersonNames, '/person_names', self.personIdentificationCallback, 10)
        self.create_subscription(AudioAnalysis, 'audio_analysis', self.audioAnalyzerCallback, 10)

        self.video_ignored_classes = [x.lower() for x in [
            "Person", "Car", "Street Lights", "Plate", "Helmet", "Boat", "Bench", "Bowl/Basin", "SUV", "Traffic Light",
            "Bicycle", "Barrel/bucket", "Van", "Drum", "Bus", "Wild Bird", "Motorcycle", "Truck", "Traffic cone",
            "Cymbal", "Lifesaver", "Towel", "Sailboat", "Awning", "Faucet", "Tent", "Horse",
            "Sink", "Paddle", "Traffic Sign", "Dog", "Cow", "Cake", "Other Fish", "Orange/Tangerine", "Toiletry",
            "Machinery Vehicle", "Airplane", "Train", "Trolley", "Sports Car", "Stop Sign", "Scooter", "Stroller",
            "Crane", "Duck", "Elephant", "Gun", "Pigeon", "Snowboard", "Fire Hydrant", "Zebra", "Giraffe", "Tricycle",
            "Fire Truck", "Billiards", "Heavy Truck", "Extractor", "Extension Cord", "Tong", "Tennis", "Ship",
            "Swing", "Slide", "Carriage", "Washing Machine/Drier", "Chicken", "Scale", "Deer", "Ambulance", "Parking meter",
            "Hurdle", "Fishing Rod", "Medal", "Penguin", "Swan", "Helicopter", "Speed Limit Sign",
            "Rickshaw", "Goldfish", "Pliers", "Hammer", "Screwdriver", "Bear", "Pig", "Showerhead", "Crosswalk Sign",
            "Camel", "Formula1", "Crab", "Antelope", "Parrot", "Seal", "Butterfly", "Donkey", "Lion", "Urinal",
            "Dolphin", "Jellyfish", "Target", "Monkey", "Rabbit", "Yak", "Barbell", "Scallop", "Oyster", "Table Tennis",
            "Paddle", "Chainsaw", "Lobster"
        ]]

    def videoAnalyzerCallback(self, msg):
        current_time = time.time()
        formatted_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

        detected_objects_this_frame = set()

        for obj in msg.objects:
            if obj.object_confidence > self.MIN_CONFIDENCE and obj.object_class.lower() not in self.video_ignored_classes:
                detected_objects_this_frame.add(obj.object_class)

                history = self.currently_visible_objects_detection_status_historic.setdefault(obj.object_class, [])
                timestamps = self.currently_visible_objects_timestamp_historic.setdefault(obj.object_class, [])

                history.append(True)
                timestamps.append(current_time)

                if obj.object_class not in self.currently_visible_objects:
                    if len(history) >= self.n_thresh_detection and all(history[-self.n_thresh_detection:]):
                        self.get_logger().warn(f"Objet détecté constamment: {obj.object_class}")
                        self.currently_visible_objects.add(obj.object_class)
                        self.object_pub.publish(DetectedObjects(
                            header=self.get_clock().now().to_msg(),
                            object_name=obj.object_class,
                            formatted_detection_time=formatted_time,
                            is_visible=True
                        ))
            
        for obj in list(self.currently_visible_objects):
            if obj not in detected_objects_this_frame:
                self.currently_visible_objects_detection_status_historic[obj].append(False)
                self.currently_visible_objects_timestamp_historic[obj].append(current_time)

                history = self.currently_visible_objects_detection_status_historic[obj]
                if len(history) > self.n_thresh_disparition and not any(history[-self.n_thresh_disparition:]):
                    self.get_logger().warn(f"Objet disparu: {obj}")
                    self.object_pub.publish(DetectedObjects(
                        header=self.get_clock().now().to_msg(),
                        object_name=obj,
                        formatted_detection_time=formatted_time,
                        is_visible=False
                    ))
                    del self.currently_visible_objects_detection_status_historic[obj]
                    del self.currently_visible_objects_timestamp_historic[obj]
                    self.currently_visible_objects.remove(obj)

    def personIdentificationCallback(self, msg):
        current_time = time.time()
        formatted_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

        detected_persons_this_frame = set()

        for person in msg.names:
            name = person.name
            detected_persons_this_frame.add(name)
            history = self.currently_identified_persons_detection_status_historic.setdefault(name, [])
            timestamps = self.currently_identified_persons_timestamp_historic.setdefault(name, [])

            history.append(True)
            timestamps.append(current_time)

            if len(history) >= self.n_thresh_detection and all(history[-self.n_thresh_detection:]) and name not in self.currently_identified_persons:
                self.get_logger().warn(f"Personne détectée: {name}")
                self.currently_identified_persons.add(name)
                self.identity_pub.publish(IdentifiedPerson(
                    person_name=name,
                    formatted_detection_time=formatted_time,
                    is_visible=True
                ))

        for name in list(self.currently_identified_persons):
            if name not in detected_persons_this_frame:
                history = self.currently_identified_persons_detection_status_historic[name]
                history.append(False)
                self.currently_identified_persons_timestamp_historic[name].append(current_time)

                if len(history) > self.n_thresh_disparition and not any(history[-self.n_thresh_disparition:]):
                    self.get_logger().warn(f"Personne disparue: {name}")
                    self.identity_pub.publish(IdentifiedPerson(
                        person_name=name,
                        formatted_detection_time=formatted_time,
                        is_visible=False
                    ))
                    del self.currently_identified_persons_detection_status_historic[name]
                    del self.currently_identified_persons_timestamp_historic[name]
                    self.currently_identified_persons.remove(name)

    def audioAnalyzerCallback(self, msg):
        currenthttps://www.messenger.com/_time = time.time()
        formatted_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

        acknowledged_classes = [x.lower() for x in [
            "Musical_instrument", "Music", "Applause", "Harmonica", "Guitar", "Acoustic_guitar", "Knock",
            "Cough", "Laughter", "Bell", "Alarm", "Sneeze", "Singing", "Telephone", "Computer_keyboard",
            "Electric_guitar", "Door", "Microwave", "oven", "Typing"
        ]]

        detected_now = set(audio_class.audio_class.lower() for audio_class in msg.audio_classes)

        for audio_class in detected_now:
            if audio_class in acknowledged_classes and not self.audio_event_active[audio_class]:
                self.audio_event_active[audio_class] = True
                self.get_logger().warn(f"Son actif: {audio_class}")
                self.detectedAudio_pub.publish(DetectedAudio(
                    header=msg.header,
                    audio_class_name=audio_class,
                    is_active=True,
                    formatted_detection_time=formatted_time
                ))

        for audio_class in list(self.audio_event_active):
            if audio_class not in detected_now and self.audio_event_active[audio_class]:
                self.audio_event_active[audio_class] = False
                self.get_logger().warn(f"Son inactif: {audio_class}")
                self.detectedAudio_pub.publish(DetectedAudio(
                    header=msg.header,
                    audio_class_name=audio_class,
                    is_active=False,
                    formatted_detection_time=formatted_time
                ))
