#!/usr/bin/env python3

import threading

import numpy as np

import rospy
from hbba_lite.srv import SetOnOffFilterState
from audio_analyzer.msg import AudioAnalysis

import person_identification


INACTIVE_SLEEP_DURATION = 0.1


class CaptureVoiceNode:
    def __init__(self):
        self._name = rospy.get_param('~name')
        self._mean_size = rospy.get_param('~mean_size')

        self._descriptors_lock = threading.Lock()
        self._descriptors = []
        self.audio_analysis = rospy.Subscriber('audio_analysis', AudioAnalysis, self._audio_analysis_cb, queue_size=1)

    def _audio_analysis_cb(self, msg):
        if len(msg.voice_descriptor) > 0:
            with self._descriptors_lock:
                self._descriptors.append(msg.voice_descriptor)
            print('voice')
        else:
            print('no voice')

    def run(self):
        self.enable_audio_analyzer()

        while not rospy.is_shutdown():
            with self._descriptors_lock:
                size = len(self._descriptors)

            if size == self._mean_size:
                self.audio_analysis.unregister()
                self._save_new_descriptor()
                return
            else:
                rospy.sleep(INACTIVE_SLEEP_DURATION)

    def enable_audio_analyzer(self):
        AUDIO_ANALYZER_FILTER_STATE_SERVICE = 'audio_analyzer/filter_state'

        rospy.wait_for_service(AUDIO_ANALYZER_FILTER_STATE_SERVICE)
        filter_state = rospy.ServiceProxy(AUDIO_ANALYZER_FILTER_STATE_SERVICE, SetOnOffFilterState)
        filter_state(is_filtering_all_messages=False)

    def _save_new_descriptor(self):
        with self._descriptors_lock:
            descriptor = np.array(self._descriptors).mean(axis=0).tolist()
            people = person_identification.load_people()
            person_identification.add_person(people, self._name, {'voice': descriptor})
            person_identification.save_people(people)

        rospy.loginfo('*********************** FINISHED ***********************')


def main():
    rospy.init_node('capture_voice_node')
    capture_voice_node = CaptureVoiceNode()
    capture_voice_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
