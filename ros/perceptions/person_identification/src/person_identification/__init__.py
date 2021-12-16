import os
import json

import numpy as np

import rospkg

PACKAGE_PATH = rospkg.RosPack().get_path('person_identification')
PEOPLE_FILE_PATH = os.path.join(PACKAGE_PATH, 'people.json')


def verify_people(people):
    if not isinstance(people, dict):
        raise ValueError('People must be a dictionary')

    for name, descriptors in people.items():
        verify_name(name)
        verify_descriptors(descriptors)


def verify_name(name):
    if not isinstance(name, str):
        raise ValueError('The name must be a string')
    if name == '':
        raise ValueError('THe name must not be empty')


def verify_descriptors(descriptors):
    if not isinstance(descriptors, dict):
        raise ValueError('The descriptors must be a dictionary')

    if 'face' not in descriptors and 'voice' not in descriptors:
        raise ValueError('The descriptors must contain "face" or/and "voice"')
    if len(descriptors) > 2:
        raise ValueError('The descriptors must contain only "face" or/and "voice"')

    if 'face' in descriptors:
        verify_descriptor(descriptors['face'])
    if 'voice' in descriptors:
        verify_descriptor(descriptors['voice'])


def verify_descriptor(descriptor):
    if not isinstance(descriptor, list):
        raise ValueError('The descriptor must be a list')

    try:
        np.array(descriptor, dtype=np.float)
    except Exception:
        raise ValueError('The descriptor is not valid')


def load_people():
    if not os.path.exists(PEOPLE_FILE_PATH):
        return {}

    with open(PEOPLE_FILE_PATH, 'r') as people_file:
        people = json.load(people_file)
    verify_people(people)

    return people


def save_people(people):
    verify_people(people)

    with open(PEOPLE_FILE_PATH, 'w') as people_file:
        people = json.dump(people, people_file)


def add_person(people, name, descriptors):
    verify_name(name)
    verify_descriptors(descriptors)

    if name in people:
        if 'face' in descriptors:
            people[name]['face'] = descriptors['face']
        if 'voice' in descriptors:
            people[name]['voice'] = descriptors['voice']
    else:
        people[name] = descriptors
