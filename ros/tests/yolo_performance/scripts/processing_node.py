#!/usr/bin/env python3

import os
import json

import cv2
import torch

from tqdm import tqdm

import rospy

from dnn_utils import Yolo


PERSON_CLASS = 'person'


class Performance:
    def __init__(self, classes):
        self._eps = 1e-9
        self.tp_by_class = {c: 0 for c in classes}
        self.fp_by_class = {c: 0 for c in classes}
        self.fn_by_class = {c: 0 for c in classes}
        self.tn_by_class = {c: 0 for c in classes}

    def precision_by_class(self):
        precision_by_class = {}
        for c in self.tp_by_class.keys():
            precision_by_class[c] = self.tp_by_class[c] / (self.tp_by_class[c] + self.fp_by_class[c] + self._eps)
        return precision_by_class

    def recall_by_class(self):
        recall_by_class = {}
        for c in self.tp_by_class.keys():
            recall_by_class[c] = self.tp_by_class[c] / (self.tp_by_class[c] + self.fn_by_class[c] + self._eps)
        return recall_by_class

    def accuracy_by_class(self):
        accuracy_by_class = {}
        for c in self.tp_by_class.keys():
            accuracy_by_class[c] = (self.tp_by_class[c] + self.tn_by_class[c]) / (self.tp_by_class[c] + self.tn_by_class[c] + self.fp_by_class[c] + self.fn_by_class[c] + self._eps)
        return accuracy_by_class

    def print(self, title):
        precision_by_class = self.precision_by_class()
        recall_by_class = self.recall_by_class()
        accuracy_by_class = self.accuracy_by_class()

        print(f'------- {title} -------')
        for c in precision_by_class.keys():
            print(f'\t{c}:')
            print(f'\t\tPrecision={precision_by_class[c]}')
            print(f'\t\tRecall={recall_by_class[c]}')
            print(f'\t\tAccuracy={accuracy_by_class[c]}')
            print()


def main():
    rospy.init_node('processing_node')

    input_path = rospy.get_param('~input_path')
    output_path = os.path.join(input_path, '..', 'results')
    yolo_models = rospy.get_param('~yolo_models')
    confidence_threshold = rospy.get_param('~confidence_threshold')
    nms_threshold = rospy.get_param('~nms_threshold')
    inference_type = rospy.get_param('~neural_network_inference_type')

    setups, classes = _load_setups_classes(input_path)
    performances_by_setup_by_model = _init_performances(yolo_models, setups, classes)
    models_by_name = _load_models(yolo_models, confidence_threshold, nms_threshold, inference_type)

    _process_all_images(input_path, output_path, performances_by_setup_by_model, classes, models_by_name)

    for m in sorted(yolo_models):
        for s in sorted(setups):
            performances_by_setup_by_model[s][m].print(f'{m} - {s}')


def _load_setups_classes(input_path):
    setups = os.listdir(input_path)
    classes = os.listdir(os.path.join(input_path, setups[0]))

    return setups, classes


def _init_performances(models, setups, classes):
    performances_by_setup_by_model = {}
    for s in setups:
        performances_by_setup_by_model[s] = {}
        for m in models:
            performances_by_setup_by_model[s][m] = Performance(classes)

    return performances_by_setup_by_model


def _load_models(yolo_models, confidence_threshold, nms_threshold, inference_type):
    models_by_name = {}
    for m in yolo_models:
        models_by_name[m] = Yolo(m, confidence_threshold=confidence_threshold,
                                 nms_threshold=nms_threshold, inference_type=inference_type)

    return models_by_name


def _process_all_images(input_path, output_path, performances_by_setup_by_model, classes, models_by_name):
    for s in sorted(performances_by_setup_by_model.keys()):
        for m in sorted(performances_by_setup_by_model[s].keys()):
            for c in classes:
                print(f'Processing {c} - {s} images for {m}', flush=True)
                image_files = os.listdir(os.path.join(input_path, s, c))
                input_image_paths = [os.path.join(input_path, s, c, f) for f in image_files]
                output_image_paths = [os.path.join(output_path, m, s, c, f) for f in image_files]
                r = _process_class_images(input_image_paths, output_image_paths, performances_by_setup_by_model[s][m], c, models_by_name[m])


def _process_class_images(input_image_paths, output_image_paths, performances, expected_class, model):
    for input_path, output_path in tqdm(list(zip(input_image_paths, output_image_paths))):
        bgr_image = cv2.imread(input_path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        tensor = _convert_color_image_to_tensor(rgb_image)

        objects = model(tensor.to(model.device()))
        _compute_stats(objects, model.get_class_names(), performances, expected_class)
        _draw_objects(bgr_image, objects, model.get_class_names(), expected_class, performances.tp_by_class.keys())

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, bgr_image)


def _convert_color_image_to_tensor(rgb_image):
        return torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255


def _compute_stats(objects, model_class_tags, performances, expected_class):
    detected_classes = [model_class_tags[o.class_index] for o in objects]
    expected_class_count = detected_classes.count(expected_class)
    person_count = detected_classes.count(PERSON_CLASS)

    if expected_class_count == 0:
        performances.fn_by_class[expected_class] += 1
    else:
        performances.tp_by_class[expected_class] += 1
        performances.fp_by_class[expected_class] += expected_class_count - 1

    if expected_class_count != PERSON_CLASS and person_count > 1 and PERSON_CLASS in performances.fp_by_class:
        performances.fp_by_class[PERSON_CLASS] += person_count - 1

    for c in detected_classes:
        if c == expected_class or c == PERSON_CLASS or c not in performances.fp_by_class:
            continue
        performances.fp_by_class[c] += 1

    tn_excluding_classes = set(detected_classes + [PERSON_CLASS, expected_class])
    for c in performances.tn_by_class:
        if c in tn_excluding_classes:
            continue
        performances.tn_by_class[c] += 1


def _draw_objects(bgr_image, objects, model_class_tags, expected_class, valid_classes):
    valid_classes = set(valid_classes)
    for object in objects:
        c = model_class_tags[object.class_index]
        if c not in valid_classes:
            continue

        x0, y0, x1, y1 = _get_bbox(object)

        if c == expected_class:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(bgr_image, (x0, y0), (x1, y1), color, thickness=4)
        text = f'{c}({object.confidence:.2f}, {object.class_probabilities[object.class_index]:.2f})'
        cv2.putText(bgr_image, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=color, thickness=3)


def _get_bbox(object):
        x0 = int(object.center_x - object.width / 2)
        y0 = int(object.center_y - object.height / 2)
        x1 = int(object.center_x + object.width / 2)
        y1 = int(object.center_y + object.height / 2)
        return x0, y0, x1, y1


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
