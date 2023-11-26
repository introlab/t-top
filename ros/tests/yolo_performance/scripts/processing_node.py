#!/usr/bin/env python3

import os
import json

import cv2
import torch

from tqdm import tqdm

import rospy

from dnn_utils import Yolo


def main():
    rospy.init_node('processing_node')

    input_path = rospy.get_param('~input_path')
    yolo_models = rospy.get_param('~yolo_models')
    confidence_threshold = rospy.get_param('~confidence_threshold')
    nms_threshold = rospy.get_param('~nms_threshold')
    inference_type = rospy.get_param('~neural_network_inference_type')

    classes_by_class_name, paths_by_class_name = _load_image_paths(input_path)
    models_by_name = _load_models(yolo_models, confidence_threshold, nms_threshold, inference_type)

    recall_by_class_name = _process_all_images(classes_by_class_name, paths_by_class_name, models_by_name)
    _show_stats(recall_by_class_name, models_by_name.keys())


def _load_image_paths(input_path):
    class_names = os.listdir(input_path)
    classes_by_class_name = {}
    paths_by_class_name = {}

    for n in class_names:
        with open(os.path.join(input_path, n, 'config.json')) as f:
            classes_by_class_name[n] = (json.load(f)['class'])
        paths_by_class_name[n] = [os.path.join(input_path, n, f) for f in os.listdir(os.path.join(input_path, n)) if f.endswith('.jpg')]

    return classes_by_class_name, paths_by_class_name


def _load_models(yolo_models, confidence_threshold, nms_threshold, inference_type):
    models_by_name = {}
    for m in yolo_models:
        models_by_name[m] = Yolo(m, confidence_threshold=confidence_threshold,
                                 nms_threshold=nms_threshold, inference_type=inference_type)

    return models_by_name


def _process_all_images(classes_by_class_name, paths_by_class_name, models_by_name):
    recall_by_class_name = {}

    for class_name in classes_by_class_name.keys():
        print(f'Processing {class_name} images', flush=True)
        r = _process_class_images(classes_by_class_name[class_name], paths_by_class_name[class_name], models_by_name)
        recall_by_class_name[class_name] = r

    return recall_by_class_name


def _process_class_images(class_tag, paths, models_by_name):
    true_positive_by_model_name = {}
    false_negative_by_model_name = {}

    for model_name in models_by_name.keys():
        true_positive_by_model_name[model_name] = 0
        false_negative_by_model_name[model_name] = 0

    for path in tqdm(paths):
        bgr_image = cv2.imread(path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        tensor = _convert_color_image_to_tensor(rgb_image)

        for model_name, model in models_by_name.items():
            objects = model(tensor.to(model.device()))
            true_positive, false_negative = _compute_stats(objects, model.get_class_names(), class_tag)

            true_positive_by_model_name[model_name] += true_positive
            false_negative_by_model_name[model_name] += false_negative

    recall_by_model_name = {}

    for model_name in models_by_name.keys():
        tp = true_positive_by_model_name[model_name]
        fn = false_negative_by_model_name[model_name]

        recall_by_model_name[model_name] = tp / (tp + fn)

    return recall_by_model_name



def _convert_color_image_to_tensor(rgb_image):
        return torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255


def _compute_stats(objects, model_class_tags, class_tag):
    true_positive = 0
    false_negative = 0

    detected_class_tags = [model_class_tags[o.class_index] for o in objects]

    count = detected_class_tags.count(class_tag)
    if count == 0:
        false_negative = 1
    else:
        true_positive = 1

    return true_positive, false_negative,


def _show_stats(recall_by_class_name, model_names):
    model_names = list(model_names)
    model_names.sort()
    for model_name in model_names:
        print(model_name)
        class_names = list(recall_by_class_name.keys())
        class_names.sort()
        for class_name in class_names:
            print(f'\t{class_name}')
            print(f'\t\tRecall={recall_by_class_name[class_name][model_name]}')


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
