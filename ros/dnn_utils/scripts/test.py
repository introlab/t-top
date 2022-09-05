#!/usr/bin/env python3

import traceback

import numpy as np

import torch

import rospy

from dnn_utils import DescriptorYoloV4, YoloV4, PoseEstimator, FaceDescriptorExtractor
from dnn_utils import MulticlassAudioDescriptorExtractor, VoiceDescriptorExtractor, TTopKeywordSpotter


def mean_abs_diff(a, b):
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)

    return torch.mean(torch.abs(a.cpu() - b.cpu())).item()


def launch_test(function):
    try:
        function()
    except Exception as e:
        print('Test error: {} \n {}'.format(e, traceback.format_exc()))

    print()


def test_descriptor_yolo_v4():
    print('----------test_descriptor_yolo_v4----------')

    cpu_model = DescriptorYoloV4(inference_type='cpu')
    torch_gpu_model = DescriptorYoloV4(inference_type='torch_gpu')
    trt_gpu_model = DescriptorYoloV4(inference_type='trt_gpu')

    IMAGE_SIZE = cpu_model.get_supported_image_size()
    x = torch.rand(3, IMAGE_SIZE[0], IMAGE_SIZE[1])

    _, cpu_predictions =  cpu_model.forward_raw(x)
    _, torch_gpu_predictions = torch_gpu_model.forward_raw(x)
    _, trt_gpu_predictions = trt_gpu_model.forward_raw(x)

    for i in range(len(cpu_predictions)):
        print('mean(abs(cpu_predictions[{}] - torch_gpu_predictions[{}])) ='.format(i, i),
            mean_abs_diff(cpu_predictions[i], torch_gpu_predictions[i]))
        print('mean(abs(cpu_predictions[{}] - trt_gpu_predictions[{}])) ='.format(i, i),
            mean_abs_diff(cpu_predictions[i], trt_gpu_predictions[i]))


def test_yolo_v4():
    print('----------test_yolo_v4----------')

    cpu_model = YoloV4(inference_type='cpu')
    torch_gpu_model = YoloV4(inference_type='torch_gpu')
    trt_gpu_model = YoloV4(inference_type='trt_gpu')

    IMAGE_SIZE = cpu_model.get_supported_image_size()
    x = torch.rand(3, IMAGE_SIZE[0], IMAGE_SIZE[1])

    _, cpu_predictions =  cpu_model.forward_raw(x)
    _, torch_gpu_predictions = torch_gpu_model.forward_raw(x)
    _, trt_gpu_predictions = trt_gpu_model.forward_raw(x)

    for i in range(len(cpu_predictions)):
        print('mean(abs(cpu_predictions[{}] - torch_gpu_predictions[{}])) ='.format(i, i),
            mean_abs_diff(cpu_predictions[i], torch_gpu_predictions[i]))
        print('mean(abs(cpu_predictions[{}] - trt_gpu_predictions[{}])) ='.format(i, i),
            mean_abs_diff(cpu_predictions[i], trt_gpu_predictions[i]))


def test_pose_estimator():
    print('----------test_pose_estimator----------')

    cpu_model = PoseEstimator(inference_type='cpu')
    torch_gpu_model = PoseEstimator(inference_type='torch_gpu')
    trt_gpu_model = PoseEstimator(inference_type='trt_gpu')

    IMAGE_SIZE = cpu_model.get_supported_image_size()
    x = torch.rand(3, IMAGE_SIZE[0], IMAGE_SIZE[1])

    cpu_scaled_coordinates, cpu_presence = cpu_model(x)
    torch_gpu_scaled_coordinates, torch_gpu_presence = torch_gpu_model(x)
    trt_gpu_scaled_coordinates, trt_gpu_presence = trt_gpu_model(x)

    print('mean(abs(cpu_scaled_coordinates - torch_gpu_scaled_coordinates)) =',
          mean_abs_diff(cpu_scaled_coordinates, torch_gpu_scaled_coordinates))
    print('mean(abs(cpu_scaled_coordinates - trt_gpu_scaled_coordinates)) =',
          mean_abs_diff(cpu_scaled_coordinates, trt_gpu_scaled_coordinates))
    print('mean(abs(cpu_presence - torch_gpu_presence)) =',
          mean_abs_diff(cpu_presence, torch_gpu_presence))
    print('mean(abs(cpu_presence - trt_gpu_presence)) =',
          mean_abs_diff(cpu_presence, trt_gpu_presence))


def test_face_descriptor_extractor():
    print('----------test_face_descriptor_extractor----------')

    cpu_model = FaceDescriptorExtractor(inference_type='cpu')
    torch_gpu_model = FaceDescriptorExtractor(inference_type='torch_gpu')
    trt_gpu_model = FaceDescriptorExtractor(inference_type='trt_gpu')

    IMAGE_SIZE = cpu_model.get_supported_image_size()
    x = torch.rand(3, IMAGE_SIZE[0], IMAGE_SIZE[1])

    pose_coordinates = np.array([[0.5 * IMAGE_SIZE[1], 0.5144414 * IMAGE_SIZE[0]],
                                 [0.75 * IMAGE_SIZE[1], 0.25 * IMAGE_SIZE[0]],
                                 [0.25 * IMAGE_SIZE[1], 0.25 * IMAGE_SIZE[0]]])
    pose_presence = np.array([1.0, 1.0, 1.0, 0.0, 0.0])

    cpu_descriptor = cpu_model(x, pose_coordinates, pose_presence)[0]
    torch_gpu_descriptor = torch_gpu_model(x, pose_coordinates, pose_presence)[0]
    trt_gpu_descriptor = trt_gpu_model(x, pose_coordinates, pose_presence)[0]

    print('mean(abs(cpu_descriptor - torch_gpu_descriptor)) =',
          mean_abs_diff(cpu_descriptor, torch_gpu_descriptor))
    print('mean(abs(cpu_descriptor - trt_gpu_descriptor)) =',
          mean_abs_diff(cpu_descriptor, trt_gpu_descriptor))


def test_multiclass_audio_descriptor_extractor():
    print('----------test_multiclass_audio_descriptor_extractor----------')

    cpu_model = MulticlassAudioDescriptorExtractor(inference_type='cpu')
    torch_gpu_model = MulticlassAudioDescriptorExtractor(inference_type='torch_gpu')
    trt_gpu_model = MulticlassAudioDescriptorExtractor(inference_type='trt_gpu')

    x = torch.randn(cpu_model.get_supported_duration(), dtype=torch.float32)
    cpu_descriptor, cpu_class_probabilities = cpu_model(x)
    torch_gpu_descriptor, torch_gpu_class_probabilities = torch_gpu_model(x)
    trt_gpu_descriptor, trt_gpu_class_probabilities = trt_gpu_model(x)

    print('mean(abs(cpu_descriptor - torch_gpu_descriptor)) =',
          mean_abs_diff(cpu_descriptor, torch_gpu_descriptor))
    print('mean(abs(cpu_descriptor - trt_gpu_descriptor)) =',
          mean_abs_diff(cpu_descriptor, trt_gpu_descriptor))
    print('mean(abs(cpu_class_probabilities - torch_gpu_class_probabilities)) =',
          mean_abs_diff(cpu_class_probabilities, torch_gpu_class_probabilities))
    print('mean(abs(cpu_class_probabilities - trt_gpu_class_probabilities)) =',
          mean_abs_diff(cpu_class_probabilities, trt_gpu_class_probabilities))


def test_voice_descriptor_extractor():
    print('----------test_voice_descriptor_extractor----------')

    cpu_model = VoiceDescriptorExtractor(inference_type='cpu')
    torch_gpu_model = VoiceDescriptorExtractor(inference_type='torch_gpu')
    trt_gpu_model = VoiceDescriptorExtractor(inference_type='trt_gpu')

    x = torch.randn(cpu_model.get_supported_duration(), dtype=torch.float32)
    cpu_descriptor = cpu_model(x)
    torch_gpu_descriptor = torch_gpu_model(x)
    trt_gpu_descriptor = trt_gpu_model(x)

    print('mean(abs(cpu_descriptor - torch_gpu_descriptor)) =',
          mean_abs_diff(cpu_descriptor, torch_gpu_descriptor))
    print('mean(abs(cpu_descriptor - trt_gpu_descriptor)) =',
          mean_abs_diff(cpu_descriptor, trt_gpu_descriptor))


def test_ttop_keyword_spotter():
    print('----------test_ttop_keyword_spotter----------')

    cpu_model = TTopKeywordSpotter(inference_type='cpu')
    torch_gpu_model = TTopKeywordSpotter(inference_type='torch_gpu')
    trt_gpu_model = TTopKeywordSpotter(inference_type='trt_gpu')

    x = torch.randn(cpu_model.get_supported_duration(), dtype=torch.float32)
    cpu_class_probabilities = cpu_model(x)
    torch_gpu_class_probabilities = torch_gpu_model(x)
    trt_gpu_class_probabilities = trt_gpu_model(x)

    print('mean(abs(cpu_class_probabilities - torch_gpu_class_probabilities)) =',
          mean_abs_diff(cpu_class_probabilities, torch_gpu_class_probabilities))
    print('mean(abs(cpu_class_probabilities - trt_gpu_class_probabilities))) =',
          mean_abs_diff(cpu_class_probabilities, trt_gpu_class_probabilities))


def main():
    rospy.init_node('dnn_utils_test')

    launch_test(test_descriptor_yolo_v4)
    launch_test(test_yolo_v4)
    launch_test(test_pose_estimator)
    launch_test(test_face_descriptor_extractor)
    launch_test(test_audio_descriptor_extractor)
    launch_test(test_multiclass_audio_descriptor_extractor)
    launch_test(test_voice_descriptor_extractor)
    launch_test(test_ttop_keyword_spotter)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
