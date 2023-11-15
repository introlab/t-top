import os
import sys

import torch
import torch.nn.functional as F
import torchaudio

from dnn_utils.dnn_model import PACKAGE_PATH, DnnModel
from dnn_utils.yolo import COCO_CLASS_NAMES, OBJECTS365_CLASS_NAMES

sys.path.append(os.path.join(PACKAGE_PATH, '..', '..', '..', 'tools', 'dnn_training'))
from object_detection.modules.yolo_layer import X_INDEX, Y_INDEX, W_INDEX, H_INDEX, CONFIDENCE_INDEX, CLASSES_INDEX
from object_detection.filter_yolo_predictions import group_predictions, filter_yolo_predictions


IMAGE_SIZE_BY_MODEL_NAME = {
    'yolo_v4_tiny_coco' : (416, 416),
    'yolo_v7_coco' : (640, 640),
    'yolo_v7_objects365' : (640, 640),
}

IN_CHANNELS = 3

CLASS_NAMES_BY_MODEL_NAME = {
    'yolo_v4_tiny_coco' : COCO_CLASS_NAMES,
    'yolo_v7_coco' : COCO_CLASS_NAMES,
    'yolo_v7_objects365' : OBJECTS365_CLASS_NAMES,
}


class DescriptorYoloPrediction:
    def __init__(self, prediction_tensor, scale, class_count):
        self.center_x = (prediction_tensor[X_INDEX] / scale).item()
        self.center_y = (prediction_tensor[Y_INDEX] / scale).item()
        self.width = (prediction_tensor[W_INDEX] / scale).item()
        self.height = (prediction_tensor[H_INDEX] / scale).item()
        self.confidence = prediction_tensor[CONFIDENCE_INDEX].item()

        class_probabilities =  F.softmax(prediction_tensor[CLASSES_INDEX:CLASSES_INDEX + class_count], dim=0)
        self.class_index = torch.argmax(class_probabilities, dim=0).item()
        self.class_probabilities = class_probabilities.tolist()

        self.descriptor = prediction_tensor[CLASSES_INDEX + class_count:].tolist()


class DescriptorYolo(DnnModel):
    def __init__(self, model_name, confidence_threshold=0.99, nms_threshold=0.5, inference_type=None):
        if model_name not in IMAGE_SIZE_BY_MODEL_NAME:
            raise ValueError(f'Invalid model name ({model_name})')

        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold

        self._image_size = IMAGE_SIZE_BY_MODEL_NAME[model_name]

        torch_script_model_path = os.path.join(PACKAGE_PATH, 'models', f'descriptor_{model_name}.ts.pth')
        tensor_rt_model_path = os.path.join(PACKAGE_PATH, 'models', f'descriptor_{model_name}.trt.pth')
        sample_input = torch.ones(1, 3, self._image_size[0], self._image_size[1])

        super(DescriptorYolo, self).__init__(torch_script_model_path, tensor_rt_model_path, sample_input,
                                             inference_type=inference_type)
        self._padded_image = torch.ones(1, 3, self._image_size[0], self._image_size[1]).to(self._device)

    def get_supported_image_size(self):
        return IMAGE_SIZE_BY_MODEL_NAME[self._model_name]

    def get_class_names(self):
        return CLASS_NAMES_BY_MODEL_NAME[self._model_name]

    def __call__(self, image_tensor):
        with torch.no_grad():
            scale, predictions = self.forward_raw(image_tensor.to(self._device))
            predictions = group_predictions(predictions)[0]
            predictions = filter_yolo_predictions(predictions,
                                                  confidence_threshold=self._confidence_threshold,
                                                  nms_threshold=self._nms_threshold)

            class_count = len(CLASS_NAMES_BY_MODEL_NAME[self._model_name])
            return [DescriptorYoloPrediction(prediction.cpu(), scale, class_count) for prediction in predictions]

    def forward_raw(self, image_tensor):
        scale = self._set_image(image_tensor.to(self._device).unsqueeze(0))
        predictions = super(DescriptorYolo, self).__call__(self._padded_image)
        return scale, predictions

    def _set_image(self, image_tensor):
        scale = min(self._padded_image.size()[2] / image_tensor.size()[2], self._padded_image.size()[3] / image_tensor.size()[3])
        output_size = ((int(image_tensor.size()[2] * scale), int(image_tensor.size()[3] * scale)))

        image_tensor = F.interpolate(image_tensor, size=output_size, mode='bilinear')
        self._padded_image[:, :, :image_tensor.size()[2], :image_tensor.size()[3]] = image_tensor
        self._padded_image[:, :, image_tensor.size()[2]:, image_tensor.size()[3]:] = 0.44705882352
        return scale
