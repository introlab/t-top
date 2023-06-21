import os
import sys

import torch
import torch.nn.functional as F
import torchaudio

from dnn_utils.dnn_model import PACKAGE_PATH, DnnModel

sys.path.append(os.path.join(PACKAGE_PATH, '..', '..', '..', 'tools', 'dnn_training'))
from object_detection.modules.yolo_layer import X_INDEX, Y_INDEX, W_INDEX, H_INDEX, CONFIDENCE_INDEX, CLASSES_INDEX
from object_detection.filter_yolo_predictions import group_predictions, filter_yolo_predictions_by_classes


IMAGE_SIZE_BY_MODEL_NAME = {
    'yolo_v4' : (608, 608),
    'yolo_v4_tiny' : (416, 416),
    'yolo_v7' : (640, 640),
    'yolo_v7_tiny' : (640, 640),
}
IN_CHANNELS = 3
CLASS_COUNT = 80


class YoloPrediction:
    def __init__(self, prediction_tensor, scale, offset_x, offset_y):
        self.center_x = ((prediction_tensor[X_INDEX] - offset_x) / scale).item()
        self.center_y = ((prediction_tensor[Y_INDEX] - offset_y) / scale).item()
        self.width = (prediction_tensor[W_INDEX] / scale).item()
        self.height = (prediction_tensor[H_INDEX] / scale).item()
        self.confidence = prediction_tensor[CONFIDENCE_INDEX].item()

        class_probabilities =  prediction_tensor[CLASSES_INDEX:CLASSES_INDEX + CLASS_COUNT]
        self.class_index = torch.argmax(class_probabilities, dim=0).item()
        self.class_probabilities = class_probabilities.tolist()

        self.descriptor = []


class Yolo(DnnModel):
    def __init__(self, model_name, confidence_threshold=0.99, nms_threshold=0.5, inference_type=None):
        if model_name not in IMAGE_SIZE_BY_MODEL_NAME:
            raise ValueError(f'Invalid model name ({model_name})')

        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold

        torch_script_model_path = os.path.join(PACKAGE_PATH, 'models', f'{model_name}.ts.pth')
        tensor_rt_model_path = os.path.join(PACKAGE_PATH, 'models', f'{model_name}.trt.pth')
        sample_input = torch.ones(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])

        super(YoloV4, self).__init__(torch_script_model_path, tensor_rt_model_path, sample_input,
                                               inference_type=inference_type)
        self._padded_image = torch.ones(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(self._device)

    def get_supported_image_size(self):
        return IMAGE_SIZE_BY_MODEL_NAME[self._model_name]

    def get_class_names(self):
        return ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __call__(self, image_tensor):
        with torch.no_grad():
            scale, offset_x, offset_y, predictions = self.forward_raw(image_tensor.to(self._device))
            predictions = group_predictions(predictions)[0]
            predictions = filter_yolo_predictions_by_classes(predictions,
                                                             confidence_threshold=self._confidence_threshold,
                                                             nms_threshold=self._nms_threshold)

            return [YoloPrediction(prediction.cpu(), scale, offset_x, offset_y) for prediction in predictions]

    def forward_raw(self, image_tensor):
        scale, offset_x, offset_y = self._set_image(image_tensor.to(self._device).unsqueeze(0))
        predictions = super(YoloV4, self).__call__(self._padded_image)
        return scale, offset_x, offset_y, predictions

    def _set_image(self, image_tensor):
        scale = min(self._padded_image.size()[2] / image_tensor.size()[2], self._padded_image.size()[3] / image_tensor.size()[3])
        output_size = ((int(image_tensor.size()[2] * scale), int(image_tensor.size()[3] * scale)))
        offset_y = int((self._padded_image.size()[2] - output_size[0]) / 2)
        offset_x = int((self._padded_image.size()[3] - output_size[1]) / 2)

        image_tensor = F.interpolate(image_tensor, size=output_size, mode='bilinear')
        self._padded_image[:] = 0.44705882352
        self._padded_image[:, :, offset_y:offset_y + image_tensor.size()[2], offset_x:offset_x + image_tensor.size()[3]] = image_tensor
        return scale, offset_x, offset_y
