import os
import sys

import torch
import torch.nn.functional as F
import torchaudio

from dnn_utils.dnn_model import PACKAGE_PATH, DnnModel
from dnn_utils.yolo_v4 import YoloV4Prediction

sys.path.append(os.path.join(PACKAGE_PATH, '..', '..', '..', 'tools', 'dnn_training'))
from object_detection.modules.yolo_layer import X_INDEX, Y_INDEX, W_INDEX, H_INDEX, CONFIDENCE_INDEX, CLASSES_INDEX
from object_detection.filter_yolo_predictions import group_predictions, filter_yolo_predictions


IMAGE_SIZE = (608, 608)
IN_CHANNELS = 3
CLASS_COUNT = 80


class YoloV4Best(DnnModel):
    def __init__(self, confidence_threshold=0.99, nms_threshold=0.5, inference_type=None):
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold

        torch_script_model_path = os.path.join(PACKAGE_PATH, 'models', 'yolo_v4_best.ts.pth')
        tensor_rt_model_path = os.path.join(PACKAGE_PATH, 'models', 'yolo_v4_best.trt.pth')
        sample_input = torch.ones(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])

        super(YoloV4Best, self).__init__(torch_script_model_path, tensor_rt_model_path, sample_input,
                                               inference_type=inference_type)
        self._padded_image = torch.ones(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(self._device)

    def get_supported_image_size(self):
        return IMAGE_SIZE

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
            scale, predictions = self.forward_raw(image_tensor.to(self._device))
            predictions = group_predictions(predictions)[0]
            predictions = filter_yolo_predictions(predictions,
                                                  confidence_threshold=self._confidence_threshold,
                                                  nms_threshold=self._nms_threshold)

            return [YoloV4Prediction(prediction.cpu(), scale) for prediction in predictions]

    def forward_raw(self, image_tensor):
        scale = self._set_image(image_tensor.to(self._device).unsqueeze(0))
        predictions = super(YoloV4, self).__call__(self._padded_image)
        return scale, predictions

    def _set_image(self, image_tensor):
        scale = min(self._padded_image.size()[2] / image_tensor.size()[2], self._padded_image.size()[3] / image_tensor.size()[3])
        output_size = ((int(image_tensor.size()[2] * scale), int(image_tensor.size()[3] * scale)))

        image_tensor = F.interpolate(image_tensor, size=output_size, mode='bilinear')
        self._padded_image[:, :, :image_tensor.size()[2], :image_tensor.size()[3]] = image_tensor
        self._padded_image[:, :, image_tensor.size()[2]:, image_tensor.size()[3]:] = 128
        return scale
