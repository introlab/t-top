import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from dnn_utils.dnn_model import PACKAGE_PATH, DnnModel


IMAGE_SIZE = (270, 480)


class SemanticSegmentationNetwork(DnnModel):
    def __init__(self, inference_type=None, dataset='coco'):
        if dataset not in ['coco', 'kitchen_open_images', 'person_other_open_images']:
            raise ValueError('Invalid semantic segmentation dataset')

        self._dataset = dataset

        torch_script_model_path = os.path.join(PACKAGE_PATH, 'models', f'semantic_segmentation_network_{dataset}.ts.pth')
        tensor_rt_model_path = os.path.join(PACKAGE_PATH, 'models', f'semantic_segmentation_network_{dataset}.trt.pth')
        sample_input = torch.ones((1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))

        super(SemanticSegmentationNetwork, self).__init__(torch_script_model_path, tensor_rt_model_path, sample_input,
                                                          inference_type=inference_type)

    def get_supported_image_size(self):
        return IMAGE_SIZE

    def get_class_names(self):
        if self._dataset == 'coco':
            return ['background', 'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                    'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
                    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        elif self._dataset == 'kitchen_open_images':
            return ['background', 'apple', 'artichoke', 'bagel', 'banana', 'band-aid', 'beer', 'bell pepper', 'book',
                    'bottle', 'bottle opener', 'bowl', 'bread', 'broccoli', 'burrito', 'cabbage', 'cake', 'cantaloupe',
                    'carrot', 'cheese', 'chopsticks', 'cocktail shaker', 'coffee cup', 'common fig', 'cookie', 'croissant',
                    'cucumber', 'doughnut', 'drinking straw', 'food processor', 'frying pan', 'garden Asparagus', 'grape',
                    'grapefruit', 'guacamole', 'hamburger', 'hot dog', 'juice', 'kitchen knife', 'lemon', 'mango', 'milk',
                    'mixing bowl', 'orange', 'pancake', 'person', 'pizza', 'pizza cutter', 'pomegranate', 'popcorn',
                    'potato', 'pressure cooker', 'pretzel', 'sandwich', 'submarine sandwich', 'tomato', 'waffle', 'wine',
                    'winter melon', 'zucchini']
        elif self._dataset == 'person_other_open_images':
            return ['background', 'person', 'other']
        else:
            raise ValueError('Invalid semantic segmentation dataset')

    def __call__(self, equalized_image_tensor):
        with torch.no_grad():
            equalized_image_tensor = F.interpolate(equalized_image_tensor.to(self._device).unsqueeze(0),
                                                   size=IMAGE_SIZE, mode='bilinear')
            prediction = super(SemanticSegmentationNetwork, self).__call__(equalized_image_tensor)[0]
            semantic_segmentation = prediction[0].argmax(dim=0)
            return semantic_segmentation.cpu()
