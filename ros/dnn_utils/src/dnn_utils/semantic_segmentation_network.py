import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from dnn_utils.dnn_model import PACKAGE_PATH, DnnModel


IMAGE_SIZE = (270, 480)


class SemanticSegmentationNetwork(DnnModel):
    def __init__(self, inference_type=None):
        torch_script_model_path = os.path.join(PACKAGE_PATH, 'models', 'semantic_segmentation_network.ts.pth')
        tensor_rt_model_path = os.path.join(PACKAGE_PATH, 'models', 'semantic_segmentation_network.trt.pth')
        sample_input = torch.ones((1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))

        super(SemanticSegmentationNetwork, self).__init__(torch_script_model_path, tensor_rt_model_path, sample_input,
                                                          inference_type=inference_type)
        self._normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_supported_image_size(self):
        return IMAGE_SIZE

    def get_class_names(self):
        return ['background', 'apple', 'artichoke', 'bagel', 'banana', 'band-aid', 'beer', 'bell pepper', 'book',
                'bottle', 'bottle opener', 'bowl', 'bread', 'broccoli', 'burrito', 'cabbage', 'cake', 'cantaloupe',
                'carrot', 'cheese', 'chopsticks', 'cocktail shaker', 'coffee cup', 'common fig', 'cookie', 'croissant',
                'cucumber', 'doughnut', 'drinking straw', 'food processor', 'frying pan', 'garden Asparagus', 'grape',
                'grapefruit', 'guacamole', 'hamburger', 'hot dog', 'juice', 'kitchen knife', 'lemon', 'mango', 'milk',
                'mixing bowl', 'orange', 'pancake', 'person', 'pizza', 'pizza cutter', 'pomegranate', 'popcorn',
                'potato', 'pressure cooker', 'pretzel', 'sandwich', 'submarine sandwich', 'tomato', 'waffle', 'wine',
                'winter melon', 'zucchini']

    def __call__(self, image_tensor):
        with torch.no_grad():
            image_tensor = F.interpolate(image_tensor.to(self._device).unsqueeze(0), size=IMAGE_SIZE, mode='bilinear')
            image_tensor = self._normalization(image_tensor.squeeze(0))

            prediction = super(SemanticSegmentationNetwork, self).__call__(image_tensor.unsqueeze(0))[0]
            semantic_segmentation = prediction[0].argmax(dim=0)
            return semantic_segmentation.cpu()