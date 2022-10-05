import random
from PIL import Image

import torch
import torch.nn as nn

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from common.datasets import RandomSharpnessChange, RandomAutocontrast, RandomEqualize, RandomPosterize


def _random_scale_translation(output_image_size, input_image, input_target, min_scale=0.5):
    if min_scale >= 1.0 or min_scale <= 0.0:
        raise ValueError('min_scale must be between 0 and 1')

    output_image_width = output_image_size[1]
    output_image_height = output_image_size[0]

    background_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    output_image = Image.new('RGB', (output_image_width, output_image_height), background_color)

    base_scale = min(output_image_width, output_image_height) / max(input_image.width, input_image.height)
    scale = (random.random() * (1 - min_scale) + min_scale) * base_scale

    scaled_image_width = int(input_image.width * scale)
    scaled_image_height = int(input_image.height * scale)
    scaled_image = input_image.resize((scaled_image_width, scaled_image_height), Image.BILINEAR)

    offset_x = random.randint(0, output_image_width - scaled_image_width)
    offset_y = random.randint(0, output_image_height - scaled_image_height)
    output_image.paste(scaled_image, (offset_x, offset_y))

    output_target = []
    for input_mask, class_index in input_target:
        output_mask = Image.new('1', (output_image_size[1], output_image_size[0]))
        scaled_mask = input_mask.resize((scaled_image_width, scaled_image_height), Image.BILINEAR)
        output_mask.paste(scaled_mask, (offset_x, offset_y))
        output_target.append((output_mask, class_index))

    return output_image, output_target


class SemanticSegmentationTransforms(nn.Module):
    def __init__(self, image_size):
        super(SemanticSegmentationTransforms, self).__init__()

        self._image_size = image_size
        self._resize_transform = transforms.Resize(image_size)

    def _target_to_tensor(self, target):
        target_tensor = torch.zeros(self._image_size[0], self._image_size[1], dtype=torch.long)

        for mask, class_index in target:
            mask_tensor = F.to_tensor(mask)[0]
            target_tensor[mask_tensor >= 0.5] = class_index

        return target_tensor


class SemanticSegmentationTrainingTransforms(SemanticSegmentationTransforms):
    def __init__(self, image_size):
        super(SemanticSegmentationTrainingTransforms, self).__init__(image_size)

        self._image_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, hue=0.2),
            transforms.RandomGrayscale(p=0.1),
            RandomSharpnessChange(),
            RandomAutocontrast(),
            RandomEqualize(),
            RandomPosterize(),
        ])

        self._horizontal_flip_p = 0.5
        self._scale_translation_p = 0.25

    def forward(self, image, target):
        if random.random() < self._scale_translation_p:
            image = self._image_transform(image)
            image, target = _random_scale_translation(self._image_size, image, target)
        else:
            image = self._image_transform(self._resize_transform(image))
            target = [(self._resize_transform(mask), class_index) for mask, class_index in target]

        if random.random() < self._horizontal_flip_p:
            image = F.hflip(image)
            target = [(F.hflip(mask), class_index) for mask, class_index in target]

        image = F.equalize(image)
        image_tensor = F.to_tensor(image)
        target_tensor = self._target_to_tensor(target)

        return image_tensor, target_tensor, {}


class SemanticSegmentationValidationTransforms(SemanticSegmentationTransforms):
    def forward(self, image, target):
        image = self._resize_transform(image)
        target = [(self._resize_transform(mask), class_index) for mask, class_index in target]

        image = F.equalize(image)
        image_tensor = F.to_tensor(image)
        target_tensor = self._target_to_tensor(target)

        return image_tensor, target_tensor, {}
