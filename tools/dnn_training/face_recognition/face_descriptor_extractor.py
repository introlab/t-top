import torch.nn as nn

import torchvision.models as models

from common.modules import L2Normalization

from common.modules import InceptionModule, PaddedLPPool2d, Lrn2d, NormalizedLinear, GlobalAvgPool2d


class OpenFaceBackbone(nn.Module):
    def __init__(self):
        super(OpenFaceBackbone, self).__init__()

        self._features_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Lrn2d(size=5, alpha=0.0001, beta=0.75),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),

            Lrn2d(size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            InceptionModule(in_channels=192,
                            kernel_size=[3, 5],
                            kernel_stride=[1, 1],
                            out_channels=[128, 32],
                            reduce_size=[96, 16, 32, 64],
                            pool=nn.MaxPool2d(kernel_size=3, stride=1, padding=1)),
            InceptionModule(in_channels=256,
                            kernel_size=[3, 5],
                            kernel_stride=[1, 1],
                            out_channels=[128, 64],
                            reduce_size=[96, 32, 64, 64],
                            pool=PaddedLPPool2d(norm_type=2, kernel_size=3, stride=1, padding=1)),
            InceptionModule(in_channels=320,
                            kernel_size=[3, 5],
                            kernel_stride=[2, 2],
                            out_channels=[256, 64],
                            reduce_size=[128, 32, None, None],
                            pool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

            InceptionModule(in_channels=640,
                            kernel_size=[3, 5],
                            kernel_stride=[1, 1],
                            out_channels=[192, 64],
                            reduce_size=[96, 32, 128, 256],
                            pool=PaddedLPPool2d(norm_type=2, kernel_size=3, stride=1, padding=1)),
            InceptionModule(in_channels=640,
                            kernel_size=[3, 5],
                            kernel_stride=[2, 2],
                            out_channels=[256, 128],
                            reduce_size=[160, 64, None, None],
                            pool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

            InceptionModule(in_channels=1024,
                            kernel_size=[3],
                            kernel_stride=[1],
                            out_channels=[384],
                            reduce_size=[96, 96, 256],
                            pool=PaddedLPPool2d(norm_type=2, kernel_size=3, stride=1, padding=1)),
            InceptionModule(in_channels=736,
                            kernel_size=[3],
                            kernel_stride=[1],
                            out_channels=[384],
                            reduce_size=[96, 96, 256],
                            pool=nn.MaxPool2d(kernel_size=3, stride=1, padding=1)),

            nn.AvgPool2d(kernel_size=3, stride=(2, 1))
        )

    def forward(self, x):
        return self._features_layers(x)

    def last_channel_count(self):
        return 736


class EfficientNetBackbone(nn.Module):
    SUPPORTED_TYPES = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                       'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']
    LAST_CHANNEL_COUNT_BY_TYPE = {'efficientnet_b0': 1280,
                                  'efficientnet_b1': 1280,
                                  'efficientnet_b2': 1408,
                                  'efficientnet_b3': 1536,
                                  'efficientnet_b4': 1792,
                                  'efficientnet_b5': 2048,
                                  'efficientnet_b6': 2304,
                                  'efficientnet_b7': 2560}
    def __init__(self, type, pretrained_backbone=True):
        super(EfficientNetBackbone, self).__init__()

        if pretrained_backbone:
            backbone_weights = 'DEFAULT'
        else:
            backbone_weights = None

        if (type not in self.SUPPORTED_TYPES or type not in self.LAST_CHANNEL_COUNT_BY_TYPE):
            raise ValueError('Invalid backbone type')

        self._features_layers = models.__dict__[type](weights=backbone_weights).features
        self._last_channel_count = self.LAST_CHANNEL_COUNT_BY_TYPE[type]

    def forward(self, x):
        return self._features_layers(x)

    def last_channel_count(self):
        return self._last_channel_count


# Based on OpenFace (https://cmusatyalab.github.io/openface/)
class FaceDescriptorExtractor(nn.Module):
    def __init__(self, backbone, embedding_size=128, class_count=None, normalized_linear=False):
        super(FaceDescriptorExtractor, self).__init__()

        self._backbone = backbone
        self._global_avg_pool = GlobalAvgPool2d()
        self._descriptor_layers = nn.Sequential(
            nn.Linear(in_features=self._backbone.last_channel_count(), out_features=embedding_size),
            L2Normalization()
        )

        self._class_count = class_count
        if class_count is not None and normalized_linear:
            self._classifier = NormalizedLinear(embedding_size, class_count)
        elif class_count is not None:
            self._classifier = nn.Linear(embedding_size, class_count)
        else:
            self._classifier = None

    def class_count(self):
        return self._class_count

    def forward(self, x):
        features = self._global_avg_pool(self._backbone(x))
        descriptor = self._descriptor_layers(features.view(x.size()[0], -1))

        if self._classifier is not None:
            class_scores = self._classifier(descriptor)
            return descriptor, class_scores
        else:
            return descriptor
