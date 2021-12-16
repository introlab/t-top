import torch.nn as nn

from common.modules import L2Normalization

from common.modules import InceptionModule, PaddedLPPool2d, Lrn2d


# Based on OpenFace (https://cmusatyalab.github.io/openface/)
class FaceDescriptorExtractor(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceDescriptorExtractor, self).__init__()

        self._features_layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, stride=2, padding=1),
            Lrn2d(5, alpha=0.0001, beta=0.75),

            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            Lrn2d(5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(3, stride=2, padding=1),

            InceptionModule(in_channels=192,
                            kernel_size=[3, 5],
                            kernel_stride=[1, 1],
                            out_channels=[128, 32],
                            reduce_size=[96, 16, 32, 64],
                            pool=nn.MaxPool2d(3, stride=1, padding=1)),
            InceptionModule(in_channels=256,
                            kernel_size=[3, 5],
                            kernel_stride=[1, 1],
                            out_channels=[128, 64],
                            reduce_size=[96, 32, 64, 64],
                            pool=PaddedLPPool2d(2, kernel_size=3, stride=1, padding=1)),
            InceptionModule(in_channels=320,
                            kernel_size=[3, 5],
                            kernel_stride=[2, 2],
                            out_channels=[256, 64],
                            reduce_size=[128, 32, None, None],
                            pool=nn.MaxPool2d(3, stride=2, padding=1)),

            InceptionModule(in_channels=640,
                            kernel_size=[3, 5],
                            kernel_stride=[1, 1],
                            out_channels=[192, 64],
                            reduce_size=[96, 32, 128, 256],
                            pool=PaddedLPPool2d(2, kernel_size=3, stride=1, padding=1)),
            InceptionModule(in_channels=640,
                            kernel_size=[3, 5],
                            kernel_stride=[2, 2],
                            out_channels=[256, 128],
                            reduce_size=[160, 64, None, None],
                            pool=nn.MaxPool2d(3, stride=2, padding=1)),

            InceptionModule(in_channels=1024,
                            kernel_size=[3],
                            kernel_stride=[1],
                            out_channels=[384],
                            reduce_size=[96, 96, 256],
                            pool=PaddedLPPool2d(2, kernel_size=3, stride=1, padding=1)),
            InceptionModule(in_channels=736,
                            kernel_size=[3],
                            kernel_stride=[1],
                            out_channels=[384],
                            reduce_size=[96, 96, 256],
                            pool=nn.MaxPool2d(3, stride=1, padding=1)),

            nn.AvgPool2d(3, stride=(2, 1))
        )

        self._descriptor_layers = nn.Sequential(
            nn.Linear(736, embedding_size),
            L2Normalization()
        )

    def forward(self, x):
        features = self._features_layers(x)
        descriptor = self._descriptor_layers(features.view(x.size()[0], -1))
        return descriptor
