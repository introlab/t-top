import torch.nn as nn

from common.modules import L2Normalization

from common.modules import InceptionModule, PaddedLPPool2d, Lrn2d, AmSoftmaxLinear


# Based on OpenFace (https://cmusatyalab.github.io/openface/)
class FaceDescriptorExtractor(nn.Module):
    def __init__(self, embedding_size=128, class_count=None, am_softmax_linear=False):
        super(FaceDescriptorExtractor, self).__init__()

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

        self._descriptor_layers = nn.Sequential(
            nn.Linear(in_features=736, out_features=embedding_size),
            L2Normalization()
        )

        self._class_count = class_count
        if class_count is not None and am_softmax_linear:
            self._classifier = AmSoftmaxLinear(embedding_size, class_count)
        elif class_count is not None:
            self._classifier = nn.Linear(embedding_size, class_count)
        else:
            self._classifier = None

    def class_count(self):
        return self._class_count

    def forward(self, x):
        features = self._features_layers(x)
        descriptor = self._descriptor_layers(features.view(x.size()[0], -1))

        if self._classifier is not None:
            class_scores = self._classifier(descriptor)
            return descriptor, class_scores
        else:
            return descriptor
