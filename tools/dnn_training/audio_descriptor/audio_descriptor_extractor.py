import torch.nn as nn

from common.modules import L2Normalization, GlobalAvgPool2d, AmSoftmaxLinear


class AudioDescriptorExtractor(nn.Module):
    def __init__(self, backbone, embedding_size=128, class_count=None, am_softmax_linear=False):
        super(AudioDescriptorExtractor, self).__init__()

        self._backbone = backbone
        self._global_avg_pool2d = GlobalAvgPool2d()

        self._descriptor_layers = nn.Sequential(
            nn.Linear(backbone.last_channel_count(), embedding_size),
            L2Normalization()
        )

        if class_count is not None:
            if am_softmax_linear:
                self._classifier = AmSoftmaxLinear(embedding_size, class_count)
            else:
                self._classifier = nn.Linear(embedding_size, class_count)
        else:
            self._classifier = None

        self._class_count = class_count

    def class_count(self):
        return self._class_count

    def forward(self, x):
        features = self._backbone(x)
        features = self._global_avg_pool2d(features)

        descriptor = self._descriptor_layers(features.view(x.size()[0], -1))
        if self._classifier is not None:
            class_scores = self._classifier(descriptor)
            return descriptor, class_scores
        else:
            return descriptor
