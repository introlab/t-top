import torch.nn as nn

from common.modules import L2Normalization, GlobalAvgPool2d, AmSoftmaxLinear, NetVLAD


class AudioDescriptorExtractor(nn.Module):
    def __init__(self, backbone, embedding_size=128, class_count=None, am_softmax_linear=False):
        super(AudioDescriptorExtractor, self).__init__()

        self._backbone = backbone
        self._global_avg_pool2d = GlobalAvgPool2d()

        self._descriptor_layers = nn.Sequential(
            nn.Linear(backbone.last_channel_count(), embedding_size),
            L2Normalization()
        )

        self._classifier = _create_classifier(embedding_size, class_count, am_softmax_linear)
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


class AudioDescriptorExtractorVLAD(nn.Module):
    def __init__(self, backbone, embedding_size=128, class_count=None, am_softmax_linear=False):
        super(AudioDescriptorExtractorVLAD, self).__init__()

        self._backbone = backbone

        cluster_count = 8
        D = embedding_size // cluster_count
        self._conv = nn.Conv2d(backbone.last_channel_count(), D, kernel_size=1)
        self._vlad = NetVLAD(D, cluster_count, ghost_cluster_count=2)

        self._classifier = _create_classifier(embedding_size, class_count, am_softmax_linear)
        self._class_count = class_count

    def class_count(self):
        return self._class_count

    def forward(self, x):
        features = self._backbone(x)
        features = self._conv(features)

        descriptor = self._vlad(features)
        if self._classifier is not None:
            class_scores = self._classifier(descriptor)
            return descriptor, class_scores
        else:
            return descriptor


def _create_classifier(embedding_size, class_count, am_softmax_linear):
    if class_count is not None:
        if am_softmax_linear:
            return AmSoftmaxLinear(embedding_size, class_count)
        else:
            return nn.Linear(embedding_size, class_count)
    else:
        return None
