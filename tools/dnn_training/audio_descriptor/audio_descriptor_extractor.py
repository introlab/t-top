import torch.nn as nn

from common.modules import L2Normalization, GlobalAvgPool2d, GlobalHeightAvgPool2d, NormalizedLinear, NetVLAD
from audio_descriptor.modules import SAP, PSLAAttention


class AudioDescriptorExtractor(nn.Module):
    def __init__(self, backbone, embedding_size=128, class_count=None, normalized_linear=False):
        super(AudioDescriptorExtractor, self).__init__()

        self._backbone = backbone
        self._global_avg_pool2d = GlobalAvgPool2d()

        self._descriptor_layers = nn.Sequential(
            nn.Linear(backbone.last_channel_count(), embedding_size),
            L2Normalization()
        )

        self._classifier = _create_classifier(embedding_size, class_count, normalized_linear)
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
    def __init__(self, backbone, embedding_size, class_count=None, normalized_linear=False):
        super(AudioDescriptorExtractorVLAD, self).__init__()

        self._backbone = backbone
        self._global_pooling = GlobalHeightAvgPool2d()

        self._cluster_count = 8
        self._vlad = NetVLAD(backbone.last_channel_count(), self._cluster_count, ghost_cluster_count=2)
        self._output_layers = nn.Sequential(
            nn.Linear(in_features=backbone.last_channel_count() * self._cluster_count,
                      out_features=embedding_size, bias=False),
            L2Normalization()
        )

        self._classifier = _create_classifier(embedding_size, class_count, normalized_linear)
        self._class_count = class_count

    def class_count(self):
        return self._class_count

    def forward(self, x):
        features = self._global_pooling(self._backbone(x))
        full_descriptor = self._vlad(features)
        descriptor = self._output_layers(full_descriptor)
        if self._classifier is not None:
            class_scores = self._classifier(descriptor)
            return descriptor, class_scores
        else:
            return descriptor


class AudioDescriptorExtractorSAP(nn.Module):
    def __init__(self, backbone, embedding_size=128, class_count=None, normalized_linear=False):
        super(AudioDescriptorExtractorSAP, self).__init__()

        self._backbone = backbone
        self._frequency_pooling = GlobalHeightAvgPool2d()
        self._sap = SAP(backbone.last_channel_count())

        self._descriptor_layers = nn.Sequential(
            nn.Linear(backbone.last_channel_count(), embedding_size),
            L2Normalization()
        )

        self._classifier = _create_classifier(embedding_size, class_count, normalized_linear)
        self._class_count = class_count

    def class_count(self):
        return self._class_count

    def forward(self, x):
        features = self._backbone(x)
        features = self._frequency_pooling(features)
        features = self._sap(features)

        descriptor = self._descriptor_layers(features)
        if self._classifier is not None:
            class_scores = self._classifier(descriptor)
            return descriptor, class_scores
        else:
            return descriptor


class AudioDescriptorExtractorPSLA(nn.Module):
    def __init__(self, backbone, embedding_size=128, class_count=None, normalized_linear=False):
        super(AudioDescriptorExtractorPSLA, self).__init__()

        self._backbone = backbone
        self._frequency_pooling = GlobalHeightAvgPool2d()
        self._psla_attention = PSLAAttention(backbone.last_channel_count(), backbone.last_channel_count())

        self._descriptor_layers = nn.Sequential(
            nn.Linear(backbone.last_channel_count(), embedding_size),
            L2Normalization()
        )

        self._classifier = _create_classifier(embedding_size, class_count, normalized_linear)
        self._class_count = class_count

    def class_count(self):
        return self._class_count

    def forward(self, x):
        features = self._backbone(x)
        features = self._frequency_pooling(features)
        features = self._psla_attention(features)

        descriptor = self._descriptor_layers(features)
        if self._classifier is not None:
            class_scores = self._classifier(descriptor)
            return descriptor, class_scores
        else:
            return descriptor


def _create_classifier(embedding_size, class_count, normalized_linear):
    if class_count is not None:
        if normalized_linear:
            return NormalizedLinear(embedding_size, class_count)
        else:
            return nn.Linear(embedding_size, class_count)
    else:
        return None
