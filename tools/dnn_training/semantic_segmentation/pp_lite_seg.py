import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.stdc import ConvBatchNormReLU


# Inspired by : https://github.com/PaddlePaddle/PaddleSeg/blob/366bb24fc6f7187bf6c9b71d09957f6ef2e7156e/paddleseg/models/pp_liteseg.py
class SimplePyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, pool_sizes):
        super(SimplePyramidPoolingModule, self).__init__()

        self._stages = nn.ModuleList()
        for pool_size in pool_sizes:
            self._stages.append(nn.Sequential(nn.AdaptiveAvgPool2d(pool_size),
                                              ConvBatchNormReLU(in_channels, mid_channels, kernel_size=1, stride=1)))

        self._last_conv = ConvBatchNormReLU(mid_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x):
        input_size = x.size()[2:]
        accumulator = None

        for stage in self._stages:
            y = stage(x)
            y = F.interpolate(y, input_size, mode='bilinear', align_corners=False)
            if accumulator is None:
                accumulator = y
            else:
                accumulator = accumulator + y

        return self._last_conv(accumulator)


class UnifiedAttentionFusionModule(nn.Module):
    def __init__(self, low_level_in_channels, high_level_in_channels, out_channels, kernel_size):
        super(UnifiedAttentionFusionModule, self).__init__()

        self._low_level_conv = ConvBatchNormReLU(low_level_in_channels, high_level_in_channels, kernel_size, stride=1)
        self._last_conv = ConvBatchNormReLU(high_level_in_channels, out_channels, kernel_size, stride=1)
        self._attention = MeanMaxSpatialAttention()

    def forward(self, low_level_features, high_level_features):
        low_level_features = self._low_level_conv(low_level_features)
        high_level_features = F.interpolate(high_level_features, low_level_features.size()[2:], mode='bilinear')
        fused_features = self._fuse(low_level_features, high_level_features)
        return self._last_conv(fused_features)

    def _fuse(self, low_level_features, high_level_features):
        attention = self._attention(low_level_features, high_level_features)

        return low_level_features * attention + high_level_features * (1 - attention)


class MeanMaxSpatialAttention(nn.Module):
    def __init__(self):
        super(MeanMaxSpatialAttention, self).__init__()

        self._layers = nn.Sequential(
            ConvBatchNormReLU(in_channels=4, out_channels=2, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
        )

    def forward(self, low_level_features, high_level_features):
        features = torch.cat([torch.mean(low_level_features, dim=1, keepdim=True),
                              torch.max(low_level_features, dim=1, keepdim=True)[0],
                              torch.mean(high_level_features, dim=1, keepdim=True),
                              torch.max(high_level_features, dim=1, keepdim=True)[0]], dim=1)
        return self._layers(features)


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, mid_channels, class_count):
        super(SegmentationHead, self).__init__()

        self._layers = nn.Sequential(
            ConvBatchNormReLU(in_channels, mid_channels, kernel_size=3, stride=1),
            nn.Conv2d(mid_channels, class_count, kernel_size=1, bias=False)
        )

    def forward(self, x, input_size):
        return F.interpolate(self._layers(x), input_size, mode='bilinear')


class PpLiteSeg(nn.Module):
    def __init__(self, backbone, class_count, channel_scale=1):
        '''
        :param backbone: Stdc1 or Stdc2
        '''
        super(PpLiteSeg, self).__init__()
        self._class_count = class_count

        self._backbone = backbone
        self._sppm = SimplePyramidPoolingModule(1024, 128 * channel_scale, 128 * channel_scale, [1, 2, 4])
        self._uafm1 = UnifiedAttentionFusionModule(1024, 128 * channel_scale, 128 * channel_scale, kernel_size=3)
        self._uafm2 = UnifiedAttentionFusionModule(512, 128 * channel_scale, 96 * channel_scale, kernel_size=3)
        self._uafm3 = UnifiedAttentionFusionModule(256, 96 * channel_scale, 64 * channel_scale, kernel_size=3)

        self._head1 = SegmentationHead(128 * channel_scale, 64 * channel_scale, class_count)
        self._head2 = SegmentationHead(96 * channel_scale, 64 * channel_scale, class_count)
        self._head3 = SegmentationHead(64 * channel_scale, 64 * channel_scale, class_count)

    def forward(self, x):
        input_size = x.size()[2:]

        _, _, features1, features2, features3 = self._backbone.forward_features(x)
        features4 = self._sppm(features3)

        y1 = self._uafm1(features3, features4)
        y2 = self._uafm2(features2, y1)
        y3 = self._uafm3(features1, y2)

        if self.training:
            segmentation1 = self._head1(y1, input_size)
            segmentation2 = self._head2(y2, input_size)
            segmentation3 = self._head3(y3, input_size)
            return [segmentation1, segmentation2, segmentation3]
        else:
            segmentation = self._head3(y3, input_size)
            return [segmentation]

    def get_class_count(self):
        return self._class_count
