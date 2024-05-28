from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn

from common.modules import NormalizedLinear

from object_detection.modules.descriptor_yolo_layer import DescriptorYoloV7Layer
from object_detection.modules.yolo_v7_modules import YoloV7SPPCSPC, RepConv


IMAGE_SIZE = (640, 640)
IN_CHANNELS = 3


# Generated from: yolov7.yaml:
class DescriptorYoloV7(nn.Module):
    def __init__(self, class_count=80, embedding_size=128, class_probs=False):
        super(DescriptorYoloV7, self).__init__()

        self._anchors = []
        self._output_strides = [8, 16, 32]
        self._anchors.append(np.array([(12, 16), (19, 36), (40, 28)]))
        self._anchors.append(np.array([(36, 75), (76, 55), (72, 146)]))
        self._anchors.append(np.array([(142, 110), (192, 243), (459, 401)]))

        self._conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.SiLU(),
        )
        self._conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.SiLU(),
        )
        self._conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.SiLU(),
        )
        self._conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.SiLU(),
        )
        self._conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.SiLU(),
        )
        self._conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.SiLU(),
        )
        self._conv7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.SiLU(),
        )
        self._conv8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.SiLU(),
        )
        self._conv9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.SiLU(),
        )

        self._conv11 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._max_pool12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._conv13 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv14 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv15 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )

        self._conv17 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv18 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv19 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv20 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv21 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv22 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )

        self._conv24 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(512, eps=0.001),
            nn.SiLU(),
        )
        self._max_pool25 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._conv26 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv27 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv28 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )

        self._conv30 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv31 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv32 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv33 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv34 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv35 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )

        self._conv37 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(1024, eps=0.001),
            nn.SiLU(),
        )
        self._max_pool38 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._conv39 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(512, eps=0.001),
            nn.SiLU(),
        )
        self._conv40 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(512, eps=0.001),
            nn.SiLU(),
        )
        self._conv41 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(512, eps=0.001),
            nn.SiLU(),
        )

        self._conv43 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv44 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv45 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv46 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv47 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv48 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )

        self._conv50 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(1024, eps=0.001),
            nn.SiLU(),
        )
        self._sppcspc51 = YoloV7SPPCSPC(1024, 512)
        self._conv52 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._upsample53 = nn.Upsample(scale_factor=2, mode='nearest')
        self._conv54 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )

        self._conv56 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv57 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv58 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv59 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv60 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv61 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )

        self._conv63 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv64 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._upsample65 = nn.Upsample(scale_factor=2, mode='nearest')
        self._conv66 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )

        self._conv68 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv69 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv70 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.SiLU(),
        )
        self._conv71 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.SiLU(),
        )
        self._conv72 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.SiLU(),
        )
        self._conv73 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.SiLU(),
        )

        self._conv75 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._max_pool76 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._conv77 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv78 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv79 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )

        self._conv81 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv82 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv83 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv84 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv85 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )
        self._conv86 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.SiLU(),
        )

        self._conv88 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._max_pool89 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._conv90 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv91 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv92 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )

        self._conv94 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(512, eps=0.001),
            nn.SiLU(),
        )
        self._conv95 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(512, eps=0.001),
            nn.SiLU(),
        )
        self._conv96 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv97 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv98 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )
        self._conv99 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(),
        )

        self._conv101 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(512, eps=0.001),
            nn.SiLU(),
        )
        self._rep_conv102 = RepConv(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, activation=nn.SiLU())
        self._rep_conv103 = RepConv(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, groups=1, activation=nn.SiLU())
        self._rep_conv104 = RepConv(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, groups=1, activation=nn.SiLU())

        self._yolo0 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=self._anchors[0].shape[0] * (embedding_size + 5), kernel_size=1),
            DescriptorYoloV7Layer(IMAGE_SIZE, 8, self._anchors[0], embedding_size)
        )
        self._yolo1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=self._anchors[1].shape[0] * (embedding_size + 5), kernel_size=1),
            DescriptorYoloV7Layer(IMAGE_SIZE, 16, self._anchors[1], embedding_size)
        )
        self._yolo2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=self._anchors[2].shape[0] * (embedding_size + 5), kernel_size=1),
            DescriptorYoloV7Layer(IMAGE_SIZE, 32, self._anchors[2], embedding_size)
        )

        self._classifier = NormalizedLinear(embedding_size, class_count)
        self._class_probs = class_probs

    def get_image_size(self):
        return IMAGE_SIZE

    def get_anchors(self):
        return self._anchors

    def get_output_strides(self):
        return self._output_strides

    def forward(self, x):
        y0 = self._conv0(x)
        y1 = self._conv1(y0)
        y2 = self._conv2(y1)
        y3 = self._conv3(y2)
        y4 = self._conv4(y3)
        y5 = self._conv5(y3)
        y6 = self._conv6(y5)
        y7 = self._conv7(y6)
        y8 = self._conv8(y7)
        y9 = self._conv9(y8)
        y10 = torch.cat([y9, y7, y5, y4], dim=1)

        y11 = self._conv11(y10)
        y12 = self._max_pool12(y11)
        y13 = self._conv13(y12)
        y14 = self._conv14(y11)
        y15 = self._conv15(y14)
        y16 = torch.cat([y15, y13], dim=1)

        y17 = self._conv17(y16)
        y18 = self._conv18(y16)
        y19 = self._conv19(y18)
        y20 = self._conv20(y19)
        y21 = self._conv21(y20)
        y22 = self._conv22(y21)
        y23 = torch.cat([y22, y20, y18, y17], dim=1)

        y24 = self._conv24(y23)
        y25 = self._max_pool25(y24)
        y26 = self._conv26(y25)
        y27 = self._conv27(y24)
        y28 = self._conv28(y27)
        y29 = torch.cat([y28, y26], dim=1)

        y30 = self._conv30(y29)
        y31 = self._conv31(y29)
        y32 = self._conv32(y31)
        y33 = self._conv33(y32)
        y34 = self._conv34(y33)
        y35 = self._conv35(y34)
        y36 = torch.cat([y35, y33, y31, y30], dim=1)

        y37 = self._conv37(y36)
        y38 = self._max_pool38(y37)
        y39 = self._conv39(y38)
        y40 = self._conv40(y37)
        y41 = self._conv41(y40)
        y42 = torch.cat([y41, y39], dim=1)

        y43 = self._conv43(y42)
        y44 = self._conv44(y42)
        y45 = self._conv45(y44)
        y46 = self._conv46(y45)
        y47 = self._conv47(y46)
        y48 = self._conv48(y47)
        y49 = torch.cat([y48, y46, y44, y43], dim=1)

        y50 = self._conv50(y49)
        y51 = self._sppcspc51(y50)
        y52 = self._conv52(y51)
        y53 = self._upsample53(y52)
        y54 = self._conv54(y37)
        y55 = torch.cat([y54, y53], dim=1)

        y56 = self._conv56(y55)
        y57 = self._conv57(y55)
        y58 = self._conv58(y57)
        y59 = self._conv59(y58)
        y60 = self._conv60(y59)
        y61 = self._conv61(y60)
        y62 = torch.cat([y61, y60, y59, y58, y57, y56], dim=1)

        y63 = self._conv63(y62)
        y64 = self._conv64(y63)
        y65 = self._upsample65(y64)
        y66 = self._conv66(y24)
        y67 = torch.cat([y66, y65], dim=1)

        y68 = self._conv68(y67)
        y69 = self._conv69(y67)
        y70 = self._conv70(y69)
        y71 = self._conv71(y70)
        y72 = self._conv72(y71)
        y73 = self._conv73(y72)
        y74 = torch.cat([y73, y72, y71, y70, y69, y68], dim=1)

        y75 = self._conv75(y74)
        y76 = self._max_pool76(y75)
        y77 = self._conv77(y76)
        y78 = self._conv78(y75)
        y79 = self._conv79(y78)
        y80 = torch.cat([y79, y77, y63], dim=1)

        y81 = self._conv81(y80)
        y82 = self._conv82(y80)
        y83 = self._conv83(y82)
        y84 = self._conv84(y83)
        y85 = self._conv85(y84)
        y86 = self._conv86(y85)
        y87 = torch.cat([y86, y85, y84, y83, y82, y81], dim=1)

        y88 = self._conv88(y87)
        y89 = self._max_pool89(y88)
        y90 = self._conv90(y89)
        y91 = self._conv91(y88)
        y92 = self._conv92(y91)
        y93 = torch.cat([y92, y90, y51], dim=1)

        y94 = self._conv94(y93)
        y95 = self._conv95(y93)
        y96 = self._conv96(y95)
        y97 = self._conv97(y96)
        y98 = self._conv98(y97)
        y99 = self._conv99(y98)
        y100 = torch.cat([y99, y98, y97, y96, y95, y94], dim=1)

        y101 = self._conv101(y100)
        y102 = self._rep_conv102(y75)
        y103 = self._rep_conv103(y88)
        y104 = self._rep_conv104(y101)

        box0, embedding0 = self._yolo0(y102)
        box1, embedding1 = self._yolo1(y103)
        box2, embedding2 = self._yolo2(y104)

        d0 = self._classify_embeddings(box0, embedding0)
        d1 = self._classify_embeddings(box1, embedding1)
        d2 = self._classify_embeddings(box2, embedding2)

        return [d0, d1, d2]

    def _classify_embeddings(self, box, embedding):
        classes = self._classifier(embedding)
        if self._class_probs:
            classes = torch.softmax(classes, dim=4)

        return torch.cat([box, classes, embedding], dim=4)

    def load_weights(self, weights_path):
        loaded_state_dict = self._filter_static_dict(torch.load(weights_path), 'anchor')
        current_state_dict = self._filter_static_dict(self.state_dict(), 'offset')

        for i, (kl, kc) in enumerate(zip(loaded_state_dict.keys(), current_state_dict.keys())):
            if current_state_dict[kc].size() != loaded_state_dict[kl].size():
                raise ValueError('Mismatching size.')
            current_state_dict[kc] = loaded_state_dict[kl]

        self.load_state_dict(current_state_dict, strict=False)

    def _filter_static_dict(self, state_dict, x):
        return OrderedDict([(k, v) for k, v in state_dict.items() if x not in k])
