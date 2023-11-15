import numpy as np

import torch
import torch.nn as nn

from common.modules import Mish

from object_detection.modules.descriptor_yolo_layer import DescriptorYoloV4Layer

IMAGE_SIZE = (608, 608)
IN_CHANNELS = 3


# Genereated from: https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg
class DescriptorYoloV4(nn.Module):
    def __init__(self, class_count, descriptor_size):
        super(DescriptorYoloV4, self).__init__()
        self._class_count = class_count
        self._descriptor_size = descriptor_size

        self._anchors = []
        self._output_strides = []
        self._conv0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Mish()
        )
        self._conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            Mish()
        )
        self._conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            Mish()
        )

        self._conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            Mish()
        )
        self._conv5 = nn.Sequential(
            nn.Conv2d(64, 32, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            Mish()
        )
        self._conv6 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            Mish()
        )

        self._conv8 = nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            Mish()
        )

        self._conv10 = nn.Sequential(
            nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            Mish()
        )
        self._conv11 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )
        self._conv12 = nn.Sequential(
            nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            Mish()
        )

        self._conv14 = nn.Sequential(
            nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            Mish()
        )
        self._conv15 = nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            Mish()
        )
        self._conv16 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            Mish()
        )

        self._conv18 = nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            Mish()
        )
        self._conv19 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            Mish()
        )

        self._conv21 = nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            Mish()
        )

        self._conv23 = nn.Sequential(
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )
        self._conv24 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )
        self._conv25 = nn.Sequential(
            nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )

        self._conv27 = nn.Sequential(
            nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )
        self._conv28 = nn.Sequential(
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )
        self._conv29 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )

        self._conv31 = nn.Sequential(
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )
        self._conv32 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )

        self._conv34 = nn.Sequential(
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )
        self._conv35 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )

        self._conv37 = nn.Sequential(
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )
        self._conv38 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )

        self._conv40 = nn.Sequential(
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )
        self._conv41 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )

        self._conv43 = nn.Sequential(
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )
        self._conv44 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )

        self._conv46 = nn.Sequential(
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )
        self._conv47 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )

        self._conv49 = nn.Sequential(
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )
        self._conv50 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )

        self._conv52 = nn.Sequential(
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            Mish()
        )

        self._conv54 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )
        self._conv55 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            Mish()
        )
        self._conv56 = nn.Sequential(
            nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )

        self._conv58 = nn.Sequential(
            nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )
        self._conv59 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )
        self._conv60 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )

        self._conv62 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )
        self._conv63 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )

        self._conv65 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )
        self._conv66 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )

        self._conv68 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )
        self._conv69 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )

        self._conv71 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )
        self._conv72 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )

        self._conv74 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )
        self._conv75 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )

        self._conv77 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )
        self._conv78 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )

        self._conv80 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )
        self._conv81 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )

        self._conv83 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            Mish()
        )

        self._conv85 = nn.Sequential(
            nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            Mish()
        )
        self._conv86 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            Mish()
        )
        self._conv87 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            Mish()
        )

        self._conv89 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            Mish()
        )
        self._conv90 = nn.Sequential(
            nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            Mish()
        )
        self._conv91 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            Mish()
        )

        self._conv93 = nn.Sequential(
            nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            Mish()
        )
        self._conv94 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            Mish()
        )

        self._conv96 = nn.Sequential(
            nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            Mish()
        )
        self._conv97 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            Mish()
        )

        self._conv99 = nn.Sequential(
            nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            Mish()
        )
        self._conv100 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            Mish()
        )

        self._conv102 = nn.Sequential(
            nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            Mish()
        )

        self._conv104 = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            Mish()
        )
        self._conv105 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv106 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv107 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._max_pool108 = nn.MaxPool2d(5, stride=1, padding=2)

        self._max_pool110 = nn.MaxPool2d(9, stride=1, padding=4)

        self._max_pool112 = nn.MaxPool2d(13, stride=1, padding=6)

        self._conv114 = nn.Sequential(
            nn.Conv2d(2048, 512, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv115 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv116 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv117 = nn.Sequential(
            nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._upsample118 = nn.Upsample(scale_factor=2, mode='nearest')

        self._conv120 = nn.Sequential(
            nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self._conv122 = nn.Sequential(
            nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv123 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv124 = nn.Sequential(
            nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv125 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv126 = nn.Sequential(
            nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv127 = nn.Sequential(
            nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._upsample128 = nn.Upsample(scale_factor=2, mode='nearest')

        self._conv130 = nn.Sequential(
            nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self._conv132 = nn.Sequential(
            nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv133 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv134 = nn.Sequential(
            nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv135 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv136 = nn.Sequential(
            nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv137 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv138 = nn.Sequential(
            nn.Conv2d(256, 3 * (5 + self._descriptor_size), 1, stride=1, padding=0, bias=True),
        )
        self._anchors.append(np.array([(12, 16), (19, 36), (40, 28)]))
        self._output_strides.append(8)
        self._yolo139 = DescriptorYoloV4Layer(IMAGE_SIZE, 8, self._anchors[-1].tolist(), class_count, descriptor_size,
                                              1.2)

        self._conv141 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self._conv143 = nn.Sequential(
            nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv144 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv145 = nn.Sequential(
            nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv146 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv147 = nn.Sequential(
            nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv148 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv149 = nn.Sequential(
            nn.Conv2d(512, 3 * (5 + self._descriptor_size), 1, stride=1, padding=0, bias=True),
        )
        self._anchors.append(np.array([(36, 75), (76, 55), (72, 146)]))
        self._output_strides.append(16)
        self._yolo150 = DescriptorYoloV4Layer(IMAGE_SIZE, 16, self._anchors[-1].tolist(), class_count, descriptor_size,
                                              1.1)

        self._conv152 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self._conv154 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv155 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv156 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv157 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv158 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv159 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv160 = nn.Sequential(
            nn.Conv2d(1024, 3 * (5 + self._descriptor_size), 1, stride=1, padding=0, bias=True),
        )
        self._anchors.append(np.array([(142, 110), (192, 243), (459, 401)]))
        self._output_strides.append(32)
        self._yolo161 = DescriptorYoloV4Layer(IMAGE_SIZE, 32, self._anchors[-1].tolist(), class_count, descriptor_size,
                                              1.05)

    def get_image_size(self):
        return IMAGE_SIZE

    def get_class_count(self):
        return self._class_count

    def get_descriptor_size(self):
        return self._descriptor_size

    def get_anchors(self):
        return self._anchors

    def get_output_strides(self):
        return self._output_strides

    def forward(self, x):
        y0 = self._conv0(x)
        y1 = self._conv1(y0)
        y2 = self._conv2(y1)

        y4 = self._conv4(y1)
        y5 = self._conv5(y4)
        y6 = self._conv6(y5)
        y7 = y6 + y4

        y8 = self._conv8(y7)
        y9 = torch.cat([y8, y2], dim=1)

        y10 = self._conv10(y9)
        y11 = self._conv11(y10)
        y12 = self._conv12(y11)

        y14 = self._conv14(y11)
        y15 = self._conv15(y14)
        y16 = self._conv16(y15)
        y17 = y16 + y14

        y18 = self._conv18(y17)
        y19 = self._conv19(y18)
        y20 = y19 + y17

        y21 = self._conv21(y20)
        y22 = torch.cat([y21, y12], dim=1)

        y23 = self._conv23(y22)
        y24 = self._conv24(y23)
        y25 = self._conv25(y24)

        y27 = self._conv27(y24)
        y28 = self._conv28(y27)
        y29 = self._conv29(y28)
        y30 = y29 + y27

        y31 = self._conv31(y30)
        y32 = self._conv32(y31)
        y33 = y32 + y30

        y34 = self._conv34(y33)
        y35 = self._conv35(y34)
        y36 = y35 + y33

        y37 = self._conv37(y36)
        y38 = self._conv38(y37)
        y39 = y38 + y36

        y40 = self._conv40(y39)
        y41 = self._conv41(y40)
        y42 = y41 + y39

        y43 = self._conv43(y42)
        y44 = self._conv44(y43)
        y45 = y44 + y42

        y46 = self._conv46(y45)
        y47 = self._conv47(y46)
        y48 = y47 + y45

        y49 = self._conv49(y48)
        y50 = self._conv50(y49)
        y51 = y50 + y48

        y52 = self._conv52(y51)
        y53 = torch.cat([y52, y25], dim=1)

        y54 = self._conv54(y53)
        y55 = self._conv55(y54)
        y56 = self._conv56(y55)

        y58 = self._conv58(y55)
        y59 = self._conv59(y58)
        y60 = self._conv60(y59)
        y61 = y60 + y58

        y62 = self._conv62(y61)
        y63 = self._conv63(y62)
        y64 = y63 + y61

        y65 = self._conv65(y64)
        y66 = self._conv66(y65)
        y67 = y66 + y64

        y68 = self._conv68(y67)
        y69 = self._conv69(y68)
        y70 = y69 + y67

        y71 = self._conv71(y70)
        y72 = self._conv72(y71)
        y73 = y72 + y70

        y74 = self._conv74(y73)
        y75 = self._conv75(y74)
        y76 = y75 + y73

        y77 = self._conv77(y76)
        y78 = self._conv78(y77)
        y79 = y78 + y76

        y80 = self._conv80(y79)
        y81 = self._conv81(y80)
        y82 = y81 + y79

        y83 = self._conv83(y82)
        y84 = torch.cat([y83, y56], dim=1)

        y85 = self._conv85(y84)
        y86 = self._conv86(y85)
        y87 = self._conv87(y86)

        y89 = self._conv89(y86)
        y90 = self._conv90(y89)
        y91 = self._conv91(y90)
        y92 = y91 + y89

        y93 = self._conv93(y92)
        y94 = self._conv94(y93)
        y95 = y94 + y92

        y96 = self._conv96(y95)
        y97 = self._conv97(y96)
        y98 = y97 + y95

        y99 = self._conv99(y98)
        y100 = self._conv100(y99)
        y101 = y100 + y98

        y102 = self._conv102(y101)
        y103 = torch.cat([y102, y87], dim=1)

        y104 = self._conv104(y103)
        y105 = self._conv105(y104)
        y106 = self._conv106(y105)
        y107 = self._conv107(y106)
        y108 = self._max_pool108(y107)

        y110 = self._max_pool110(y107)

        y112 = self._max_pool112(y107)
        y113 = torch.cat([y112, y110, y108, y107], dim=1)

        y114 = self._conv114(y113)
        y115 = self._conv115(y114)
        y116 = self._conv116(y115)
        y117 = self._conv117(y116)
        y118 = self._upsample118(y117)

        y120 = self._conv120(y85)
        y121 = torch.cat([y120, y118], dim=1)

        y122 = self._conv122(y121)
        y123 = self._conv123(y122)
        y124 = self._conv124(y123)
        y125 = self._conv125(y124)
        y126 = self._conv126(y125)
        y127 = self._conv127(y126)
        y128 = self._upsample128(y127)

        y130 = self._conv130(y54)
        y131 = torch.cat([y130, y128], dim=1)

        y132 = self._conv132(y131)
        y133 = self._conv133(y132)
        y134 = self._conv134(y133)
        y135 = self._conv135(y134)
        y136 = self._conv136(y135)
        y137 = self._conv137(y136)
        y138 = self._conv138(y137)
        y139 = self._yolo139(y138)

        y141 = self._conv141(y136)
        y142 = torch.cat([y141, y126], dim=1)

        y143 = self._conv143(y142)
        y144 = self._conv144(y143)
        y145 = self._conv145(y144)
        y146 = self._conv146(y145)
        y147 = self._conv147(y146)
        y148 = self._conv148(y147)
        y149 = self._conv149(y148)
        y150 = self._yolo150(y149)

        y152 = self._conv152(y147)
        y153 = torch.cat([y152, y116], dim=1)

        y154 = self._conv154(y153)
        y155 = self._conv155(y154)
        y156 = self._conv156(y155)
        y157 = self._conv157(y156)
        y158 = self._conv158(y157)
        y159 = self._conv159(y158)
        y160 = self._conv160(y159)
        y161 = self._yolo161(y160)

        return [y139, y150, y161]
