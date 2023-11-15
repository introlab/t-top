import numpy as np

import torch
import torch.nn as nn

from object_detection.modules.descriptor_yolo_layer import DescriptorYoloV4Layer

IMAGE_SIZE = (416, 416)
IN_CHANNELS = 3


# Genereated from: https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny.cfg
class DescriptorYoloV4Tiny(nn.Module):
    def __init__(self, class_count, descriptor_size):
        super(DescriptorYoloV4Tiny, self).__init__()
        self._class_count = class_count
        self._descriptor_size = descriptor_size

        self._anchors = []
        self._output_strides = []
        self._conv0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self._conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self._conv7 = nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self._max_pool9 = nn.MaxPool2d(2, stride=2, padding=0)
        self._conv10 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self._conv12 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv13 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self._conv15 = nn.Sequential(
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self._max_pool17 = nn.MaxPool2d(2, stride=2, padding=0)
        self._conv18 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self._conv20 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv21 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self._conv23 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self._max_pool25 = nn.MaxPool2d(2, stride=2, padding=0)
        self._conv26 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv27 = nn.Sequential(
            nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv28 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv29 = nn.Sequential(
            nn.Conv2d(512, 3 * (5 + self._descriptor_size), 1, stride=1, padding=0, bias=True),
        )
        self._anchors.append(np.array([(81, 82), (135, 169), (344, 319)]))
        self._output_strides.append(32)
        self._yolo30 = DescriptorYoloV4Layer(IMAGE_SIZE, 32, self._anchors[-1].tolist(), class_count, descriptor_size,
                                             1.05)

        self._conv32 = nn.Sequential(
            nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._upsample33 = nn.Upsample(scale_factor=2, mode='nearest')

        self._conv35 = nn.Sequential(
            nn.Conv2d(384, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self._conv36 = nn.Sequential(
            nn.Conv2d(256, 3 * (5 + self._descriptor_size), 1, stride=1, padding=0, bias=True),
        )
        self._anchors.append(np.array([(23, 27), (37, 58), (81, 82)]))
        self._output_strides.append(16)
        self._yolo37 = DescriptorYoloV4Layer(IMAGE_SIZE, 16, self._anchors[-1].tolist(), class_count, descriptor_size,
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
        C = y2.size()[1]
        y3 = y2[:, C // 2 * 1:C // 2 * (1 + 1), :, :]

        y4 = self._conv4(y3)
        y5 = self._conv5(y4)
        y6 = torch.cat([y5, y4], dim=1)

        y7 = self._conv7(y6)
        y8 = torch.cat([y2, y7], dim=1)

        y9 = self._max_pool9(y8)
        y10 = self._conv10(y9)
        C = y10.size()[1]
        y11 = y10[:, C // 2 * 1:C // 2 * (1 + 1), :, :]

        y12 = self._conv12(y11)
        y13 = self._conv13(y12)
        y14 = torch.cat([y13, y12], dim=1)

        y15 = self._conv15(y14)
        y16 = torch.cat([y10, y15], dim=1)

        y17 = self._max_pool17(y16)
        y18 = self._conv18(y17)
        C = y18.size()[1]
        y19 = y18[:, C // 2 * 1:C // 2 * (1 + 1), :, :]

        y20 = self._conv20(y19)
        y21 = self._conv21(y20)
        y22 = torch.cat([y21, y20], dim=1)

        y23 = self._conv23(y22)
        y24 = torch.cat([y18, y23], dim=1)

        y25 = self._max_pool25(y24)
        y26 = self._conv26(y25)
        y27 = self._conv27(y26)
        y28 = self._conv28(y27)
        y29 = self._conv29(y28)
        y30 = self._yolo30(y29)

        y32 = self._conv32(y27)
        y33 = self._upsample33(y32)
        y34 = torch.cat([y33, y23], dim=1)

        y35 = self._conv35(y34)
        y36 = self._conv36(y35)
        y37 = self._yolo37(y36)

        return [y30, y37]
