from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn

from object_detection.modules.yolo_layer import YoloV7Layer

from object_detection.datasets.object_detection_coco import CLASS_COUNT as COCO_CLASS_COUNT

IMAGE_SIZE = (640, 640)
IN_CHANNELS = 3


# Generated from: yolov7-tiny.yaml:
class YoloV7Tiny(nn.Module):
    def __init__(self, dataset_type='coco', class_probs=False):
        super(YoloV7Tiny, self).__init__()

        self._anchors = []
        self._output_strides = [8, 16, 32]

        if dataset_type == 'coco':
            class_count = COCO_CLASS_COUNT
            self._anchors.append(np.array([(12, 16), (19, 36), (40, 28)]))
            self._anchors.append(np.array([(36, 75), (76, 55), (72, 146)]))
            self._anchors.append(np.array([(142, 110), (192, 243), (459, 401)]))
        elif dataset_type == 'objects365':
            class_count = 365
            self._anchors.append(np.array([(8, 7), (15, 14), (17, 36)]))
            self._anchors.append(np.array([(38, 22), (39, 53), (93, 59)]))
            self._anchors.append(np.array([(55, 122), (126, 179), (257, 324)]))

        self._conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.LeakyReLU(0.1),
        )

        self._conv7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._max_pool8 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._conv9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv12 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.LeakyReLU(0.1),
        )

        self._conv14 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._max_pool15 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._conv16 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv17 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv18 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv19 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.LeakyReLU(0.1),
        )

        self._conv21 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._max_pool22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._conv23 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv24 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv25 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv26 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.LeakyReLU(0.1),
        )

        self._conv28 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(512, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv29 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv30 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._max_pool31 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self._max_pool32 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self._max_pool33 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

        self._conv35 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.LeakyReLU(0.1),
        )

        self._conv37 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv38 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._upsample39 = nn.Upsample(scale_factor=2, mode='nearest')
        self._conv40 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.LeakyReLU(0.1),
        )

        self._conv42 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv43 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv44 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv45 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.LeakyReLU(0.1),
        )

        self._conv47 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv48 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._upsample49 = nn.Upsample(scale_factor=2, mode='nearest')
        self._conv50 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.LeakyReLU(0.1),
        )

        self._conv52 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv53 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv54 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv55 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001),
            nn.LeakyReLU(0.1),
        )

        self._conv57 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv58 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.LeakyReLU(0.1),
        )

        self._conv60 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv61 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )
        self._conv62 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv63 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.LeakyReLU(0.1),
        )

        self._conv65 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv66 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.LeakyReLU(0.1),
        )

        self._conv68 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv69 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv70 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv71 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.LeakyReLU(0.1),
        )

        self._conv73 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv74 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv75 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.LeakyReLU(0.1),
        )
        self._conv76 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(512, eps=0.001),
            nn.LeakyReLU(0.1),
        )

        self._yolo0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=self._anchors[0].shape[0] * (class_count + 5), kernel_size=1),
            YoloV7Layer(IMAGE_SIZE, 8, self._anchors[0], class_count, class_probs=class_probs)
        )
        self._yolo1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=self._anchors[1].shape[0] * (class_count + 5), kernel_size=1),
            YoloV7Layer(IMAGE_SIZE, 16, self._anchors[1], class_count, class_probs=class_probs)
        )
        self._yolo2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=self._anchors[2].shape[0] * (class_count + 5), kernel_size=1),
            YoloV7Layer(IMAGE_SIZE, 32, self._anchors[2], class_count, class_probs=class_probs)
        )

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
        y3 = self._conv3(y1)
        y4 = self._conv4(y3)
        y5 = self._conv5(y4)
        y6 = torch.cat([y5, y4, y3, y2], dim=1)

        y7 = self._conv7(y6)
        y8 = self._max_pool8(y7)
        y9 = self._conv9(y8)
        y10 = self._conv10(y8)
        y11 = self._conv11(y10)
        y12 = self._conv12(y11)
        y13 = torch.cat([y12, y11, y10, y9], dim=1)

        y14 = self._conv14(y13)
        y15 = self._max_pool15(y14)
        y16 = self._conv16(y15)
        y17 = self._conv17(y15)
        y18 = self._conv18(y17)
        y19 = self._conv19(y18)
        y20 = torch.cat([y19, y18, y17, y16], dim=1)

        y21 = self._conv21(y20)
        y22 = self._max_pool22(y21)
        y23 = self._conv23(y22)
        y24 = self._conv24(y22)
        y25 = self._conv25(y24)
        y26 = self._conv26(y25)
        y27 = torch.cat([y26, y25, y24, y23], dim=1)

        y28 = self._conv28(y27)
        y29 = self._conv29(y28)
        y30 = self._conv30(y28)
        y31 = self._max_pool31(y30)
        y32 = self._max_pool32(y30)
        y33 = self._max_pool33(y30)
        y34 = torch.cat([y33, y32, y31, y30], dim=1)

        y35 = self._conv35(y34)
        y36 = torch.cat([y35, y29], dim=1)

        y37 = self._conv37(y36)
        y38 = self._conv38(y37)
        y39 = self._upsample39(y38)
        y40 = self._conv40(y21)
        y41 = torch.cat([y40, y39], dim=1)

        y42 = self._conv42(y41)
        y43 = self._conv43(y41)
        y44 = self._conv44(y43)
        y45 = self._conv45(y44)
        y46 = torch.cat([y45, y44, y43, y42], dim=1)

        y47 = self._conv47(y46)
        y48 = self._conv48(y47)
        y49 = self._upsample49(y48)
        y50 = self._conv50(y14)
        y51 = torch.cat([y50, y49], dim=1)

        y52 = self._conv52(y51)
        y53 = self._conv53(y51)
        y54 = self._conv54(y53)
        y55 = self._conv55(y54)
        y56 = torch.cat([y55, y54, y53, y52], dim=1)

        y57 = self._conv57(y56)
        y58 = self._conv58(y57)
        y59 = torch.cat([y58, y47], dim=1)

        y60 = self._conv60(y59)
        y61 = self._conv61(y59)
        y62 = self._conv62(y61)
        y63 = self._conv63(y62)
        y64 = torch.cat([y63, y62, y61, y60], dim=1)

        y65 = self._conv65(y64)
        y66 = self._conv66(y65)
        y67 = torch.cat([y66, y37], dim=1)

        y68 = self._conv68(y67)
        y69 = self._conv69(y67)
        y70 = self._conv70(y69)
        y71 = self._conv71(y70)
        y72 = torch.cat([y71, y70, y69, y68], dim=1)

        y73 = self._conv73(y72)
        y74 = self._conv74(y57)
        y75 = self._conv75(y65)
        y76 = self._conv76(y73)

        d0 = self._yolo0(y74)
        d1 = self._yolo1(y75)
        d2 = self._yolo2(y76)
        return [d0, d1, d2]

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
