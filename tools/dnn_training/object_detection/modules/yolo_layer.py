import torch
import torch.nn as nn

X_INDEX = 0
Y_INDEX = 1
W_INDEX = 2
H_INDEX = 3
CONFIDENCE_INDEX = 4
CLASSES_INDEX = 5


class YoloLayer(nn.Module):
    def __init__(self, image_size, stride, anchors, class_count, scale_x_y):
        super(YoloLayer, self).__init__()
        self._grid_size = (image_size[1] // stride, image_size[0] // stride)
        self._stride = stride

        self._anchors = [(a[0] // stride, a[1] // stride) for a in anchors]
        self._class_count = class_count
        self._scale_x_y = scale_x_y

        x = torch.arange(self._grid_size[1])
        y = torch.arange(self._grid_size[0])
        y_offset, x_offset = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('_x_offset', x_offset.float().clone())
        self.register_buffer('_y_offset', y_offset.float().clone())

        # Fix scripting errors
        self._x_index = X_INDEX
        self._y_index = Y_INDEX
        self._w_index = W_INDEX
        self._h_index = H_INDEX
        self._confidence_index = CONFIDENCE_INDEX
        self._classes_index = CLASSES_INDEX

    def forward(self, t):
        N = t.size()[0]
        N_ANCHORS = len(self._anchors)
        H = self._grid_size[1]
        W = self._grid_size[0]
        N_PREDICTION = 5 + self._class_count

        # Transform x
        t = t.permute(0, 2, 3, 1).reshape(N, H, W, N_ANCHORS, N_PREDICTION).permute(0, 3, 1, 2, 4)
        x = torch.sigmoid(t[:, :, :, :, self._x_index]) * self._scale_x_y - 0.5 * (self._scale_x_y - 1)
        x += self._x_offset
        x *= self._stride
        x = x.unsqueeze(4).permute(0, 2, 3, 1, 4)

        # Transform y
        y = torch.sigmoid(t[:, :, :, :, self._y_index]) * self._scale_x_y - 0.5 * (self._scale_x_y - 1)
        y += self._y_offset
        y *= self._stride
        y = y.unsqueeze(4).permute(0, 2, 3, 1, 4)

        t = t.permute(0, 2, 3, 1, 4)

        # Transform w and h
        w = []
        h = []
        for i in range(N_ANCHORS):
            w.append(
                torch.exp(t[:, :, :, i, self._w_index:self._w_index + 1]).clamp(max=1000) * self._anchors[i][0])
            h.append(
                torch.exp(t[:, :, :, i, self._h_index:self._h_index + 1]).clamp(max=1000) * self._anchors[i][1])

        w = torch.cat(w, dim=3).unsqueeze(4) * self._stride
        h = torch.cat(h, dim=3).unsqueeze(4) * self._stride

        # Transform confidence
        confidence = torch.sigmoid(t[:, :, :, :, self._confidence_index:self._confidence_index + 1])
        descriptors = t[:, :, :, :, self._classes_index:]

        return torch.cat([x, y, w, h, confidence, descriptors], dim=4)
