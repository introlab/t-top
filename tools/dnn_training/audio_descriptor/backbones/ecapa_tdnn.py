import torch
import torch.nn as nn

from common.modules import GlobalAvgPool1d


# Inspired by https://github.com/TaoRuijie/ECAPA-TDNN and https://github.com/lawlict/ECAPA-TDNN
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, bottleneck_channels):
        super(SqueezeExcitation, self).__init__()

        self._layers = nn.Sequential(
            GlobalAvgPool1d(),
            nn.Conv1d(in_channels=in_channels, out_channels=bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_features=bottleneck_channels),
            nn.ReLU(),

            nn.Conv1d(in_channels=bottleneck_channels, out_channels=in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self._layers(x)


class SqueezeExcitationRes2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, scale):
        super(SqueezeExcitationRes2Block, self).__init__()
        self._width = out_channels // scale
        self._input_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=self._width * scale, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_features=self._width * scale),
            nn.ReLU()
        )

        self._count = scale - 1

        padding = dilation * (kernel_size - 1) // 2
        self._scale_layers = nn.ModuleList()
        for _ in range(self._count):
            self._scale_layers.append(nn.Sequential(
                nn.Conv1d(in_channels=self._width, out_channels=self._width, kernel_size=kernel_size, dilation=dilation,
                          padding=padding, bias=False),
                nn.BatchNorm1d(num_features=self._width),
                nn.ReLU()
            ))

        self._output_layers = nn.Sequential(
            nn.Conv1d(in_channels=self._width * scale, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_features=self._width * scale),
            nn.ReLU(),

            SqueezeExcitation(in_channels=out_channels, bottleneck_channels=128)
        )

    def forward(self, x):
        y = self._input_layers(x)

        outputs = []
        spx = torch.split(y, self._width, dim=1)
        for i in range(self._count):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self._scale_layers[i](sp)
            outputs.append(sp)

        outputs.append(spx[self._count])

        y = self._output_layers(torch.cat(outputs, dim=1)) + x
        return y + x


class EcapaTdnnAttentionPooling(nn.Module):
    def __init__(self, in_channels, bottleneck_channels):
        super(EcapaTdnnAttentionPooling, self).__init__()

        self._layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels * 3, out_channels=bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_features=bottleneck_channels),
            nn.Tanh(),
            nn.Conv1d(in_channels=bottleneck_channels, out_channels=in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_features=in_channels),
            nn.Softmax(dim=2),
        )

    def forward(self, x, eps=1e-4):
        global_mean = torch.mean(x, dim=2, keepdim=True)
        global_std = torch.sqrt((torch.mean((x - global_mean) ** 2, dim=2, keepdim=True)).clamp(min=eps))
        global_x = torch.cat([x,
                              global_mean.expand(x.size(0), x.size(1), x.size(2)),
                              global_std.expand(x.size(0), x.size(1), x.size(2))],
                             dim=1)
        w = self._layers(global_x)

        mean = torch.sum(w * x, dim=2)
        std = torch.sqrt((torch.sum((x**2) * w, dim=2) - mean**2).clamp(min=eps))

        return torch.cat([mean, std], dim=1)


class SmallEcapaTdnnAttentionPooling(nn.Module):
    def __init__(self, in_channels, bottleneck_channels):
        super(SmallEcapaTdnnAttentionPooling, self).__init__()

        self._layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_features=bottleneck_channels),
            nn.Tanh(),
            nn.Conv1d(in_channels=bottleneck_channels, out_channels=in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_features=in_channels),
            nn.Softmax(dim=2),
        )

    def forward(self, x, eps=1e-4):
        w = self._layers(x)
        mean = torch.sum(w * x, dim=2)
        std = torch.sqrt((torch.sum((x**2) * w, dim=2) - mean**2).clamp(min=eps))

        return torch.cat([mean, std], dim=1)


class EcapaTdnn(nn.Module):
    def __init__(self, n_features, channels=512):
        super(EcapaTdnn, self).__init__()

        self._layer1 = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(num_features=channels),
            nn.ReLU(),
        )

        self._layer2 = SqueezeExcitationRes2Block(channels, channels, kernel_size=2, dilation=2, scale=8)
        self._layer3 = SqueezeExcitationRes2Block(channels, channels, kernel_size=3, dilation=3, scale=8)
        self._layer4 = SqueezeExcitationRes2Block(channels, channels, kernel_size=3, dilation=4, scale=8)

        self._layer5 = nn.Sequential(
            nn.Conv1d(in_channels=3 * channels, out_channels=1536, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_features=1536),
            nn.ReLU()
        )

        self._pooling = EcapaTdnnAttentionPooling(in_channels=1536, bottleneck_channels=256)

    def forward(self, x):
        x = x.squeeze(1)

        y1 = self._layer1(x)

        y2 = self._layer2(y1)
        y3 = self._layer3(y2)
        y4 = self._layer4(y3)

        y5 = self._layer5(torch.cat([y2, y3, y4], dim=1))

        y6 = self._pooling(y5)

        return y6.unsqueeze(2).unsqueeze(3)

    def last_channel_count(self):
        return 3072


class SmallEcapaTdnn(nn.Module):
    def __init__(self, n_features, channels=256):
        super(SmallEcapaTdnn, self).__init__()

        self._layer1 = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(num_features=channels),
            nn.ReLU(),
        )

        self._layer2 = SqueezeExcitationRes2Block(channels, channels, kernel_size=2, dilation=2, scale=4)
        self._layer3 = SqueezeExcitationRes2Block(channels, channels, kernel_size=3, dilation=3, scale=4)
        self._layer4 = SqueezeExcitationRes2Block(channels, channels, kernel_size=3, dilation=4, scale=4)

        self._layer5 = nn.Sequential(
            nn.Conv1d(in_channels=3 * channels, out_channels=1536, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_features=1536),
            nn.ReLU()
        )

        self._pooling = SmallEcapaTdnnAttentionPooling(in_channels=1536, bottleneck_channels=256)

    def forward(self, x):
        x = x.squeeze(1)

        y1 = self._layer1(x)

        y2 = self._layer2(y1)
        y3 = self._layer3(y2)
        y4 = self._layer4(y3)

        y5 = self._layer5(torch.cat([y2, y3, y4], dim=1))

        y6 = self._pooling(y5)

        return y6.unsqueeze(2).unsqueeze(3)

    def last_channel_count(self):
        return 3072
