import torch
import torch.nn as nn


class PSLAAttention(nn.Module):
    def __init__(self, in_channels, out_channels, head_count=4):
        super(PSLAAttention, self).__init__()

        self._attention_convs = nn.ModuleList()
        self._feature_convs = nn.ModuleList()
        for i in range(head_count):
            self._attention_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid()
            ))
            self._feature_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid()
            ))

        self._head_weights = nn.Parameter(torch.tensor([1.0 / head_count] * head_count))

    def forward(self, x, attention_eps=1e-7):
        y = []
        for attention_conv, feature_conv, head_weight in zip(self._attention_convs, self._feature_convs, self._head_weights):
            attention = torch.clamp(attention_conv(x), attention_eps, 1.0 - attention_eps)
            normalized_attention = attention / torch.sum(attention, dim=3, keepdim=True)

            feature = feature_conv(x)
            y.append(torch.sum(normalized_attention * feature, dim=3) * head_weight)

        return torch.stack(y, dim=0).sum(dim=0).squeeze(2)
