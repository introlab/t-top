import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    def __init__(self, in_channels, kernel_size, kernel_stride, out_channels, reduce_size, pool, conv_bias=False):
        super(InceptionModule, self).__init__()

        self._branches = self._create_branches(in_channels, kernel_size, kernel_stride, out_channels, reduce_size, pool,
                                               conv_bias)

    def _create_branches(self, in_channels, kernel_size, kernel_stride, out_channels, reduce_size, pool, conv_bias):
        branches = nn.ModuleList()
        for i in range(len(kernel_size)):
            branches.append(self._create_branch(in_channels, kernel_size[i], kernel_stride[i], out_channels[i],
                                                reduce_size[i], conv_bias))

        branches.append(self._create_pool_conv_1x1_branch(in_channels, reduce_size[-2], pool, conv_bias))
        if reduce_size[-1] is not None:
            branches.append(self._create_conv_1x1_branch(in_channels, reduce_size[-1], conv_bias))

        return branches

    def _create_branch(self, in_channels, kernel_size, kernel_stride, out_channels, reduce_size, conv_bias):
        return nn.Sequential(
            nn.Conv2d(in_channels, reduce_size, kernel_size=1, bias=conv_bias),
            nn.BatchNorm2d(reduce_size),
            nn.ReLU(inplace=True),

            nn.Conv2d(reduce_size, out_channels, kernel_size,
                      stride=kernel_stride, padding=kernel_size // 2, bias=conv_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def _create_pool_conv_1x1_branch(self, in_channels, reduce_size, pool, conv_bias):
        if reduce_size is None:
            return nn.Sequential(pool)
        else:
            return nn.Sequential(
                pool,
                nn.Conv2d(in_channels, reduce_size, kernel_size=1, bias=conv_bias),
                nn.BatchNorm2d(reduce_size),
                nn.ReLU(inplace=True))

    def _create_conv_1x1_branch(self, in_channels, reduce_size, conv_bias):
        return nn.Sequential(
            nn.Conv2d(in_channels, reduce_size, kernel_size=1, bias=conv_bias),
            nn.BatchNorm2d(reduce_size),
            nn.ReLU(inplace=True))

    def forward(self, x):
        branch_results = [branch(x) for branch in self._branches]
        return torch.cat(branch_results, dim=1)
