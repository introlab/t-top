import torch
import torch.nn as nn
import torch.nn.functional as F


# Inspired by https://github.com/Nanne/pytorch-NetVlad/blob/master/netvlad.py,
# https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py and
# https://github.com/WeidiXie/VGG-Speaker-Recognition/blob/master/src/model.py
class NetVLAD(nn.Module):
    def __init__(self, embedding_size, cluster_count, ghost_cluster_count=0, normalize_input=True):
        super(NetVLAD, self).__init__()
        self._embedding_size = embedding_size
        self._cluster_count = cluster_count
        self._ghost_cluster_count = ghost_cluster_count
        self._normalize_input = normalize_input
        self._all_cluster_count = cluster_count + ghost_cluster_count

        self._conv = nn.Conv2d(embedding_size, self._all_cluster_count, kernel_size=1, bias=True)
        nn.init.orthogonal_(self._conv.weight)
        nn.init.constant_(self._conv.bias, 0.0)

        self._centroids = nn.Parameter(torch.empty(self._all_cluster_count, embedding_size))
        nn.init.orthogonal_(self._centroids)

    def forward(self, x):
        N = x.size()[0]
        C = x.size()[1]
        assert (C == self._embedding_size), 'C must equal to embedding_size ({} != {})'.format(C, self._embedding_size)

        if self._normalize_input:
            x = F.normalize(x, p=2, dim=1)

        soft_assignment = F.softmax(self._conv(x).view(N, self._all_cluster_count, -1), dim=1)

        # Compute the VLAD
        x = x.view(N, C, -1)
        a = x.unsqueeze(0).expand(self._all_cluster_count, x.size(0), x.size(1), x.size(2))
        a = a.unsqueeze(0).permute(0, 2, 1, 3, 4)
        b = self._centroids.contiguous().unsqueeze(0)
        b = b.expand(x.size(-1), self._centroids.size(0), self._centroids.size(1))
        b = b.unsqueeze(0).permute(0, 2, 3, 1).unsqueeze(0)

        residual = (a - b)[(0,)]
        vlad = (residual * soft_assignment.unsqueeze(2)).sum(dim=3)

        vlad = vlad[:, :self._cluster_count, :]  # Remove the ghost clusters

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        return F.normalize(vlad.view(N, -1), p=2, dim=1)  # L2 normalize
