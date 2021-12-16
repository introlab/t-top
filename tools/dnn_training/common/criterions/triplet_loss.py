import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2, noise_variance=0.001):
        super(TripletLoss, self).__init__()
        self._margin = margin
        self._noise_variance = noise_variance

    def forward(self, x, target):
        pairwise_distances = self._get_pairwise_distances(x)
        positive_mask, negative_mask = self._get_masks(target)

        positive_distances = pairwise_distances[positive_mask]
        negative_distances = pairwise_distances[negative_mask]
        if negative_distances.size()[0] == 0:
            return torch.tensor([0.0], dtype=x.dtype, device=x.device, requires_grad=True)

        negative_distances_indexes = torch.multinomial(1 / (negative_distances + 1e-6), positive_distances.size()[0],
                                                       replacement=True)
        negative_distances = negative_distances[negative_distances_indexes]

        semi_hard_mask = positive_distances < negative_distances
        positive_distances = positive_distances[semi_hard_mask]
        negative_distances = negative_distances[semi_hard_mask]

        return torch.clamp(positive_distances - negative_distances + self._margin, min=0.0).mean()

    def _get_pairwise_distances(self, x):
        x = F.normalize(x + self._noise_variance * torch.randn_like(x), dim=1, p=2)
        return torch.norm(x.unsqueeze(1) - x, dim=2, p=2)

    def _get_masks(self, target):
        N = target.size()[0]
        expanded_target = target.expand(N, N)
        positive_mask = expanded_target == expanded_target.t()
        positive_mask.fill_diagonal_(False)
        negative_mask = expanded_target != expanded_target.t()

        return positive_mask, negative_mask
