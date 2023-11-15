import torch
import torch.nn as nn


class AmSoftmaxLoss(nn.Module):
    def __init__(self, s=10.0, m=0.2, start_annealing_epoch=0, end_annealing_epoch=0):
        super(AmSoftmaxLoss, self).__init__()
        self._s = s
        self._m = 0.0
        self._target_m = m

        self._epoch = -1
        self._start_annealing_epoch = start_annealing_epoch
        self._end_annealing_epoch = end_annealing_epoch
        self.next_epoch()

    def forward(self, scores, target):
        scores = scores.clone()

        numerator = self._s * (scores[range(scores.size(0)), target] - self._m)
        scores[range(scores.size(0)), target] = -float('inf')
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self._s * scores), dim=1)
        loss = numerator - torch.log(denominator)
        return -loss.mean()

    def next_epoch(self):
        self._epoch += 1
        if self._epoch >= self._end_annealing_epoch:
            self._m = self._target_m
        elif self._epoch < self._start_annealing_epoch:
            self._m = 0.0
        else:
            diff = (self._end_annealing_epoch - self._start_annealing_epoch)
            self._m = self._target_m * (self._epoch - self._start_annealing_epoch) / diff
