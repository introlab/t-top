import torch.nn as nn
import torch.nn.functional as F


# Inspired by https://discuss.pytorch.org/t/jensen-shannon-divergence/2626/10
class JensenShannonDivergence(nn.Module):
    def forward(self, logits_1, logits_2):
        probabilities_1 = F.softmax(logits_1, dim=1)
        probabilities_2 = F.softmax(logits_2, dim=1)

        target_m = (probabilities_1 + probabilities_2) / 2.0
        log_target_m = target_m.log()
        loss = F.kl_div(log_target_m, F.log_softmax(logits_1, dim=1), reduction='batchmean', log_target=True) + \
               F.kl_div(log_target_m, F.log_softmax(logits_2, dim=1), reduction='batchmean', log_target=True)

        return loss / 2.0
