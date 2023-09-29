import random

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class MixupClassificationDataset(Dataset):
    def __init__(self, dataset, mixup_rate=0.5, mixup_alpha=10.0):
        self._dataset = dataset
        self._mixup_rate = mixup_rate
        self._mixup_alpha = mixup_alpha

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        image, class_index = self._dataset[index]
        target = F.one_hot(torch.tensor(class_index), self._dataset.class_count()).float()

        if random.random() < self._mixup_rate:
            mixup_index = random.randrange(len(self))
            mixup_image, mixup_class_index = self._dataset[mixup_index]
            mixup_target = F.one_hot(torch.tensor(mixup_class_index), self._dataset.class_count()).float()

            l = random.betavariate(self._mixup_alpha, self._mixup_alpha)
            image = l * image + (1.0 - l) * mixup_image
            target = l * target + (1.0 - l) * mixup_target

        return image, target


