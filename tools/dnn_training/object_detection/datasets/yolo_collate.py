import torch
import torch.utils.data


def yolo_collate(batch):
    images = torch.utils.data.dataloader.default_collate([e[0] for e in batch])
    target = [e[1] for e in batch]
    metadata = torch.utils.data.dataloader.default_collate([e[2] for e in batch])

    return images, target, metadata
