import torch


class ImbalancedFaceDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, face_dataset):
        self._image_count = len(face_dataset)

        class_weights = [1.0 / (c + 1e-6) for c in face_dataset.lens_by_class()]
        self._image_weights = [class_weights[class_index] for class_index in face_dataset.class_indexes()]
        self._image_weights = torch.tensor(self._image_weights)

    def __iter__(self):
        indexes = torch.multinomial(self._image_weights, self._image_count, replacement=True)
        return iter(indexes.tolist())

    def __len__(self):
        return self._image_count
