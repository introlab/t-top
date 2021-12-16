import numpy as np

from torch.utils.data.sampler import Sampler


class TripletLossBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=300, image_per_class_count=20):
        super(TripletLossBatchSampler, self).__init__(dataset)

        self._dataset = dataset
        self._image_per_class_count = image_per_class_count
        self._class_per_batch_count = batch_size // self._image_per_class_count

    def __iter__(self):
        batch_size = self._image_per_class_count * self._class_per_batch_count

        lens_by_class = self._dataset.lens_by_class()
        image_count_by_class = [0 for _ in lens_by_class]
        count = 0

        batch = []
        while count < len(self._dataset):
            class_order = np.random.permutation(len(lens_by_class)).tolist()

            for i in class_order:
                image_count = 0
                while image_count < self._image_per_class_count and image_count_by_class[i] < lens_by_class[i]:
                    batch.append(self._dataset.get_all_indexes(i, image_count_by_class[i]))
                    image_count_by_class[i] += 1
                    count += 1
                    image_count += 1

                    if len(batch) >= batch_size:
                        break
                if len(batch) >= batch_size:
                    break

            if len(batch) == batch_size:
                yield batch
                batch = []

    def __len__(self):
        return len(self._dataset) // (self._class_per_batch_count * self._image_per_class_count)
