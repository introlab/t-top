import torch


class ImbalancedMulticlassAudioDescriptorDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, multiclass_audio_descriptor_dataset):
        self._sound_count = len(multiclass_audio_descriptor_dataset)

        class_counts = torch.ones(multiclass_audio_descriptor_dataset.class_count()) * 1e-6
        for i in range(self._sound_count):
            class_counts += multiclass_audio_descriptor_dataset.get_target(i)

        class_weights = 1.0 / class_counts
        self._image_weights = []
        for i in range(self._sound_count):
            self._image_weights.append((class_weights * multiclass_audio_descriptor_dataset.get_target(i)).sum())
        self._image_weights = torch.tensor(self._image_weights)

    def __iter__(self):
        indexes = torch.multinomial(self._image_weights, self._sound_count, replacement=True)
        return iter(indexes.tolist())

    def __len__(self):
        return self._sound_count
