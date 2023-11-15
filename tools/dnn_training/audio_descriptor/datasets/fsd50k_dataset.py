import os
import csv

from audio_descriptor.datasets.multiclass_audio_descriptor_dataset import MulticlassAudioDescriptorDataset


class Fsd50kDataset(MulticlassAudioDescriptorDataset):
    def _list_classes(self, root):
        class_indexes_by_name = {}
        with open(os.path.join(root, 'FSD50K.ground_truth', 'vocabulary.csv'), newline='') as vocabulary_file:
            vocabulary_reader = csv.reader(vocabulary_file, delimiter=',', quotechar='"')
            for class_index, class_name, _ in vocabulary_reader:
                class_indexes_by_name[class_name] = int(class_index)

        return class_indexes_by_name

    def _list_sounds(self, root, split, enhanced_targets):
        folder, filename = self._get_folder_and_sound_file(split, enhanced_targets)

        sounds = []
        with open(os.path.join(root, filename), 'r') as sound_file:
            for line in sound_file:
                values = line.split(' ')
                class_names = (n.strip() for n in values[1:])
                sounds.append({
                    'path': os.path.join(folder, values[0]),
                    'target': self._create_target(class_names)
                })

        return sounds

    def _get_folder_and_sound_file(self, split, enhanced_targets):
        if split == 'training' and enhanced_targets:
            return 'FSD50K.dev_audio', 'train_enhanced.txt'
        elif split == 'training' and not enhanced_targets:
            return 'FSD50K.dev_audio', 'train.txt'
        elif split == 'validation' and enhanced_targets:
            return 'FSD50K.dev_audio', 'validation_enhanced.txt'
        elif split == 'validation' and not enhanced_targets:
            return 'FSD50K.dev_audio', 'validation.txt'
        elif split == 'testing':
            return 'FSD50K.eval_audio', 'test.txt'
        else:
            raise ValueError('Invalid split')
