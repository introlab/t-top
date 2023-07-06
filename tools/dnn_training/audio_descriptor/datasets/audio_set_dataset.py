import os
import json

from audio_descriptor.datasets.multiclass_audio_descriptor_dataset import MulticlassAudioDescriptorDataset


class AudioSetDataset(MulticlassAudioDescriptorDataset):
    def _list_classes(self, root):
        with open(os.path.join(root, 'ontology.json')) as ontology_file:
            ontology_data = json.load(ontology_file)

        class_names = [d['id'] for d in ontology_data]
        class_names.sort()
        return {i: c for i, c in enumerate(class_names)}

    def _list_sounds(self, root, split, enhanced_targets):
        folder, filename = self._get_folder_and_sound_file(split, enhanced_targets)

        sounds = []
        with open(os.path.join(root, filename), 'r') as sound_file:
            for line in sound_file:
                values = line.split(' ')
                filename = values[0]
                class_names = (n.strip() for n in values[1:])
                sounds.append({
                    'path': os.path.join(folder, filename[:2], filename),
                    'target': self._create_target(class_names)
                })

        return sounds

    def _get_folder_and_sound_file(self, split, enhanced_targets):
        if split == 'training' and enhanced_targets:
            return 'train', 'train_enhanced.txt'
        elif split == 'training' and not enhanced_targets:
            return 'train', 'train.txt'
        elif split == 'validation' and enhanced_targets:
            return 'balanced_train', 'validation_enhanced.txt'
        elif split == 'validation' and not enhanced_targets:
            return 'balanced_train', 'validation.txt'
        elif split == 'testing':
            return 'eval', 'test.txt'
        else:
            raise ValueError('Invalid split')
