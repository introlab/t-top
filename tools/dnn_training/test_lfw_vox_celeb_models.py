import argparse
import os
import time

import numpy as np
from PIL import Image

import torch
import torchaudio

from tqdm import tqdm

from common.modules import load_checkpoint
from common.metrics.roc_evaluation import RocDistancesThresholdsEvaluation

from face_recognition.trainers.face_descriptor_extractor_trainer import create_validation_image_transform
from train_face_descriptor_extractor import create_model as create_face_model

from audio_descriptor.datasets import AudioDescriptorTestTransforms
from train_audio_descriptor_extractor import create_model as create_voice_model


class LfwVoxCelebEvaluation(RocDistancesThresholdsEvaluation):
    def __init__(self, device, lfw_dataset_root, vox_celeb_dataset_root, pairs_file, output_path,
                 face_model, face_transforms, voice_model, voice_transforms):
        super().__init__(output_path, thresholds=np.arange(0, 10, 0.00001))
        self._device = device
        self._lfw_dataset_root = lfw_dataset_root
        self._vox_celeb_dataset_root = vox_celeb_dataset_root
        self._pairs_file = pairs_file

        self._face_model = face_model
        self._face_transforms = face_transforms
        self._voice_model = voice_model
        self._voice_transforms = voice_transforms

        self._pairs = self._read_pairs()

    def _read_pairs(self):
        pairs = []
        with open(self._pairs_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            sections = line.strip().split(' ')
            voice_class_name_0 = sections[0]
            voice_video_id_0 = sections[1]
            voice_file_0 = sections[2]
            voice_class_name_1 = sections[3]
            voice_video_id_1 = sections[4]
            voice_file_1 = sections[5]
            voice_path_0 = os.path.join(self._vox_celeb_dataset_root, 'sounds', voice_class_name_0, voice_video_id_0,
                                        voice_file_0)
            voice_path_1 = os.path.join(self._vox_celeb_dataset_root, 'sounds', voice_class_name_1, voice_video_id_1,
                                        voice_file_1)

            face_class_name_0 = sections[6]
            face_file_0 = sections[7]
            face_class_name_1 = sections[8]
            face_file_1 = sections[9]
            face_path_0 = os.path.join(self._lfw_dataset_root, face_class_name_0, face_file_0)
            face_path_1 = os.path.join(self._lfw_dataset_root, face_class_name_1, face_file_1)

            if not os.path.exists(voice_path_0) or not os.path.exists(voice_path_1) or \
                    not os.path.exists(face_path_0) or not os.path.exists(face_path_1):
                raise ValueError('Invalid paths')

            if voice_class_name_0 == voice_class_name_1 and face_class_name_0 == face_class_name_1:
                is_same_person = True
            elif voice_class_name_0 != voice_class_name_1 and face_class_name_0 != face_class_name_1:
                is_same_person = False
            else:
                raise ValueError('Invalid class association')

            pairs.append((voice_path_0, voice_path_1, face_path_0, face_path_1, is_same_person))

        return pairs

    def evaluate(self):
        print('Calculate distances')
        face_distances, voice_distances, face_voice_distances = self._calculate_distances()
        is_same_person_target = self._get_is_same_person_target()

        self._evaluate(face_distances, is_same_person_target, 'face_')
        self._evaluate(voice_distances, is_same_person_target, 'voice_')
        self._evaluate(face_voice_distances, is_same_person_target, 'face_voice_')

    def _evaluate(self, distances, is_same_person_target, prefix):
        best_accuracy, best_threshold, true_positive_rate_curve, false_positive_rate_curve, thresholds = \
            self._calculate_accuracy_true_positive_rate_false_positive_rate(distances, is_same_person_target)
        auc = self._calculate_auc(true_positive_rate_curve, false_positive_rate_curve)
        eer = self._calculate_eer(true_positive_rate_curve, false_positive_rate_curve)

        print(prefix)
        print('Best accuracy: {}, threshold: {}, AUC: {}, EER: {}'.format(best_accuracy, best_threshold, auc, eer))
        print()

        self._save_roc_curve(true_positive_rate_curve, false_positive_rate_curve, prefix=prefix)
        self._save_roc_curve_data(true_positive_rate_curve, false_positive_rate_curve, thresholds, prefix=prefix)
        self._save_performances({
            'best_accuracy': best_accuracy,
            'best_threshold': best_threshold,
            'auc': auc,
            'eer': eer
        }, prefix=prefix)

    def _calculate_distances(self):
        face_distances = []
        voice_distances = []
        face_voice_distances = []

        for voice_path_0, voice_path_1, face_path_0, face_path_1, _ in tqdm(self._pairs):
            voice_sound_0 = self._load_voice_sound(voice_path_0).to(self._device)
            voice_sound_1 = self._load_voice_sound(voice_path_1).to(self._device)
            face_image_0 = self._load_face_image(face_path_0).to(self._device)
            face_image_1 = self._load_face_image(face_path_1).to(self._device)

            voice_descriptor_0 = self._voice_model(voice_sound_0)
            voice_descriptor_1 = self._voice_model(voice_sound_1)
            face_descriptors = self._face_model(torch.stack((face_image_0, face_image_1)))

            face_distance = torch.dist(face_descriptors[0], face_descriptors[1], p=2).item()
            voice_distance = torch.dist(voice_descriptor_0[0], voice_descriptor_1[0], p=2).item()
            face_voice_distance = torch.dist(torch.cat((voice_descriptor_0[0], face_descriptors[0])),
                                             torch.cat((voice_descriptor_1[0], face_descriptors[1])), p=2).item()
            face_distances.append(face_distance)
            voice_distances.append(voice_distance)
            face_voice_distances.append(face_voice_distance)

        return torch.tensor(face_distances), torch.tensor(voice_distances), torch.tensor(face_voice_distances)

    def _load_voice_sound(self, path):
        waveform, sample_rate = torchaudio.load(path)
        class_index = 0

        metadata = {
            'original_sample_rate': sample_rate
        }

        if self._voice_transforms is not None:
            waveform, _, _ = self._voice_transforms(waveform, class_index, metadata)

        return waveform

    def _load_face_image(self, path):
        image = Image.open(path).convert('RGB')
        if self._face_transforms is not None:
            image = self._face_transforms(image)

        return image

    def _get_is_same_person_target(self):
        return torch.tensor([pair[4] for pair in self._pairs])


def main():
    parser = argparse.ArgumentParser(description='Test exported face descriptor extractor')
    parser.add_argument('--use_gpu', action='store_true')

    parser.add_argument('--lfw_dataset_root', type=str, help='Choose the lfw dataset root path', required=True)
    parser.add_argument('--vox_celeb_dataset_root', type=str, help='Choose the vox celeb dataset root path',
                        required=True)
    parser.add_argument('--pairs_file', type=str, help='Choose the file that contains the pairs', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)

    parser.add_argument('--face_embedding_size', type=int, help='Set the face embedding size', required=True)
    parser.add_argument('--face_model_checkpoint', type=str, help='Choose the face model checkpoint path',
                        required=True)

    parser.add_argument('--voice_backbone_type', choices=['mnasnet0.5', 'mnasnet1.0',
                                                          'resnet18', 'resnet34', 'resnet50',
                                                          'open_face_inception', 'thin_resnet_34',
                                                          'ecapa_tdnn_512', 'ecapa_tdnn_1024',
                                                          'small_ecapa_tdnn_128', 'small_ecapa_tdnn_256',
                                                          'small_ecapa_tdnn_512'],
                        help='Choose the backbone type', required=True)
    parser.add_argument('--voice_embedding_size', type=int, help='Set the voice embedding size', required=True)
    parser.add_argument('--voice_pooling_layer', choices=['avg', 'vlad', 'sap'], help='Set the voice pooling layer')
    parser.add_argument('--voice_waveform_size', type=int, help='Set the voice waveform size', required=True)
    parser.add_argument('--voice_n_features', type=int, help='Set voice n_features', required=True)
    parser.add_argument('--voice_n_fft', type=int, help='Set voice n_fft', required=True)
    parser.add_argument('--voice_audio_transform_type', choices=['mfcc', 'mel_spectrogram', 'spectrogram'],
                        help='Choose the voice audio transform type', required=True)
    parser.add_argument('--voice_model_checkpoint', type=str, help='Choose the voice model checkpoint path',
                        required=True)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    face_model = create_face_model(args.face_embedding_size).to(device)
    load_checkpoint(face_model, args.face_model_checkpoint, keys_to_remove=['_classifier._weight'])
    face_model.eval()
    face_transforms = create_validation_image_transform()

    voice_model = create_voice_model(args.voice_backbone_type, args.voice_n_features, args.voice_embedding_size,
                                     pooling_layer=args.voice_pooling_layer).to(device)
    load_checkpoint(voice_model, args.voice_model_checkpoint, keys_to_remove=['_classifier._weight'])
    voice_model.eval()
    voice_transforms = AudioDescriptorTestTransforms(waveform_size=args.voice_waveform_size,
                                                     n_features=args.voice_n_features,
                                                     n_fft=args.voice_n_fft,
                                                     audio_transform_type=args.voice_audio_transform_type)

    evaluation = LfwVoxCelebEvaluation(device,
                                       args.lfw_dataset_root,
                                       args.vox_celeb_dataset_root,
                                       args.pairs_file,
                                       args.output_path,
                                       face_model,
                                       face_transforms,
                                       voice_model,
                                       voice_transforms)
    evaluation.evaluate()


if __name__ == '__main__':
    main()
