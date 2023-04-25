import os

import torch
import torch.nn as nn
import torch.utils.data

from tqdm import tqdm

from common.criterions import TripletLoss
from common.datasets import TripletLossBatchSampler
from common.trainers import Trainer
from common.metrics import ClassificationAccuracyMetric, LossMetric, LossLearningCurves, LossAccuracyLearningCurves, \
    TopNClassificationAccuracyMetric, ClassificationMeanAveragePrecisionMetric

from audio_descriptor.criterions import AudioDescriptorAmSoftmaxLoss
from audio_descriptor.datasets import AudioDescriptorDataset, \
    AudioDescriptorTrainingTransforms, AudioDescriptorValidationTransforms, AudioDescriptorTestTransforms
from audio_descriptor.metrics import AudioDescriptorEvaluation


class AudioDescriptorExtractorTrainer(Trainer):
    def __init__(self, device, model, dataset_root='', output_path='',
                 epoch_count=10, learning_rate=0.01, weight_decay=0.0, batch_size=128, criterion_type='triplet_loss',
                 waveform_size=64000, n_features=128, n_fft=400, audio_transform_type='mel_spectrogram',
                 enable_pitch_shifting=False, enable_time_stretching=False,
                 enable_time_masking=False, enable_frequency_masking=False, margin=0.2,
                 model_checkpoint=None):
        self._criterion_type = criterion_type
        self._waveform_size = waveform_size
        self._n_features = n_features
        self._n_fft = n_fft
        self._audio_transform_type = audio_transform_type
        self._enable_pitch_shifting = enable_pitch_shifting
        self._enable_time_stretching = enable_time_stretching
        self._enable_time_masking = enable_time_masking
        self._enable_frequency_masking = enable_frequency_masking
        self._margin = margin
        self._class_count = model.class_count()
        super(AudioDescriptorExtractorTrainer, self).__init__(device, model,
                                                              dataset_root=dataset_root,
                                                              output_path=output_path,
                                                              epoch_count=epoch_count,
                                                              learning_rate=learning_rate,
                                                              weight_decay=weight_decay,
                                                              batch_size=batch_size,
                                                              batch_size_division=1,
                                                              model_checkpoint=model_checkpoint)

        self._dataset_root = dataset_root

        self._training_loss_metric = LossMetric()
        self._validation_loss_metric = LossMetric()

        if self._criterion_type == 'triplet_loss':
            self._learning_curves = LossLearningCurves()
        else:
            self._learning_curves = LossAccuracyLearningCurves()
            self._training_accuracy_metric = ClassificationAccuracyMetric()
            self._validation_accuracy_metric = ClassificationAccuracyMetric()

    def _create_criterion(self, model):
        if self._criterion_type == 'triplet_loss':
            return TripletLoss(margin=self._margin)
        elif self._criterion_type == 'cross_entropy_loss':
            criterion = nn.CrossEntropyLoss()
            return lambda model_output, target: criterion(model_output[1], target)
        elif self._criterion_type == 'am_softmax_loss':
            return AudioDescriptorAmSoftmaxLoss(s=30.0, m=self._margin,
                                                start_annealing_epoch=0,
                                                end_annealing_epoch=self._epoch_count // 4)
        else:
            raise ValueError('Invalid criterion type')

    def _create_training_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transforms = AudioDescriptorTrainingTransforms(waveform_size=self._waveform_size,
                                                       n_features=self._n_features,
                                                       n_fft=self._n_fft,
                                                       noise_root=os.path.join(dataset_root, 'noises'),
                                                       noise_volume=0.25,
                                                       noise_p=0.5,
                                                       audio_transform_type=self._audio_transform_type,
                                                       enable_pitch_shifting=self._enable_pitch_shifting,
                                                       enable_time_stretching=self._enable_time_stretching,
                                                       enable_time_masking=self._enable_time_masking,
                                                       enable_frequency_masking=self._enable_frequency_masking)
        return self._create_dataset_loader(dataset_root, batch_size, batch_size_division, 'training', transforms)

    def _create_validation_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transforms = AudioDescriptorValidationTransforms(waveform_size=self._waveform_size,
                                                         n_features=self._n_features,
                                                         n_fft=self._n_fft,
                                                         audio_transform_type=self._audio_transform_type)
        return self._create_dataset_loader(dataset_root, batch_size, batch_size_division, 'validation', transforms)

    def _create_testing_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transforms = AudioDescriptorTestTransforms(waveform_size=self._waveform_size,
                                                   n_features=self._n_features,
                                                   n_fft=self._n_fft,
                                                   audio_transform_type=self._audio_transform_type)
        return self._create_dataset_loader(dataset_root, 1, 1, 'testing', transforms)

    def _create_dataset_loader(self, dataset_root, batch_size, batch_size_division, split, transforms):
        dataset = AudioDescriptorDataset(dataset_root, split=split, transforms=transforms)

        if self._criterion_type == 'triplet_loss':
            batch_sampler = TripletLossBatchSampler(dataset, batch_size=batch_size // batch_size_division)
            return torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=1)
        else:
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size // batch_size_division, shuffle=True,
                                               num_workers=4)

    def _clear_between_training(self):
        self._learning_curves.clear()

    def _clear_between_training_epoch(self):
        self._training_loss_metric.clear()
        if self._criterion_type != 'triplet_loss':
            self._training_accuracy_metric.clear()

    def _move_target_to_device(self, target, device):
        return target.to(device)

    def _measure_training_metrics(self, loss, model_output, target):
        self._training_loss_metric.add(loss.item())
        if self._criterion_type != 'triplet_loss':
            self._training_accuracy_metric.add(model_output[1], target)

    def _clear_between_validation_epoch(self):
        self._validation_loss_metric.clear()
        if self._criterion_type != 'triplet_loss':
            self._validation_accuracy_metric.clear()

    def _measure_validation_metrics(self, loss, model_output, target):
        self._validation_loss_metric.add(loss.item())
        if self._criterion_type != 'triplet_loss':
            self._validation_accuracy_metric.add(model_output[1], target)

    def _print_performances(self):
        if self._criterion_type != 'triplet_loss':
            print('\nTraining : Loss={}, Accuracy={}'.format(self._training_loss_metric.get_loss(),
                                                             self._training_accuracy_metric.get_accuracy()))
            print('Validation : Loss={}, Accuracy={}\n'.format(self._validation_loss_metric.get_loss(),
                                                               self._validation_accuracy_metric.get_accuracy()))
        else:
            print('\nTraining : Loss={}'.format(self._training_loss_metric.get_loss()))
            print('Validation : Loss={}\n'.format(self._validation_loss_metric.get_loss()))

    def _save_learning_curves(self):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())
        if self._criterion_type != 'triplet_loss':
            self._learning_curves.add_training_accuracy_value(self._training_accuracy_metric.get_accuracy())
            self._learning_curves.add_validation_accuracy_value(self._validation_accuracy_metric.get_accuracy())

        self._learning_curves.save(os.path.join(self._output_path, 'learning_curves.png'),
                                   os.path.join(self._output_path, 'learning_curves.json'))

    def _evaluate(self, model, device, dataset_loader, output_path):
        print('Evaluation - ROC', flush=True)

        evaluation = AudioDescriptorEvaluation(model, device, dataset_loader.dataset.transforms(), self._dataset_root,
                                               output_path)
        evaluation.evaluate()

        if self._criterion_type != 'triplet_loss':
            self._evaluate_classification_accuracy(model, device, dataset_loader)

    def _evaluate_classification_accuracy(self, model, device, dataset_loader):
        print('Evaluation - Classification')
        top1_accuracy_metric = ClassificationAccuracyMetric()
        top5_accuracy_metric = TopNClassificationAccuracyMetric(5)
        map_metric = ClassificationMeanAveragePrecisionMetric(self._class_count)

        for data in tqdm(dataset_loader):
            model_output = model(data[0].to(device))
            target = self._move_target_to_device(data[1], device)
            top1_accuracy_metric.add(model_output[1], target)
            top5_accuracy_metric.add(model_output[1], target)
            map_metric.add(model_output[1], target)

        print('\nTest : Top 1 Accuracy={}, Top 5 Accuracy={}, mAP={}'.format(top1_accuracy_metric.get_accuracy(),
                                                                             top5_accuracy_metric.get_accuracy(),
                                                                             map_metric.get_value()))
