import os

import torch
import torch.nn as nn
import torch.utils.data

from tqdm import tqdm

from common.trainers import Trainer
from common.criterions import SigmoidFocalLossWithLogits
from common.metrics import MulticlassClassificationAccuracyMetric, MulticlassClassificationPrecisionRecallMetric, \
    LossMetric, LossAccuracyLearningCurves, MulticlassClassificationMeanAveragePrecisionMetric

from audio_descriptor.datasets import Fsd50kDataset, AudioDescriptorTrainingTransforms, \
    AudioDescriptorValidationTransforms
from audio_descriptor.metrics import AudioDescriptorEvaluation


class MulticlassAudioDescriptorExtractorTrainer(Trainer):
    def __init__(self, device, model, dataset_root='', output_path='', epoch_count=10, learning_rate=0.01,
                 batch_size=128, criterion_type='bce_loss',
                 waveform_size=64000, n_features=128, n_fft=400, audio_transform_type='mel_spectrogram',
                 enable_pitch_shifting=False, enable_time_stretching=False,
                 model_checkpoint=None, optimizer_checkpoint=None, scheduler_checkpoint=None):
        self._criterion_type = criterion_type
        self._waveform_size = waveform_size
        self._n_features = n_features
        self._n_fft = n_fft
        self._audio_transform_type = audio_transform_type
        self._enable_pitch_shifting = enable_pitch_shifting
        self._enable_time_stretching = enable_time_stretching
        self._class_count = model.class_count()
        super(MulticlassAudioDescriptorExtractorTrainer, self).__init__(device, model,
                                                              dataset_root=dataset_root,
                                                              output_path=output_path,
                                                              epoch_count=epoch_count,
                                                              learning_rate=learning_rate,
                                                              batch_size=batch_size,
                                                              batch_size_division=1,
                                                              model_checkpoint=model_checkpoint,
                                                              optimizer_checkpoint=optimizer_checkpoint,
                                                              scheduler_checkpoint=scheduler_checkpoint)

        self._dataset_root = dataset_root

        self._training_loss_metric = LossMetric()
        self._validation_loss_metric = LossMetric()

        self._learning_curves = LossAccuracyLearningCurves()
        self._training_accuracy_metric = MulticlassClassificationAccuracyMetric()
        self._validation_accuracy_metric = MulticlassClassificationAccuracyMetric()
        self._training_precision_recall_metric = MulticlassClassificationPrecisionRecallMetric()
        self._validation_precision_recall_metric = MulticlassClassificationPrecisionRecallMetric()

    def _create_criterion(self, model):
        if self._criterion_type == 'bce_loss':
            criterion = nn.BCEWithLogitsLoss()
            return lambda model_output, target: criterion(model_output[1], target)
        elif self._criterion_type == 'sigmoid_focal_loss':
            criterion = SigmoidFocalLossWithLogits()
            return lambda model_output, target: criterion(model_output[1], target)
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
                                                       enable_time_stretching=self._enable_time_stretching)
        return self._create_dataset_loader(dataset_root, batch_size, batch_size_division, 'training', transforms)

    def _create_validation_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transforms = AudioDescriptorValidationTransforms(waveform_size=self._waveform_size,
                                                         n_features=self._n_features,
                                                         n_fft=self._n_fft,
                                                         audio_transform_type=self._audio_transform_type)
        return self._create_dataset_loader(dataset_root, batch_size, batch_size_division, 'validation', transforms)

    def _create_dataset_loader(self, dataset_root, batch_size, batch_size_division, split, transforms):
        dataset = Fsd50kDataset(dataset_root, split=split, transforms=transforms)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size // batch_size_division, shuffle=True,
                                           num_workers=4)

    def _clear_between_training(self):
        self._learning_curves.clear()

    def _clear_between_training_epoch(self):
        self._training_loss_metric.clear()
        self._training_accuracy_metric.clear()
        self._training_precision_recall_metric.clear()

    def _move_target_to_device(self, target, device):
        return target.to(device)

    def _measure_training_metrics(self, loss, model_output, target):
        self._training_loss_metric.add(loss.item())
        self._training_accuracy_metric.add(model_output[1], target)
        self._training_precision_recall_metric.add(model_output[1], target)

    def _clear_between_validation_epoch(self):
        self._validation_loss_metric.clear()
        self._validation_accuracy_metric.clear()
        self._validation_precision_recall_metric.clear()

    def _measure_validation_metrics(self, loss, model_output, target):
        self._validation_loss_metric.add(loss.item())
        self._validation_accuracy_metric.add(model_output[1], target)
        self._validation_precision_recall_metric.add(model_output[1], target)

    def _print_performances(self):
        print('\nTraining : Loss={}, Accuracy={}, Precision={}, Recall={}'.format(
            self._training_loss_metric.get_loss(),
            self._training_accuracy_metric.get_accuracy(),
            self._training_precision_recall_metric.get_precision(),
            self._training_precision_recall_metric.get_recall()))
        print('Validation : Loss={}, Accuracy={}, Precision={}, Recall={}\n'.format(
            self._validation_loss_metric.get_loss(),
            self._validation_accuracy_metric.get_accuracy(),
            self._validation_precision_recall_metric.get_precision(),
            self._validation_precision_recall_metric.get_recall()))

    def _save_learning_curves(self):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())
        self._learning_curves.add_training_accuracy_value(self._training_accuracy_metric.get_accuracy())
        self._learning_curves.add_validation_accuracy_value(self._validation_accuracy_metric.get_accuracy())

        self._learning_curves.save_figure(os.path.join(self._output_path, 'learning_curves.png'))

    def _evaluate(self, model, device, dataset_loader, output_path):
        print('Evaluation - ROC', flush=True)

        evaluation = AudioDescriptorEvaluation(model, device, dataset_loader.dataset.transforms(), self._dataset_root,
                                               output_path)
        evaluation.evaluate()

        self._evaluate_classification_accuracy(model, device, dataset_loader)

    def _evaluate_classification_accuracy(self, model, device, dataset_loader):
        print('Evaluation - Classification')
        accuracy_metric = MulticlassClassificationAccuracyMetric()
        precision_recall_metric = MulticlassClassificationPrecisionRecallMetric()
        map_metric = MulticlassClassificationMeanAveragePrecisionMetric(self._class_count)

        for data in tqdm(dataset_loader):
            model_output = model(data[0].to(device))
            target = self._move_target_to_device(data[1], device)
            accuracy_metric.add(model_output[1], target)
            precision_recall_metric.add(model_output[1], target)
            map_metric.add(model_output[1], target)

        print('\nTest : Accuracy={}, Precision={}, Recall={}, mAP={}'.format(accuracy_metric.get_accuracy(),
                                                                             precision_recall_metric.get_precision(),
                                                                             precision_recall_metric.get_recall(),
                                                                             map_metric.get_value()))
