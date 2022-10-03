import os
import time

from tqdm import tqdm

import torch
import torch.nn as nn

from common.metrics import ClassificationAccuracyMetric, LossMetric, LossAccuracyLearningCurves
from common.trainers import Trainer

from keyword_spotting.datasets import GoogleSpeechCommands, TtopKeyword, \
    KeywordSpottingTrainingTransforms, KeywordSpottingValidationTransforms


class KeywordSpotterTrainer(Trainer):
    def __init__(self, device, model, dataset_type='google_speech_commands', mfcc_feature_count=40, dataset_root='',
                 output_path='', epoch_count=10, learning_rate=0.01, weight_decay=0.0,
                 batch_size=128, batch_size_division=4,
                 model_checkpoint=None):
        self._dataset_type = dataset_type
        self._mfcc_feature_count = mfcc_feature_count

        super(KeywordSpotterTrainer, self).__init__(device, model,
                                                    dataset_root=dataset_root,
                                                    output_path=output_path,
                                                    epoch_count=epoch_count,
                                                    learning_rate=learning_rate,
                                                    weight_decay=weight_decay,
                                                    batch_size=batch_size,
                                                    batch_size_division=batch_size_division,
                                                    model_checkpoint=model_checkpoint)

        self._training_loss_metric = LossMetric()
        self._training_accuracy_metric = ClassificationAccuracyMetric()
        self._validation_loss_metric = LossMetric()
        self._validation_accuracy_metric = ClassificationAccuracyMetric()
        self._learning_curves = LossAccuracyLearningCurves()

    def _create_criterion(self, model):
        return create_criterion()

    def _create_training_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transforms = KeywordSpottingTrainingTransforms(get_noise_root(self._dataset_type, dataset_root),
                                                       n_mfcc=self._mfcc_feature_count)

        dataset = self._create_dataset(dataset_root, 'training', transforms)

        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size // batch_size_division,
                                           shuffle=True,
                                           num_workers=8)

    def _create_validation_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transforms = KeywordSpottingValidationTransforms(n_mfcc=self._mfcc_feature_count)

        dataset = self._create_dataset(dataset_root, 'validation', transforms)

        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size // batch_size_division,
                                           shuffle=True,
                                           num_workers=8)

    def _create_testing_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transforms = KeywordSpottingValidationTransforms(n_mfcc=self._mfcc_feature_count)

        dataset = self._create_dataset(dataset_root, 'testing', transforms)

        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size // batch_size_division,
                                           shuffle=True,
                                           num_workers=8)

    def _create_dataset(self, dataset_root, split, transforms):
        return create_dataset(self._dataset_type, dataset_root, split, transforms)

    def _clear_between_training(self):
        self._learning_curves.clear()

    def _clear_between_training_epoch(self):
        self._training_loss_metric.clear()
        self._training_accuracy_metric.clear()

    def _move_target_to_device(self, target, device):
        return move_target_to_device(target, device)

    def _measure_training_metrics(self, loss, predicted_class_scores, target):
        self._training_loss_metric.add(loss.item())
        self._training_accuracy_metric.add(predicted_class_scores, target)

    def _clear_between_validation_epoch(self):
        self._validation_loss_metric.clear()
        self._validation_accuracy_metric.clear()

    def _measure_validation_metrics(self, loss, predicted_class_scores, target):
        self._validation_loss_metric.add(loss.item())
        self._validation_accuracy_metric.add(predicted_class_scores, target)

    def _print_performances(self):
        print('\nTraining : Loss={}, Accuracy={}'.format(self._training_loss_metric.get_loss(),
                                                         self._training_accuracy_metric.get_accuracy()))
        print('Validation : Loss={}, Accuracy={}\n'.format(self._validation_loss_metric.get_loss(),
                                                           self._validation_accuracy_metric.get_accuracy()))

    def _save_learning_curves(self):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_training_accuracy_value(self._training_accuracy_metric.get_accuracy())
        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())
        self._learning_curves.add_validation_accuracy_value(self._validation_accuracy_metric.get_accuracy())

        self._learning_curves.save(os.path.join(self._output_path, 'learning_curves.png'),
                                   os.path.join(self._output_path, 'learning_curves.json'))

    def _evaluate(self, model, device, dataset_loader, output_path):
        print('Evaluation', flush=True)
        evaluate(model, device, dataset_loader)


def create_criterion():
    return nn.CrossEntropyLoss()


def get_noise_root(dataset_type, dataset_root):
    if dataset_type == 'google_speech_commands':
        return os.path.join(dataset_root, '_background_noise_')
    elif dataset_type == 'ttop_keyword':
        return os.path.join(dataset_root, 'train', 'noise')
    else:
        raise ValueError('Invalid dataset')


def create_dataset(dataset_type, dataset_root, split, transforms):
    if dataset_type == 'google_speech_commands':
        return GoogleSpeechCommands(dataset_root, split=split, transforms=transforms)
    elif dataset_type == 'ttop_keyword':
        return TtopKeyword(dataset_root, split=split, transforms=transforms)
    else:
        raise ValueError('Invalid dataset')


def move_target_to_device(target, device):
    return target.to(device)


def evaluate(model, device, dataset_loader):
    criterion = create_criterion()
    loss_metric = LossMetric()
    accuracy_metric = ClassificationAccuracyMetric()

    for data in tqdm(dataset_loader):
        model_output = model(data[0].to(device))
        target = move_target_to_device(data[1], device)
        loss = criterion(model_output, target)

        loss_metric.add(loss.item())
        accuracy_metric.add(model_output, target)

    print('Loss={}, Accuracy={}'.format(loss_metric.get_loss(), accuracy_metric.get_accuracy()))
