import os

from tqdm import tqdm

from common.trainers import Trainer
from common.metrics import LossMetric, LossLearningCurves

from ego_noise.datasets import LibriSpeechDataset, EgoNoiseAutoencoderTransforms

import torch
import torch.nn as nn
import torch.utils.data


class EgoNoiseAutoencoderTrainer(Trainer):
    def __init__(self, device, model, dataset_root='', output_path='', epoch_count=10, learning_rate=0.01,
                 batch_size=128, sample_rate=44100, n_fft=2048,
                 model_checkpoint=None, optimizer_checkpoint=None, scheduler_checkpoint=None):
        self._sample_rate = sample_rate
        self._n_fft = n_fft

        super(EgoNoiseAutoencoderTrainer, self).__init__(device, model,
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
        self._learning_curves = LossLearningCurves()

    def _create_criterion(self, model):
        return create_criterion()

    def _create_training_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transforms = EgoNoiseAutoencoderTransforms(sample_rate=self._sample_rate, n_fft=self._n_fft)
        dataset = LibriSpeechDataset(dataset_root, split='training', transforms=transforms)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size // batch_size_division,
                                           collate_fn=self._collate,
                                           shuffle=True,
                                           num_workers=4)

    def _create_validation_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transforms = EgoNoiseAutoencoderTransforms(sample_rate=self._sample_rate, n_fft=self._n_fft)
        dataset = LibriSpeechDataset(dataset_root, split='validation', transforms=transforms)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size // batch_size_division,
                                           collate_fn=self._collate,
                                           shuffle=True,
                                           num_workers=4)

    def _create_testing_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transforms = EgoNoiseAutoencoderTransforms(sample_rate=self._sample_rate, n_fft=self._n_fft)
        dataset = LibriSpeechDataset(dataset_root, split='testing', transforms=transforms)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size // batch_size_division,
                                           collate_fn=self._collate,
                                           shuffle=True,
                                           num_workers=4)


    def _collate(self, batch):
        magnitudes = torch.cat([e[0] for e in batch], 0)
        targets = torch.cat([e[1] for e in batch], 0)
        return magnitudes, targets

    def _clear_between_training(self):
        self._learning_curves.clear()

    def _clear_between_training_epoch(self):
        self._training_loss_metric.clear()

    def _move_target_to_device(self, target, device):
        return target.to(device)

    def _measure_training_metrics(self, loss, embedding, target):
        self._training_loss_metric.add(loss.item())

    def _clear_between_validation_epoch(self):
        self._validation_loss_metric.clear()

    def _measure_validation_metrics(self, loss, embedding, target):
        self._validation_loss_metric.add(loss.item())

    def _print_performances(self):
        print('\nTraining : Loss={}'.format(self._training_loss_metric.get_loss()))
        print('Validation : Loss={}\n'.format(self._validation_loss_metric.get_loss()))

    def _save_learning_curves(self):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())

        self._learning_curves.save_figure(os.path.join(self._output_path, 'learning_curves.png'))

    def _evaluate(self, model, device, dataset_loader, output_path):
        print('Evaluation', flush=True)
        evaluate(model, device, dataset_loader)


def create_criterion():
    return nn.MSELoss()


def evaluate(model, device, dataset_loader):
    criterion = create_criterion()
    loss_metric = LossMetric()

    for data in tqdm(dataset_loader):
        model_output = model(data[0].to(device))
        loss = criterion(model_output, data[1].to(device))

        loss_metric.add(loss.item())

    print('Loss={}'.format(loss_metric.get_loss()))
