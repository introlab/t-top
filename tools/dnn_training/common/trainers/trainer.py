import os

import torch
import torch.nn as nn

from tqdm import tqdm

from common.modules import load_checkpoint


class Trainer:
    def __init__(self, device, model, dataset_root='', output_path='',
                 epoch_count=10, learning_rate=0.01, weight_decay=0.0, batch_size=128, batch_size_division=4,
                 model_checkpoint=None):
        self._device = device
        self._output_path = output_path
        os.makedirs(self._output_path, exist_ok=True)

        self._epoch_count = epoch_count
        self._batch_size = batch_size
        self._batch_size_division = batch_size_division

        self._criterion = self._create_criterion(model)

        if model_checkpoint is not None:
            load_checkpoint(model, model_checkpoint, strict=False)
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            print('DataParallel - GPU count:', torch.cuda.device_count())
            model = nn.DataParallel(model)

        self._model = model.to(device)
        no_weight_decay_parameters = getattr(self._criterion, 'no_weight_decay_parameters', None)
        if no_weight_decay_parameters is None:
            no_weight_decay_parameters = {}
        else:
            no_weight_decay_parameters = no_weight_decay_parameters()
        no_weight_decay_parameters = [parameter for name, parameter in model.named_parameters()
                                      if name.endswith('.bias') or name in no_weight_decay_parameters]
        other_parameters = [parameter for name, parameter in model.named_parameters()
                            if not name.endswith('.bias') and name not in no_weight_decay_parameters]
        parameter_groups = [
            {'params': other_parameters},
            {'params': no_weight_decay_parameters, 'weight_decay': 0.0}
        ]
        self._optimizer = torch.optim.AdamW(parameter_groups, lr=learning_rate, weight_decay=weight_decay)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, epoch_count)

        self._training_dataset_loader = self._create_training_dataset_loader(dataset_root,
                                                                             batch_size,
                                                                             batch_size_division)
        self._validation_dataset_loader = self._create_validation_dataset_loader(dataset_root,
                                                                                 batch_size,
                                                                                 batch_size_division)
        self._testing_dataset_loader = self._create_testing_dataset_loader(dataset_root,
                                                                           batch_size,
                                                                           batch_size_division)
        if self._testing_dataset_loader is None:
            self._testing_dataset_loader = self._validation_dataset_loader

    def _create_criterion(self, model):
        raise NotImplementedError()

    def _create_training_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        raise NotImplementedError()

    def _create_validation_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        raise NotImplementedError()

    def _create_testing_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        return None

    def train(self):
        self._clear_between_training()

        for epoch in range(self._epoch_count):
            print('Training - Epoch [{}/{}]'.format(epoch + 1, self._epoch_count), flush=True)
            self._train_one_epoch()

            print('\nValidation - Epoch [{}/{}]'.format(epoch + 1, self._epoch_count), flush=True)
            self._validate()
            self._scheduler.step()
            next_epoch_method = getattr(self._criterion, 'next_epoch', None)
            if next_epoch_method is not None:
                next_epoch_method()

            self._print_performances()
            self._save_learning_curves()
            self._save_states(epoch + 1)


        with torch.no_grad():
            self._model.eval()
            self._evaluate(self._model, self._device, self._testing_dataset_loader, self._output_path)

    def _clear_between_training(self):
        raise NotImplementedError()

    def _train_one_epoch(self):
        self._clear_between_training_epoch()

        self._model.train()
        self._optimizer.zero_grad()

        division = 0
        for data in tqdm(self._training_dataset_loader):
            model_output = self._model(data[0].to(self._device))
            target = self._move_target_to_device(data[1], self._device)
            loss = self._criterion(model_output, target)

            if torch.all(torch.isfinite(loss)):
                loss.backward()
                self._measure_training_metrics(loss, model_output, target)
            else:
                print('Warning the loss is not finite.')

            division += 1
            if division == self._batch_size_division:
                division = 0
                torch.nn.utils.clip_grad_value_(self._model.parameters(), 1)
                self._optimizer.step()
                self._optimizer.zero_grad()

        if division != 0:
            torch.nn.utils.clip_grad_value_(self._model.parameters(), 1)
            self._optimizer.step()
            self._optimizer.zero_grad()

    def _clear_between_training_epoch(self):
        raise NotImplementedError()

    def _move_target_to_device(self, target, device):
        raise NotImplementedError()

    def _measure_training_metrics(self, loss, model_output, target):
        raise NotImplementedError()

    def _validate(self):
        with torch.no_grad():
            self._clear_between_validation_epoch()

            self._model.eval()

            for data in tqdm(self._validation_dataset_loader):
                model_output = self._model(data[0].to(self._device))
                target = self._move_target_to_device(data[1], self._device)
                loss = self._criterion(model_output, target)

                self._measure_validation_metrics(loss, model_output, target)

    def _clear_between_validation_epoch(self):
        raise NotImplementedError()

    def _measure_validation_metrics(self, loss, model_output, target):
        raise NotImplementedError()

    def _print_performances(self):
        raise NotImplementedError()

    def _save_learning_curves(self):
        raise NotImplementedError()

    def _save_states(self, epoch):
        torch.save(self._model.state_dict(),
                   os.path.join(self._output_path, 'model_checkpoint_epoch_{}.pth'.format(epoch)))

    def _evaluate(self, model, device, dataset_loader, output_path):
        raise NotImplementedError()
