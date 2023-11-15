import os

import torch
import torch.nn as nn
import torch.utils.data

from tqdm import tqdm

from common.criterions import JensenShannonDivergence
from common.modules import load_checkpoint
from common.metrics import ClassificationAccuracyMetric, LossMetric, LossAccuracyLearningCurves, \
    TopNClassificationAccuracyMetric

from backbone.datasets import ClassificationOpenImages, ClassificationImageNet, MixupClassificationDataset
from backbone.trainers.backbone_trainer import create_training_image_transform, create_validation_image_transform


IMAGE_SIZE = (224, 224)


class BackboneDistillationTrainer:
    def __init__(self, device, student_model, teacher_model, image_net_root='', open_images_root=None, output_path='',
                 epoch_count=10, learning_rate=0.01, weight_decay=0.0,
                 batch_size=128, student_model_checkpoint=None):
        self._device = device
        self._output_path = output_path
        os.makedirs(self._output_path, exist_ok=True)

        self._epoch_count = epoch_count
        self._batch_size = batch_size
        self._criterion = JensenShannonDivergence()

        if student_model_checkpoint is not None:
            load_checkpoint(student_model, student_model_checkpoint, strict=False)
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            print('DataParallel - GPU count:', torch.cuda.device_count())
            student_model = nn.DataParallel(student_model)
            teacher_model = nn.DataParallel(teacher_model)

        self._student_model = student_model.to(device)
        self._teacher_model = teacher_model.to(device)
        self._teacher_model.eval()

        self._optimizer = torch.optim.AdamW(self._student_model.parameters(), lr=learning_rate,
                                           weight_decay=weight_decay)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, epoch_count)

        self._training_dataset_loader = self._create_training_dataset_loader(image_net_root, open_images_root,
                                                                             batch_size)
        self._validation_dataset_loader = self._create_validation_dataset_loader(image_net_root, batch_size)

        self._learning_curves = LossAccuracyLearningCurves()

        self._training_loss_metric = LossMetric()
        self._validation_loss_metric = LossMetric()
        self._validation_accuracy_metric = ClassificationAccuracyMetric()

    def _create_training_dataset_loader(self, image_net_root, open_images_root, batch_size):
        transform = create_training_image_transform()
        datasets = []
        datasets.append(ClassificationImageNet(image_net_root, train=True, transform=transform))
        if open_images_root is not None:
            datasets.append(ClassificationOpenImages(open_images_root, split='training', image_transform=transform))

        dataset = torch.utils.data.ConcatDataset(datasets)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    def _create_validation_dataset_loader(self, image_net_root, batch_size):
        transform = create_validation_image_transform()
        dataset = ClassificationImageNet(image_net_root, train=False, transform=transform)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    def train(self):
        self._learning_curves.clear()

        for epoch in range(self._epoch_count):
            print('Training - Epoch [{}/{}]'.format(epoch + 1, self._epoch_count), flush=True)
            self._train_one_epoch()

            print('\nValidation - Epoch [{}/{}]'.format(epoch + 1, self._epoch_count), flush=True)
            self._validate()
            self._scheduler.step()

            self._print_performances()
            self._save_learning_curves()
            self._save_states(epoch + 1)

        with torch.no_grad():
            self._evaluate()

    def _train_one_epoch(self):
        self._training_loss_metric.clear()

        self._student_model.train()

        for data in tqdm(self._training_dataset_loader):
            model_data = data[0].to(self._device)
            with torch.no_grad():
                teacher_model_output = self._teacher_model(model_data)
            student_model_output = self._student_model(model_data)
            loss = self._criterion(student_model_output, teacher_model_output)

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self._student_model.parameters(), 1)
            self._optimizer.step()

            self._measure_training_metrics(loss)

    def _move_target_to_device(self, target, device):
        return target.to(device)

    def _measure_training_metrics(self, loss):
        self._training_loss_metric.add(loss.item())

    def _validate(self):
        with torch.no_grad():
            self._clear_between_validation_epoch()

            self._student_model.eval()

            for data in tqdm(self._validation_dataset_loader):
                model_data = data[0].to(self._device)
                teacher_model_output = self._teacher_model(model_data)
                student_model_output = self._student_model(model_data)
                loss = self._criterion(student_model_output, teacher_model_output)

                target = self._move_target_to_device(data[1], self._device)
                self._measure_validation_metrics(loss, student_model_output, target)

    def _clear_between_validation_epoch(self):
        self._validation_loss_metric.clear()
        self._validation_accuracy_metric.clear()

    def _measure_validation_metrics(self, loss, model_output, target):
        self._validation_loss_metric.add(loss.item())
        self._validation_accuracy_metric.add(model_output, target)

    def _print_performances(self):
        print('\nTraining : Loss={}'.format(self._training_loss_metric.get_loss()))
        print('Validation : Loss={}, Accuracy={}\n'.format(self._validation_loss_metric.get_loss(),
                                                           self._validation_accuracy_metric.get_accuracy()))

    def _save_learning_curves(self):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())
        self._learning_curves.add_validation_accuracy_value(self._validation_accuracy_metric.get_accuracy())

        self._learning_curves.save(os.path.join(self._output_path, 'learning_curves.png'),
                                   os.path.join(self._output_path, 'learning_curves.json'))

    def _save_states(self, epoch):
        torch.save(self._student_model.state_dict(),
                   os.path.join(self._output_path, 'model_checkpoint_epoch_{}.pth'.format(epoch)))

    def _evaluate(self):
        print('Evaluation - Classification', flush=True)
        top1_accuracy_metric = ClassificationAccuracyMetric()
        top5_accuracy_metric = TopNClassificationAccuracyMetric(5)

        for data in tqdm(self._validation_dataset_loader):
            student_model_output = self._student_model(data[0].to(self._device))
            target = self._move_target_to_device(data[1], self._device)
            top1_accuracy_metric.add(student_model_output, target)
            top5_accuracy_metric.add(student_model_output, target)

        print('\nTest : Top 1 Accuracy={}, Top 5 Accuracy={}'.format(top1_accuracy_metric.get_accuracy(),
                                                                     top5_accuracy_metric.get_accuracy()))
