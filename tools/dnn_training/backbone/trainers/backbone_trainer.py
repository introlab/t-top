import os

import torch
import torch.nn as nn
import torch.utils.data

import torchvision.transforms as transforms

from tqdm import tqdm

from common.criterions import OhemCrossEntropyLoss, SoftmaxFocalLoss
from common.datasets import RandomSharpnessChange, RandomAutocontrast, RandomEqualize, RandomPosterize
from common.trainers import Trainer
from common.metrics import ClassificationAccuracyMetric, LossMetric, LossAccuracyLearningCurves, \
    TopNClassificationAccuracyMetric

from backbone.datasets import ClassificationOpenImages, ClassificationImageNet, MixupClassificationDataset


IMAGE_SIZE = (224, 224)


class BackboneTrainer(Trainer):
    def __init__(self, device, model, dataset_type='image_net', dataset_root='', output_path='',
                 epoch_count=10, learning_rate=0.01, weight_decay=0.0,
                 batch_size=128, criterion_type='cross_entropy_loss',
                 model_checkpoint=None):
        self._dataset_type = dataset_type
        self._criterion_type = criterion_type
        super(BackboneTrainer, self).__init__(device, model,
                                              dataset_root=dataset_root,
                                              output_path=output_path,
                                              epoch_count=epoch_count,
                                              learning_rate=learning_rate,
                                              weight_decay=weight_decay,
                                              batch_size=batch_size,
                                              batch_size_division=1,
                                              model_checkpoint=model_checkpoint)

        self._dataset_root = dataset_root

        self._learning_curves = LossAccuracyLearningCurves()

        self._training_loss_metric = LossMetric()
        self._validation_loss_metric = LossMetric()
        self._training_accuracy_metric = ClassificationAccuracyMetric()
        self._validation_accuracy_metric = ClassificationAccuracyMetric()

    def _create_criterion(self, model):
        if self._criterion_type == 'cross_entropy_loss':
            return nn.CrossEntropyLoss()
        elif self._criterion_type == 'ohem_cross_entropy_loss':
            return OhemCrossEntropyLoss()
        elif self._criterion_type == 'softmax_focal_loss':
            return SoftmaxFocalLoss()
        else:
            raise ValueError('Invalid criterion type')

    def _create_training_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transform = create_training_image_transform()
        return self._create_dataset_loader(dataset_root, batch_size, batch_size_division, 'training', transform,
                                           shuffle=True)

    def _create_validation_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transform = create_validation_image_transform()
        return self._create_dataset_loader(dataset_root, batch_size, batch_size_division, 'validation', transform,
                                           shuffle=False)

    def _create_testing_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transform = create_validation_image_transform()
        return self._create_dataset_loader(dataset_root, batch_size, batch_size_division, 'testing', transform,
                                           shuffle=False)

    def _create_dataset_loader(self, dataset_root, batch_size, batch_size_division, split, transform, shuffle):
        if self._dataset_type == 'image_net':
            split_mapping = {'training': True, 'validation': False, 'testing': False}
            dataset = ClassificationImageNet(dataset_root, train=split_mapping[split], transform=transform)
        elif self._dataset_type == 'open_images':
            dataset = ClassificationOpenImages(dataset_root, split=split, image_transform=transform)
        else:
            raise ValueError('Invalid dataset type')

        return torch.utils.data.DataLoader(MixupClassificationDataset(dataset),
                                           batch_size=batch_size // batch_size_division,
                                           shuffle=shuffle,
                                           num_workers=4)

    def _clear_between_training(self):
        self._learning_curves.clear()

    def _clear_between_training_epoch(self):
        self._training_loss_metric.clear()
        self._training_accuracy_metric.clear()

    def _move_target_to_device(self, target, device):
        return target.to(device)

    def _measure_training_metrics(self, loss, model_output, target):
        self._training_loss_metric.add(loss.item())
        self._training_accuracy_metric.add(model_output, target)

    def _clear_between_validation_epoch(self):
        self._validation_loss_metric.clear()
        self._validation_accuracy_metric.clear()

    def _measure_validation_metrics(self, loss, model_output, target):
        self._validation_loss_metric.add(loss.item())
        self._validation_accuracy_metric.add(model_output, target)

    def _print_performances(self):
        print('\nTraining : Loss={}, Accuracy={}'.format(self._training_loss_metric.get_loss(),
                                                         self._training_accuracy_metric.get_accuracy()))
        print('Validation : Loss={}, Accuracy={}\n'.format(self._validation_loss_metric.get_loss(),
                                                           self._validation_accuracy_metric.get_accuracy()))

    def _save_learning_curves(self):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())
        self._learning_curves.add_training_accuracy_value(self._training_accuracy_metric.get_accuracy())
        self._learning_curves.add_validation_accuracy_value(self._validation_accuracy_metric.get_accuracy())

        self._learning_curves.save(os.path.join(self._output_path, 'learning_curves.png'),
                                   os.path.join(self._output_path, 'learning_curves.json'))

    def _evaluate(self, model, device, dataset_loader, output_path):
        print('Evaluation - Classification', flush=True)
        top1_accuracy_metric = ClassificationAccuracyMetric()
        top5_accuracy_metric = TopNClassificationAccuracyMetric(5)

        for data in tqdm(dataset_loader):
            model_output = model(data[0].to(device))
            target = self._move_target_to_device(data[1], device)
            top1_accuracy_metric.add(model_output, target)
            top5_accuracy_metric.add(model_output, target)

        print('\nTest : Top 1 Accuracy={}, Top 5 Accuracy={}'.format(top1_accuracy_metric.get_accuracy(),
                                                                     top5_accuracy_metric.get_accuracy()))


def create_training_image_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, hue=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        RandomSharpnessChange(),
        RandomAutocontrast(),
        RandomEqualize(),
        RandomPosterize(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing()
    ])


def create_validation_image_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
