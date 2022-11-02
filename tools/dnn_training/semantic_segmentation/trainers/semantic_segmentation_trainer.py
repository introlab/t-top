import os

import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from common.criterions import OhemCrossEntropyLoss, SoftmaxFocalLoss
from common.trainers import Trainer
from common.metrics import LossMetric

from semantic_segmentation.datasets import SemanticSegmentationCoco, SemanticSegmentationKitchenOpenImages, \
    SemanticSegmentationPersonOtherOpenImages,  SemanticSegmentationTrainingTransforms, \
    SemanticSegmentationValidationTransforms
from semantic_segmentation.metrics import LossMeanIoULearningCurves, MeanIoUMetric


IMAGE_SIZE = (270, 480)


class SemanticSegmentationTrainer(Trainer):
    def __init__(self, device, model, dataset_type='coco', dataset_root='', output_path='',
                 epoch_count=10, learning_rate=0.01, weight_decay=0.0,
                 batch_size=128, criterion_type='cross_entropy_loss',
                 model_checkpoint=None):
        self._dataset_type = dataset_type
        self._class_count = model.get_class_count()
        self._criterion_type = criterion_type
        super(SemanticSegmentationTrainer, self).__init__(device, model,
                                                          dataset_root=dataset_root,
                                                          output_path=output_path,
                                                          epoch_count=epoch_count,
                                                          learning_rate=learning_rate,
                                                          weight_decay=weight_decay,
                                                          batch_size=batch_size,
                                                          batch_size_division=1,
                                                          model_checkpoint=model_checkpoint)

        self._dataset_root = dataset_root

        self._learning_curves = LossMeanIoULearningCurves()

        self._training_loss_metric = LossMetric()
        self._validation_loss_metric = LossMetric()
        self._training_mean_iou_metric = MeanIoUMetric(self._class_count)
        self._validation_mean_iou_metric = MeanIoUMetric(self._class_count)

    def _create_criterion(self, model):
        if self._criterion_type == 'cross_entropy_loss':
            criterion = nn.CrossEntropyLoss()
        elif self._criterion_type == 'ohem_cross_entropy_loss':
            criterion = OhemCrossEntropyLoss()
        elif self._criterion_type == 'softmax_focal_loss':
            criterion = SoftmaxFocalLoss()
        else:
            raise ValueError('Invalid criterion type')

        def criterion_mean(predictions, target):
            loss = 0.0
            for prediction in predictions:
                loss += criterion(prediction, target)
            return loss / len(predictions)

        return criterion_mean

    def _create_training_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transforms = SemanticSegmentationTrainingTransforms(IMAGE_SIZE)
        return self._create_dataset_loader(dataset_root, batch_size, batch_size_division, 'training', transforms,
                                           shuffle=True)

    def _create_validation_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transforms = SemanticSegmentationValidationTransforms(IMAGE_SIZE)
        return self._create_dataset_loader(dataset_root, batch_size, batch_size_division, 'validation', transforms,
                                           shuffle=False)

    def _create_testing_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transforms = SemanticSegmentationValidationTransforms(IMAGE_SIZE)
        return self._create_dataset_loader(dataset_root, batch_size, batch_size_division, 'testing', transforms,
                                           shuffle=False)

    def _create_dataset_loader(self, dataset_root, batch_size, batch_size_division, split, transforms, shuffle):
        dataset = create_dataset(self._dataset_type, dataset_root, split, transforms)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size // batch_size_division, shuffle=shuffle,
                                           num_workers=4)

    def _clear_between_training(self):
        self._learning_curves.clear()

    def _clear_between_training_epoch(self):
        self._training_loss_metric.clear()
        self._training_mean_iou_metric.clear()

    def _move_target_to_device(self, target, device):
        return target.to(device)

    def _measure_training_metrics(self, loss, model_output, target):
        self._training_loss_metric.add(loss.item())
        self._training_mean_iou_metric.add(model_output[-1], target)

    def _clear_between_validation_epoch(self):
        self._validation_loss_metric.clear()
        self._validation_mean_iou_metric.clear()

    def _measure_validation_metrics(self, loss, model_output, target):
        self._validation_loss_metric.add(loss.item())
        self._validation_mean_iou_metric.add(model_output[-1], target)

    def _print_performances(self):
        print('\nTraining : Loss={}, Mean IoU={}'.format(self._training_loss_metric.get_loss(),
                                                         self._training_mean_iou_metric.get_mean_iou()))
        print('Validation : Loss={}, Mean IoU={}\n'.format(self._validation_loss_metric.get_loss(),
                                                           self._validation_mean_iou_metric.get_mean_iou()))

    def _save_learning_curves(self):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())
        self._learning_curves.add_training_mean_iou_value(self._training_mean_iou_metric.get_mean_iou())
        self._learning_curves.add_validation_mean_iou_value(self._validation_mean_iou_metric.get_mean_iou())

        self._learning_curves.save(os.path.join(self._output_path, 'learning_curves.png'),
                                   os.path.join(self._output_path, 'learning_curves.json'))

    def _evaluate(self, model, device, dataset_loader, output_path):
        evaluate(model, device, dataset_loader, self._class_count)


def create_dataset(dataset_type, dataset_root, split, transforms):
    if dataset_type == 'coco':
        split_mapping = {'training': True, 'validation': False, 'testing': False}
        return SemanticSegmentationCoco(dataset_root, train=split_mapping[split], transforms=transforms)
    elif dataset_type == 'kitchen_open_images':
        return SemanticSegmentationKitchenOpenImages(dataset_root, split=split, transforms=transforms)
    elif dataset_type == 'person_other_open_images':
        return SemanticSegmentationPersonOtherOpenImages(dataset_root, split=split, transforms=transforms)
    else:
        raise ValueError('Invalid dataset type')


def evaluate(model, device, dataset_loader, class_count):
    print('Evaluation - Semantic Segmentation', flush=True)
    mean_iou_metric = MeanIoUMetric(class_count)

    for data in tqdm(dataset_loader):
        model_output = model(data[0].to(device))
        target = data[1].to(device)
        mean_iou_metric.add(model_output[-1], target)

    print('\nTest : Mean IoU={}'.format(mean_iou_metric.get_mean_iou()), flush=True)
    print('IoU by class:')
    for class_index, iou in mean_iou_metric.get_iou_by_class().items():
        print('{} --> {}'.format(class_index, iou))
