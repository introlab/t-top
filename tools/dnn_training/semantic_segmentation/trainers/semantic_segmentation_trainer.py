import os

import torch
import torch.utils.data
from tqdm import tqdm

from common.criterions import OhemCrossEntropyLoss
from common.trainers import Trainer
from common.metrics import LossMetric

from semantic_segmentation.datasets import SemanticSegmentationOpenImages, SemanticSegmentationTrainingTransforms, \
    SemanticSegmentationValidationTransforms
from semantic_segmentation.metrics import LossMeanIoULearningCurves, MeanIoUMetric


IMAGE_SIZE = (360, 640)


class SemanticSegmentationTrainer(Trainer):
    def __init__(self, device, model, dataset_root='', output_path='', epoch_count=10, learning_rate=0.01,
                 batch_size=128, model_checkpoint=None, optimizer_checkpoint=None, scheduler_checkpoint=None):
        self._class_count = model.get_class_count()
        super(SemanticSegmentationTrainer, self).__init__(device, model,
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

        self._learning_curves = LossMeanIoULearningCurves()

        self._training_loss_metric = LossMetric()
        self._validation_loss_metric = LossMetric()
        self._training_mean_iou_metric = MeanIoUMetric(self._class_count)
        self._validation_mean_iou_metric = MeanIoUMetric(self._class_count)

    def _create_criterion(self, model):
        criterion = OhemCrossEntropyLoss()

        def criterion_mean(predictions, target):
            loss = 0.0
            for prediction in predictions:
                loss += criterion(prediction, target)
            return loss / len(predictions)

        return criterion_mean

    def _create_training_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transforms = SemanticSegmentationTrainingTransforms(IMAGE_SIZE, self._class_count)
        return self._create_dataset_loader(dataset_root, batch_size, batch_size_division, 'training', transforms,
                                           shuffle=True)

    def _create_validation_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transforms = SemanticSegmentationValidationTransforms(IMAGE_SIZE, self._class_count)
        return self._create_dataset_loader(dataset_root, batch_size, batch_size_division, 'validation', transforms,
                                           shuffle=False)

    def _create_testing_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transforms = SemanticSegmentationValidationTransforms(IMAGE_SIZE, self._class_count)
        return self._create_dataset_loader(dataset_root, batch_size, batch_size_division, 'testing', transforms,
                                           shuffle=False)

    def _create_dataset_loader(self, dataset_root, batch_size, batch_size_division, split, transforms, shuffle):
        dataset = SemanticSegmentationOpenImages(dataset_root, split=split, transforms=transforms)

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

        self._learning_curves.save_figure(os.path.join(self._output_path, 'learning_curves.png'))

    def _evaluate(self, model, device, dataset_loader, output_path):
        print('Evaluation - Semantic Segmentation', flush=True)
        mean_iou_metric = MeanIoUMetric(self._class_count)

        for data in tqdm(dataset_loader):
            model_output = model(data[0].to(device))
            target = self._move_target_to_device(data[1], device)
            mean_iou_metric.add(model_output[-1], target)

        print('\nTest : Mean IoU={}'.format(mean_iou_metric.get_mean_iou()))
