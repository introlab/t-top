import os

import torch
import torch.nn as nn

from object_detection.datasets import ObjectDetectionCoco
from object_detection.datasets import CocoDetectionTrainingTransforms, CocoDetectionValidationTransforms
from object_detection.datasets import ObjectDetectionOpenImages
from object_detection.datasets import OpenImagesDetectionTrainingTransforms, OpenImagesDetectionValidationTransforms
from object_detection.datasets.object_detection_open_images import HUMAN_BODY_PART_CLASS_NAMES

from object_detection.datasets.yolo_collate import yolo_collate

from object_detection.criterions import YoloV4Loss

from common.metrics import LossMetric
from object_detection.metrics import YoloAccuracyMetric, YoloLearningCurves, CocoObjectEvaluation, \
    YoloObjectDetectionEvaluation

from common.trainers import Trainer


class DescriptorYoloV4Trainer(Trainer):
    def __init__(self, device, model, dataset_root='', dataset_type='', class_criterion_type='', output_path='',
                 epoch_count=10, learning_rate=0.01, batch_size=128, batch_size_division=4,
                 model_checkpoint=None):
        self._dataset_type = dataset_type
        self._class_criterion_type = class_criterion_type

        super(DescriptorYoloV4Trainer, self).__init__(device, model,
                                                      dataset_root=dataset_root,
                                                      output_path=output_path,
                                                      epoch_count=epoch_count,
                                                      learning_rate=learning_rate,
                                                      batch_size=batch_size,
                                                      batch_size_division=batch_size_division,
                                                      model_checkpoint=model_checkpoint)

        self._training_loss_metric = LossMetric()
        self._training_accuracy_metric = YoloAccuracyMetric(model.get_class_count())
        self._validation_loss_metric = LossMetric()
        self._validation_accuracy_metric = YoloAccuracyMetric(model.get_class_count())
        self._learning_curves = YoloLearningCurves()

    def _create_criterion(self, model):
        model = _get_model(model)
        return YoloV4Loss(model.get_image_size(), model.get_anchors(), model.get_output_strides(),
                          model.get_class_count(), class_criterion_type=self._class_criterion_type)

    def _create_training_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        image_size = _get_model(self._model).get_image_size()
        one_hot_class = self._class_criterion_type != 'cross_entropy_loss'

        if self._dataset_type == 'coco':
            training_dataset = ObjectDetectionCoco(
                os.path.join(dataset_root, 'coco/train2017'),
                os.path.join(dataset_root, 'coco/instances_train2017.json'),
                transforms=CocoDetectionTrainingTransforms(image_size, one_hot_class))
        elif self._dataset_type == 'open_images':
            class_count = _get_model(self._model).get_class_count()
            training_dataset = ObjectDetectionOpenImages(
                dataset_root,
                split='training',
                transforms=OpenImagesDetectionTrainingTransforms(image_size, one_hot_class, class_count),
                ignored_class_names=HUMAN_BODY_PART_CLASS_NAMES)
        else:
            raise ValueError('Invalid dataset type')

        return self._create_dataset_loader(training_dataset, batch_size, batch_size_division)

    def _create_validation_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        image_size = _get_model(self._model).get_image_size()
        one_hot_class = self._class_criterion_type != 'cross_entropy_loss'

        if self._dataset_type == 'coco':
            validation_dataset = ObjectDetectionCoco(
                os.path.join(dataset_root, 'coco/val2017'),
                os.path.join(dataset_root, 'coco/instances_val2017.json'),
                transforms=CocoDetectionValidationTransforms(image_size, one_hot_class))
        elif self._dataset_type == 'open_images':
            class_count = _get_model(self._model).get_class_count()
            validation_dataset = ObjectDetectionOpenImages(
                dataset_root,
                split='validation',
                transforms=OpenImagesDetectionValidationTransforms(image_size, one_hot_class, class_count),
                ignored_class_names=HUMAN_BODY_PART_CLASS_NAMES)
        else:
            raise ValueError('Invalid dataset type')

        return self._create_dataset_loader(validation_dataset, batch_size, batch_size_division)

    def _create_testing_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        if self._dataset_type == 'coco':
            return super(DescriptorYoloV4Trainer, self)._create_testing_dataset_loader(dataset_root,
                                                                                       batch_size,
                                                                                       batch_size_division)
        elif self._dataset_type == 'open_images':
            image_size = _get_model(self._model).get_image_size()
            one_hot_class = self._class_criterion_type != 'cross_entropy_loss'
            class_count = _get_model(self._model).get_class_count()

            validation_dataset = ObjectDetectionOpenImages(
                dataset_root,
                split='testing',
                transforms=OpenImagesDetectionValidationTransforms(image_size, one_hot_class, class_count),
                ignored_class_names=HUMAN_BODY_PART_CLASS_NAMES)
            return self._create_dataset_loader(validation_dataset, batch_size, batch_size_division)
        else:
            raise ValueError('Invalid dataset type')

    def _create_dataset_loader(self, dataset, batch_size, batch_size_division):
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size // batch_size_division,
                                           collate_fn=yolo_collate,
                                           shuffle=True,
                                           num_workers=4)

    def _clear_between_training(self):
        self._learning_curves.clear()

    def _clear_between_training_epoch(self):
        self._training_loss_metric.clear()
        self._training_accuracy_metric.clear()

    def _move_target_to_device(self, target, device):
        moved_target = []
        for t in target:
            moved_target.append({'bbox': t['bbox'].to(device), 'class': t['class'].to(device)})

        return moved_target

    def _measure_training_metrics(self, loss, predictions, target):
        self._training_loss_metric.add(loss.item())
        self._training_accuracy_metric.add(predictions, target)

    def _clear_between_validation_epoch(self):
        self._validation_loss_metric.clear()
        self._validation_accuracy_metric.clear()

    def _measure_validation_metrics(self, loss, predictions, target):
        self._validation_loss_metric.add(loss.item())
        self._validation_accuracy_metric.add(predictions, target)

    def _print_performances(self):
        print('\nTraining : Loss={}, Accuracy (bbox)={}, Accuracy (class)={}'.format(
            self._training_loss_metric.get_loss(),
            self._training_accuracy_metric.get_bbox_accuracy(),
            self._training_accuracy_metric.get_class_accuracy()))
        print('Validation : Loss={}, Accuracy (bbox)={}, Accuracy (class)={}\n'.format(
            self._validation_loss_metric.get_loss(),
            self._validation_accuracy_metric.get_bbox_accuracy(),
            self._validation_accuracy_metric.get_class_accuracy()))

    def _save_learning_curves(self):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_training_bbox_accuracy_value(self._training_accuracy_metric.get_bbox_accuracy())
        self._learning_curves.add_training_class_accuracy_value(self._training_accuracy_metric.get_class_accuracy())
        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())
        self._learning_curves.add_validation_bbox_accuracy_value(self._validation_accuracy_metric.get_bbox_accuracy())
        self._learning_curves.add_validation_class_accuracy_value(self._validation_accuracy_metric.get_class_accuracy())

        self._learning_curves.save(os.path.join(self._output_path, 'learning_curves.png'),
                                   os.path.join(self._output_path, 'learning_curves.json'))

    def _evaluate(self, model, device, dataset_loader, output_path):
        print('Evaluation', flush=True)

        evaluation = YoloObjectDetectionEvaluation(model, device, dataset_loader, _get_model(model).get_class_count())
        evaluation.evaluate()

        if self._dataset_type == 'coco':
            print('Evaluation (Coco)', flush=True)
            coco_evaluation = CocoObjectEvaluation(_get_model(model), device, dataset_loader, output_path)
            coco_evaluation.evaluate()



def _get_model(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model
