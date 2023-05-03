import os

import torch

from common.trainers import DistillationTrainer
from common.metrics import LossMetric

from pose_estimation.criterions import PoseEstimationDistillationLoss
from pose_estimation.metrics import PoseAccuracyMetric, PoseMapMetric, PoseLearningCurves, CocoPoseEvaluation

from pose_estimation.trainers.pose_estimator_trainer import _create_training_dataset_loader,\
    _create_validation_dataset_loader


class PoseEstimatorDistillationTrainer(DistillationTrainer):
    def __init__(self, device, student_model, teacher_model, dataset_root='', output_path='',
                 epoch_count=10, learning_rate=0.01, weight_decay=0.0, batch_size=128, batch_size_division=4,
                 heatmap_sigma=10,
                 student_model_checkpoint=None, teacher_model_checkpoint=None, loss_alpha=0.25):
        self._heatmap_sigma = heatmap_sigma
        self._loss_alpha = loss_alpha

        super(PoseEstimatorDistillationTrainer, self).__init__(device, student_model, teacher_model,
                                                               dataset_root=dataset_root,
                                                               output_path=output_path,
                                                               epoch_count=epoch_count,
                                                               learning_rate=learning_rate,
                                                               weight_decay=weight_decay,
                                                               batch_size=batch_size,
                                                               batch_size_division=batch_size_division,
                                                               student_model_checkpoint=student_model_checkpoint,
                                                               teacher_model_checkpoint=teacher_model_checkpoint)

        self._training_loss_metric = LossMetric()
        self._training_accuracy_metric = PoseAccuracyMetric()
        self._training_map_metric = PoseMapMetric()
        self._validation_loss_metric = LossMetric()
        self._validation_accuracy_metric = PoseAccuracyMetric()
        self._validation_map_metric = PoseMapMetric()
        self._learning_curves = PoseLearningCurves()

    def _create_criterion(self, student_model, teacher_model):
        return PoseEstimationDistillationLoss(alpha=self._loss_alpha)

    def _create_training_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        return _create_training_dataset_loader(dataset_root, batch_size, batch_size_division, self._heatmap_sigma)

    def _create_validation_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        return _create_validation_dataset_loader(dataset_root, batch_size, batch_size_division, self._heatmap_sigma)

    def _clear_between_training(self):
        self._learning_curves.clear()

    def _clear_between_training_epoch(self):
        self._training_loss_metric.clear()
        self._training_accuracy_metric.clear()
        self._training_map_metric.clear()

    def _move_target_to_device(self, target, device):
        return target[0].to(device), target[1].to(device), target[2].to(device)

    def _measure_training_metrics(self, loss, heatmap_prediction, target):
        heatmap_target, presence_target, oks_scale = target

        self._training_loss_metric.add(loss.item())
        self._training_accuracy_metric.add(heatmap_prediction, heatmap_target, presence_target, oks_scale)
        self._training_map_metric.add(heatmap_prediction, heatmap_target, presence_target, oks_scale)

    def _clear_between_validation_epoch(self):
        self._validation_loss_metric.clear()
        self._validation_accuracy_metric.clear()
        self._validation_map_metric.clear()

    def _measure_validation_metrics(self, loss, heatmap_prediction, target):
        heatmap_target, presence_target, oks_scale = target

        self._validation_loss_metric.add(loss.item())
        self._validation_accuracy_metric.add(heatmap_prediction, heatmap_target, presence_target, oks_scale)
        self._validation_map_metric.add(heatmap_prediction, heatmap_target, presence_target, oks_scale)

    def _print_performances(self):
        print('\nTraining : Loss={}, Accuracy={}, Precision={}, Recall={}, mAP={}'.format(
            self._training_loss_metric.get_loss(),
            self._training_accuracy_metric.get_accuracy(),
            self._training_accuracy_metric.get_precision(),
            self._training_accuracy_metric.get_recall(),
            self._training_map_metric.get_map()))
        print('Validation : Loss={}, Accuracy={}, Precision={}, Recall={}, mAP={}\n'.format(
            self._validation_loss_metric.get_loss(),
            self._validation_accuracy_metric.get_accuracy(),
            self._validation_accuracy_metric.get_precision(),
            self._validation_accuracy_metric.get_recall(),
            self._validation_map_metric.get_map()))

    def _save_learning_curves(self):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_training_accuracy_value(self._training_accuracy_metric.get_accuracy())
        self._learning_curves.add_training_map_value(self._training_map_metric.get_map())

        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())
        self._learning_curves.add_validation_accuracy_value(self._validation_accuracy_metric.get_accuracy())
        self._learning_curves.add_validation_map_value(self._validation_map_metric.get_map())

        self._learning_curves.save(os.path.join(self._output_path, 'learning_curves.png'),
                                   os.path.join(self._output_path, 'learning_curves.json'))

    def _evaluate(self, model, device, dataset_loader, output_path):
        print('Evaluation', flush=True)
        coco_pose_evaluation = CocoPoseEvaluation(model, device, dataset_loader, output_path)
        coco_pose_evaluation.evaluate()
