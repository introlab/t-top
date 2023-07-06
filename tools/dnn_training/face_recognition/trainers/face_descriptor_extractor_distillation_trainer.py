import os

from common.datasets import TripletLossBatchSampler
from common.trainers import DistillationTrainer
from common.metrics import LossMetric, ClassificationAccuracyMetric, LossLearningCurves, LossAccuracyLearningCurves

from face_recognition.criterions import FaceDescriptorDistillationLoss
from face_recognition.datasets import ImbalancedFaceDatasetSampler
from face_recognition.metrics import LfwEvaluation
from face_recognition.trainers.face_descriptor_extractor_trainer import _create_criterion, _create_dataset, \
    _evaluate_classification_accuracy, create_training_image_transform, create_validation_image_transform

import torch
import torch.utils.data


class FaceDescriptorExtractorDistillationTrainer(DistillationTrainer):
    def __init__(self, device, model, teacher_model, dataset_roots='', lfw_dataset_root='', output_path='',
                 epoch_count=10, learning_rate=0.01, weight_decay=0.0, criterion_type='triplet_loss',
                 batch_size=128, margin=0.2,
                 student_model_checkpoint=None, teacher_model_checkpoint=None):
        self._lfw_dataset_root = lfw_dataset_root
        self._criterion_type = criterion_type
        self._margin = margin
        self._class_count = model.class_count()

        super(FaceDescriptorExtractorDistillationTrainer, self).__init__(
            device, model, teacher_model,
            dataset_root=dataset_roots,
            output_path=output_path,
            epoch_count=epoch_count,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            batch_size_division=1,
            student_model_checkpoint=student_model_checkpoint,
            teacher_model_checkpoint=teacher_model_checkpoint)

        self._training_loss_metric = LossMetric()
        self._validation_loss_metric = LossMetric()

        if self._criterion_type == 'triplet_loss':
            self._learning_curves = LossLearningCurves()
        else:
            self._learning_curves = LossAccuracyLearningCurves()
            self._training_accuracy_metric = ClassificationAccuracyMetric()
            self._validation_accuracy_metric = ClassificationAccuracyMetric()

    def _create_criterion(self, student_model, teacher_model):
        return FaceDescriptorDistillationLoss(_create_criterion(self._criterion_type, self._margin, self._epoch_count))

    def _create_training_dataset_loader(self, dataset_roots, batch_size, batch_size_division):
        dataset = _create_dataset(dataset_roots, 'training', create_training_image_transform())
        return self._create_dataset_loader(dataset, batch_size, batch_size_division,
                                           use_imbalanced_face_dataset_sampler=True)

    def _create_validation_dataset_loader(self, dataset_roots, batch_size, batch_size_division):
        dataset = _create_dataset(dataset_roots, 'validation', create_validation_image_transform())
        return self._create_dataset_loader(dataset, batch_size, batch_size_division)

    def _create_dataset_loader(self, dataset, batch_size, batch_size_division,
                               use_imbalanced_face_dataset_sampler=False):
        if self._criterion_type == 'triplet_loss':
            batch_sampler = TripletLossBatchSampler(dataset, batch_size=batch_size // batch_size_division)
            return torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=8)
        else:
            sampler = ImbalancedFaceDatasetSampler(dataset) if use_imbalanced_face_dataset_sampler else None
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size // batch_size_division, sampler=sampler,
                                               num_workers=8)

    def _clear_between_training(self):
        self._learning_curves.clear()

    def _clear_between_training_epoch(self):
        self._training_loss_metric.clear()
        if self._criterion_type != 'triplet_loss':
            self._training_accuracy_metric.clear()

    def _move_target_to_device(self, target, device):
        return target.to(device)

    def _measure_training_metrics(self, loss, model_output, target):
        self._training_loss_metric.add(loss.item())
        if self._criterion_type != 'triplet_loss':
            self._training_accuracy_metric.add(model_output[1], target)

    def _clear_between_validation_epoch(self):
        self._validation_loss_metric.clear()
        if self._criterion_type != 'triplet_loss':
            self._validation_accuracy_metric.clear()

    def _measure_validation_metrics(self, loss, model_output, target):
        self._validation_loss_metric.add(loss.item())
        if self._criterion_type != 'triplet_loss':
            self._validation_accuracy_metric.add(model_output[1], target)

    def _print_performances(self):
        if self._criterion_type != 'triplet_loss':
            print('\nTraining : Loss={}, Accuracy={}'.format(self._training_loss_metric.get_loss(),
                                                             self._training_accuracy_metric.get_accuracy()))
            print('Validation : Loss={}, Accuracy={}\n'.format(self._validation_loss_metric.get_loss(),
                                                               self._validation_accuracy_metric.get_accuracy()))
        else:
            print('\nTraining : Loss={}'.format(self._training_loss_metric.get_loss()))
            print('Validation : Loss={}\n'.format(self._validation_loss_metric.get_loss()))

    def _save_learning_curves(self):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())
        if self._criterion_type != 'triplet_loss':
            self._learning_curves.add_training_accuracy_value(self._training_accuracy_metric.get_accuracy())
            self._learning_curves.add_validation_accuracy_value(self._validation_accuracy_metric.get_accuracy())

        self._learning_curves.save(os.path.join(self._output_path, 'learning_curves.png'),
                                   os.path.join(self._output_path, 'learning_curves.json'))

    def _evaluate(self, model, device, dataset_loader, output_path):
        print('Evaluation', flush=True)

        lfw_evaluation = LfwEvaluation(model, device, dataset_loader.dataset.transforms(),
                                       self._lfw_dataset_root, output_path)
        lfw_evaluation.evaluate()

        if self._criterion_type != 'triplet_loss':
            _evaluate_classification_accuracy(model, device, dataset_loader, self._class_count)
