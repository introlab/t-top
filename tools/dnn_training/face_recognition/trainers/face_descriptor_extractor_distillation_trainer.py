import os

import torch.nn as nn
import torchvision.transforms as transforms

from tqdm import tqdm

from common.criterions import TripletLoss, AmSoftmaxLoss, ArcFaceLoss
from common.datasets import TripletLossBatchSampler, RandomSharpnessChange, RandomAutocontrast, RandomEqualize, \
    RandomPosterize
from common.trainers import Trainer
from common.metrics import LossMetric, ClassificationAccuracyMetric, TopNClassificationAccuracyMetric, \
    ClassificationMeanAveragePrecisionMetric, LossLearningCurves, LossAccuracyLearningCurves

from face_recognition.criterions import FaceDescriptorAmSoftmaxLoss, FaceDescriptorArcFaceLoss, \
    FaceDescriptorCrossEntropyLoss
from face_recognition.datasets import IMAGE_SIZE, Vggface2Dataset, LFW_OVERLAPPED_VGGFACE2_CLASS_NAMES
from face_recognition.metrics import LfwEvaluation

import torch
import torch.utils.data


class FaceDescriptorExtractorTrainer(Trainer):
    def __init__(self, device, model, vvgface2_dataset_root='', lfw_dataset_root='', output_path='',
                 epoch_count=10, learning_rate=0.01, weight_decay=0.0, criterion_type='triplet_loss',
                 batch_size=128, margin=0.2,
                 model_checkpoint=None):
        self._lfw_dataset_root = lfw_dataset_root
        self._criterion_type = criterion_type
        self._margin = margin
        self._class_count = model.class_count()

        torch.autograd.set_detect_anomaly(True)

        super(FaceDescriptorExtractorTrainer, self).__init__(device, model,
                                                             dataset_root=vvgface2_dataset_root,
                                                             output_path=output_path,
                                                             epoch_count=epoch_count,
                                                             learning_rate=learning_rate,
                                                             weight_decay=weight_decay,
                                                             batch_size=batch_size,
                                                             batch_size_division=1,
                                                             model_checkpoint=model_checkpoint)

        self._training_loss_metric = LossMetric()
        self._validation_loss_metric = LossMetric()

        if self._criterion_type == 'triplet_loss':
            self._learning_curves = LossLearningCurves()
        else:
            self._learning_curves = LossAccuracyLearningCurves()
            self._training_accuracy_metric = ClassificationAccuracyMetric()
            self._validation_accuracy_metric = ClassificationAccuracyMetric()

    def _create_criterion(self, model):
        if self._criterion_type == 'triplet_loss':
            return TripletLoss(margin=self._margin)
        elif self._criterion_type == 'cross_entropy_loss':
            return FaceDescriptorCrossEntropyLoss()
        elif self._criterion_type == 'am_softmax_loss':
            return FaceDescriptorAmSoftmaxLoss(s=30.0, m=self._margin,
                                               start_annealing_epoch=0,
                                               end_annealing_epoch=self._epoch_count // 4)
        elif self._criterion_type == 'arc_face_loss':
            return FaceDescriptorArcFaceLoss(s=30.0, m=self._margin,
                                             start_annealing_epoch=0,
                                             end_annealing_epoch=self._epoch_count // 4)
        else:
            raise ValueError('Invalid criterion type')

    def _create_training_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        dataset = Vggface2Dataset(dataset_root, split='training',
                                  transforms=create_training_image_transform(),
                                  ignored_classes=LFW_OVERLAPPED_VGGFACE2_CLASS_NAMES)
        return self._create_dataset_loader(dataset, batch_size, batch_size_division)

    def _create_validation_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        dataset = Vggface2Dataset(dataset_root, split='validation',
                                  transforms=create_validation_image_transform(),
                                  ignored_classes=LFW_OVERLAPPED_VGGFACE2_CLASS_NAMES)
        return self._create_dataset_loader(dataset, batch_size, batch_size_division)

    def _create_dataset_loader(self, dataset, batch_size, batch_size_division):
        if self._criterion_type == 'triplet_loss':
            batch_sampler = TripletLossBatchSampler(dataset, batch_size=batch_size // batch_size_division)
            return torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=8)
        else:
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size // batch_size_division, shuffle=True,
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
            self._evaluate_classification_accuracy(model, device, dataset_loader)

    def _evaluate_classification_accuracy(self, model, device, dataset_loader):
        print('Evaluation - Classification')
        top1_accuracy_metric = ClassificationAccuracyMetric()
        top5_accuracy_metric = TopNClassificationAccuracyMetric(5)
        map_metric = ClassificationMeanAveragePrecisionMetric(self._class_count)

        for data in tqdm(dataset_loader):
            model_output = model(data[0].to(device))
            target = self._move_target_to_device(data[1], device)
            top1_accuracy_metric.add(model_output[1], target)
            top5_accuracy_metric.add(model_output[1], target)
            map_metric.add(model_output[1], target)

        print('\nTest : Top 1 Accuracy={}, Top 5 Accuracy={}, mAP={}'.format(top1_accuracy_metric.get_accuracy(),
                                                                             top5_accuracy_metric.get_accuracy(),
                                                                             map_metric.get_value()))


def create_training_image_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
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
