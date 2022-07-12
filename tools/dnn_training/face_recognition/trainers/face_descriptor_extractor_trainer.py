import os

import torchvision.transforms as transforms

from common.criterions import TripletLoss
from common.datasets import TripletLossBatchSampler, RandomSharpnessChange, RandomAutocontrast, RandomEqualize, \
    RandomPosterize
from common.trainers import Trainer
from common.metrics import LossMetric, LossLearningCurves

from face_recognition.datasets import IMAGE_SIZE, FaceRecognitionFolderDataset, LFW_OVERLAPPED_VGGFACE2_CLASS_NAMES
from face_recognition.metrics import LfwEvaluation

import torch
import torch.utils.data


class FaceDescriptorExtractorTrainer(Trainer):
    def __init__(self, device, model, dataset_root='', output_path='', epoch_count=10, learning_rate=0.01,
                 batch_size=128, margin=0.2,
                 model_checkpoint=None):
        self._margin = margin

        super(FaceDescriptorExtractorTrainer, self).__init__(device, model,
                                                             dataset_root=dataset_root,
                                                             output_path=output_path,
                                                             epoch_count=epoch_count,
                                                             learning_rate=learning_rate,
                                                             batch_size=batch_size,
                                                             batch_size_division=1,
                                                             model_checkpoint=model_checkpoint)

        self._dataset_root = dataset_root

        self._training_loss_metric = LossMetric()
        self._validation_loss_metric = LossMetric()
        self._learning_curves = LossLearningCurves()

    def _create_criterion(self, model):
        return TripletLoss(margin=self._margin)

    def _create_training_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        training_dataset = FaceRecognitionFolderDataset(os.path.join(dataset_root, 'aligned_vggface2', 'train'),
                                                        transforms=create_training_image_transform(),
                                                        ignored_classes=LFW_OVERLAPPED_VGGFACE2_CLASS_NAMES)

        batch_sampler = TripletLossBatchSampler(training_dataset, batch_size=batch_size // batch_size_division)
        return torch.utils.data.DataLoader(training_dataset, batch_sampler=batch_sampler, num_workers=2)

    def _create_validation_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        training_dataset = FaceRecognitionFolderDataset(os.path.join(dataset_root, 'aligned_vggface2', 'test'),
                                                        transforms=create_validation_image_transform(),
                                                        ignored_classes=LFW_OVERLAPPED_VGGFACE2_CLASS_NAMES)

        batch_sampler = TripletLossBatchSampler(training_dataset, batch_size=batch_size // batch_size_division)
        return torch.utils.data.DataLoader(training_dataset, batch_sampler=batch_sampler, num_workers=2)

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

        self._learning_curves.save(os.path.join(self._output_path, 'learning_curves.png'),
                                   os.path.join(self._output_path, 'learning_curves.json'))

    def _evaluate(self, model, device, dataset_loader, output_path):
        print('Evaluation', flush=True)

        lfw_evaluation = LfwEvaluation(model, device, dataset_loader.dataset.transforms(),
                                       os.path.join(self._dataset_root, 'aligned_lfw'), output_path)
        lfw_evaluation.evaluate()


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
