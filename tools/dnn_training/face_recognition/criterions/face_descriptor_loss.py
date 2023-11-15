import torch.nn as nn
import torch.nn.functional as F

from common.criterions import AmSoftmaxLoss, ArcFaceLoss


class FaceDescriptorCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, model_output, target):
        return super(FaceDescriptorCrossEntropyLoss, self).forward(model_output[1], target)


class FaceDescriptorAmSoftmaxLoss(AmSoftmaxLoss):
    def forward(self, model_output, target):
        return super(FaceDescriptorAmSoftmaxLoss, self).forward(model_output[1], target)


class FaceDescriptorArcFaceLoss(ArcFaceLoss):
    def forward(self, model_output, target):
        return super(FaceDescriptorArcFaceLoss, self).forward(model_output[1], target)


class FaceDescriptorDistillationLoss(nn.Module):
    def __init__(self, target_criterion, alpha=0.25):
        super(FaceDescriptorDistillationLoss, self).__init__()

        self._target_criterion = target_criterion
        self._alpha = alpha

    def forward(self, student_model_output, target, teacher_model_output):
        student_embedding = student_model_output[0] if isinstance(student_model_output, tuple) else student_model_output
        teacher_embedding = teacher_model_output[0] if isinstance(teacher_model_output, tuple) else teacher_model_output

        target_loss = self._target_criterion(student_model_output, target)
        teacher_loss = F.mse_loss(student_embedding, teacher_embedding)
        return self._alpha * target_loss + (1 - self._alpha) * teacher_loss



