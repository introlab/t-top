from common.criterions import AmSoftmaxLoss


class FaceDescriptorAmSoftmaxLoss(AmSoftmaxLoss):
    def forward(self, model_output, target):
        return super(FaceDescriptorAmSoftmaxLoss, self).forward(model_output[1], target)
