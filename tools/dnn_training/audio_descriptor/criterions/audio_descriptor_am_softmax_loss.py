from common.criterions import AmSoftmaxLoss


class AudioDescriptorAmSoftmaxLoss(AmSoftmaxLoss):
    def forward(self, model_output, target):
        return super(AudioDescriptorAmSoftmaxLoss, self).forward(model_output[1], target)
