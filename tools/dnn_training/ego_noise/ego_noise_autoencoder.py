import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLuLinearGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return F.relu(x)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out


relu_linear_grad = ReLuLinearGrad.apply


class PcaEgoNoiseAutoEncoderModel(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(PcaEgoNoiseAutoEncoderModel, self).__init__()

        self._encoder = nn.Linear(input_size, embedding_size, bias=False)
        self._decoder = nn.Linear(embedding_size, input_size, bias=False)

    def forward(self, x):
        return relu_linear_grad(self._decoder(self._encoder(x)))


class TwoLayerNoiseAutoEncoderModel(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(TwoLayerNoiseAutoEncoderModel, self).__init__()

        self._encoder = nn.Sequential(
            nn.Linear(input_size, (input_size + embedding_size) // 2),
            nn.ReLU(inplace=True),
            nn.Linear(2 * embedding_size, embedding_size)
        )
        self._decoder = nn.Sequential(
            nn.Linear(embedding_size, 2 * embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(2 * embedding_size, input_size)
        )

    def forward(self, x):
        return relu_linear_grad(self._decoder(self._encoder(x)))


class FourLayerNoiseAutoEncoderModel(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(FourLayerNoiseAutoEncoderModel, self).__init__()

        self._encoder = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, input_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 8, embedding_size)
        )
        self._decoder = nn.Sequential(
            nn.Linear(embedding_size, input_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 8, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size)
        )

    def forward(self, x):
        return relu_linear_grad(self._decoder(self._encoder(x)))
