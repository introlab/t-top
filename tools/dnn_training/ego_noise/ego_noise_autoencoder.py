import torch.nn as nn
import torch.nn.functional as F


class PcaEgoNoiseAutoEncoderModel(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(PcaEgoNoiseAutoEncoderModel, self).__init__()

        self._encoder = nn.Linear(input_size, embedding_size, bias=False)
        self._decoder = nn.Linear(embedding_size, input_size, bias=False)

    def forward(self, x):
        return F.relu(self._decoder(self._encoder(x)))


class TwoLayerNoiseAutoEncoderModel(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(TwoLayerNoiseAutoEncoderModel, self).__init__()

        self._encoder = nn.Sequential(
            nn.Linear(input_size, 2 * embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(2 * embedding_size, embedding_size)
        )
        self._decoder = nn.Sequential(
            nn.Linear(embedding_size, 2 * embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(2 * embedding_size, input_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self._decoder(self._encoder(x))
