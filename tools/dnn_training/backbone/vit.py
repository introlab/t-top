import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torch.nn.functional as F


class ImagePatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, stride, embedding_size):
        super(ImagePatchEmbedding, self).__init__()
        self._projection = nn.Conv2d(in_channels, out_channels=embedding_size, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        return self._projection(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, head_count, attention_dropout_rate=0.0, projection_dropout_rate=0.0):
        super(SelfAttention, self).__init__()

        self._head_count = head_count
        head_dim = dim // head_count
        self._scale = 1.0 / math.sqrt(head_dim)

        self._qkv = nn.Linear(in_features=dim, out_features=3 * dim)
        self._attention_dropout = nn.Dropout(attention_dropout_rate)
        self._projection = nn.Linear(in_features=dim, out_features=dim)
        self._projection_dropout = nn.Dropout(projection_dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self._qkv(x).reshape(B, N, 3, self._head_count, C // self._head_count).permute(2, 0, 3, 1, 4)

        attention = (torch.matmul(qkv[0], qkv[1].transpose(-2, -1))) * self._scale
        normalized_attention = F.softmax(attention, dim=-1)
        normalized_attention = self._attention_dropout(normalized_attention)

        y = torch.matmul(normalized_attention, qkv[2]).transpose(1, 2).reshape(B, N, C)
        return self._projection_dropout(self._projection(y))


class EncoderBlock(nn.Module):
    def __init__(self, dim, head_count, dropout_rate=0.0, attention_dropout_rate=0.0):
        super(EncoderBlock, self).__init__()
        self._attention = SelfAttention(dim,
                                        head_count,
                                        projection_dropout_rate=dropout_rate,
                                        attention_dropout_rate=attention_dropout_rate)
        self._attention_norm = nn.LayerNorm(dim)
        self._mlp = nn.Sequential(
            nn.Linear(in_features=dim, out_features=4 * dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=4 * dim, out_features=dim),
            nn.Dropout(dropout_rate)
        )
        self._mlp_norm = nn.LayerNorm(dim)

    def forward(self, x):
        y0 = x + self._attention(self._attention_norm(x))
        y1 = y0 + self._mlp(self._mlp_norm(y0))
        return y1


class Vit(nn.Module):
    def __init__(self, image_size, in_channels=3, patch_size=16, stride=16, embedding_size=768, head_count=12, depth=7,
                 class_count=1000, distilled=False, patchout_time=5, patchout_freq=5,
                 dropout_rate=0.1, attention_dropout_rate=0.1, output_embeddings=False):
        super(Vit, self).__init__()

        self._patchout_time = patchout_time
        self._patchout_freq = patchout_freq
        self._class_count = class_count
        self._output_embeddings = output_embeddings

        image_size = _pair(image_size)
        grid_size = (image_size[0] // stride, image_size[1] // stride)
        self._image_patch_embedding = ImagePatchEmbedding(in_channels, patch_size, stride, embedding_size)

        self._class_token = nn.Parameter(torch.zeros(1, 1, embedding_size))
        self._distillation_token = nn.Parameter(torch.zeros(1, 1, embedding_size)) if distilled else None

        self._class_positional_embedding = nn.Parameter(torch.zeros(1, 1, embedding_size))
        self._distillation_positional_embedding = nn.Parameter(torch.zeros(1, 1, embedding_size)) if distilled else None
        self._freq_positional_embedding = nn.Parameter(torch.zeros(1, embedding_size, grid_size[0], 1))
        self._time_positional_embedding = nn.Parameter(torch.zeros(1, embedding_size, 1, grid_size[1]))

        self._encoders = nn.Sequential(*[EncoderBlock(embedding_size, head_count, dropout_rate, attention_dropout_rate)
                                         for _ in range(depth)],
                                       nn.LayerNorm(embedding_size))
        self._head = nn.Linear(in_features=embedding_size, out_features=class_count, bias=False)

        if distilled:
            self._head_distillation = nn.Linear(in_features=embedding_size, out_features=class_count, bias=False)

        self._embedding_dropout = nn.Dropout(dropout_rate)

        self._init_embeddings()
        for name, module in self.named_modules():
            self._init_module_weights(module, name)

    def class_count(self):
        return self._class_count

    def no_weight_decay_parameters(self):
        return {'_class_token', '_distillation_token', '_class_positional_embedding',
                '_distillation_positional_embedding', '_freq_positional_embedding', '_time_positional_embedding'}

    def _init_embeddings(self):
        nn.init.trunc_normal_(self._class_token, std=0.02)
        if self._distillation_token is not None:
            nn.init.trunc_normal_(self._distillation_token, std=0.02)
            nn.init.trunc_normal_(self._distillation_positional_embedding, std=0.02)
        nn.init.trunc_normal_(self._class_positional_embedding, std=0.02)
        nn.init.trunc_normal_(self._freq_positional_embedding, std=0.02)
        nn.init.trunc_normal_(self._time_positional_embedding, std=0.02)

    def _init_module_weights(self, module, name):
        if isinstance(module, nn.Linear):
            if 'head' in name:
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            else:
                nn.init.trunc_normal_(module.weight, std=.02)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        embedding = self._forward_embeddings(x)
        all_features = self._encoders(embedding)
        class_token_embedding = F.normalize(all_features[:, 0], p=2.0, dim=1)
        y = self._head(class_token_embedding)

        if self._distillation_token is not None:
            y_distillation = self._head_distillation(F.normalize(all_features[:, 1], p=2.0, dim=1))
            if self.training:
                if self._output_embeddings:
                    return class_token_embedding, y, y_distillation
                else:
                    return y, y_distillation
            else:
                if self._output_embeddings:
                    return class_token_embedding, (x + y_distillation) / 2,
                else:
                    return (x + y_distillation) / 2
        else:
            if self._output_embeddings:
                return class_token_embedding, y
            else:
                return y

    def _forward_embeddings(self, x):
        embedding = self._image_patch_embedding(x)

        if embedding.size(-1) < self._time_positional_embedding.size(-1):
            time_positional_embedding = self._time_positional_embedding[:, :, :, :embedding.size(-1)]
        elif embedding.size(-1) > self._time_positional_embedding.size(-1):
            raise ValueError('The image is bigger than the positional embedding')
        else:
            time_positional_embedding = self._time_positional_embedding

        embedding = embedding + time_positional_embedding + self._freq_positional_embedding

        # Structured patchout
        _, _, F, T = embedding.size()
        if self.training and self._patchout_time > 0:
            random_indices = torch.randperm(T)[:T - self._patchout_time].sort().values
            embedding = embedding[:, :, :, random_indices]
        if self.training and self._patchout_freq > 0:
            random_indices = torch.randperm(F)[:F - self._patchout_freq].sort().values
            embedding = embedding[:, :, random_indices, :]

        embedding = embedding.flatten(2).transpose(1, 2)
        B, N, C = embedding.size()
        class_token = self._class_token.expand(B, -1, -1) + self._class_positional_embedding#
        if self._distillation_token is None:
            return torch.cat([class_token, embedding], dim=1)
        else:
            distillation_token = self._distillation_token.expand(B, -1, -1) + self._distillation_positional_embedding
            return torch.cat([class_token, distillation_token, embedding], dim=1)
