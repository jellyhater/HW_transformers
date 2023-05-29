import math

import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size, maxlen):
        """
        emb_size - размер эмбеддингов
        maxlen - длинна контекста
        """

        # TODO: Реализуйте конструтор
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pos_emb = torch.zeros((maxlen, emb_size))
        position = torch.arange(0, maxlen).unsqueeze(1).float()
        div_term = torch.exp(
            -torch.arange(0, emb_size, 2).float() * (math.log(10000.0) / emb_size)
        )
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        pos_emb = pos_emb.unsqueeze(0)
        self.register_buffer("pos_emb", pos_emb)

    def forward(self, token_embedding):
        """
        token_embedding - тензор матрицы эмбеддингов
        """
        # TODO: Реализуйте сложение эмбединнгов токенов с позиционными эмбеддингами
        token_embedding = token_embedding + self.pos_emb[:, : token_embedding.size(1)]
        return token_embedding
