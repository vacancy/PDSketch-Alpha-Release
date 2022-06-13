import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GumbelQuantizer']


class GumbelQuantizer(nn.Module):
    def __init__(self, input_dim, nr_embeddings, embedding_dim, hard=True, temperature=0.9):
        super().__init__()
        self.input_dim = input_dim
        self.nr_embeddings = nr_embeddings
        self.embedding_dim = embedding_dim
        self.hard = hard
        self.temperature = temperature

        self.encoder = nn.Linear(input_dim, nr_embeddings)
        self.embedding = nn.Embedding(self.nr_embeddings, self.embedding_dim)

    def forward(self, x, temperature=None):
        if temperature is None:
            temperature = self.temperature

        logits = self.encoder(x)
        if self.hard:
            prob = F.gumbel_softmax(logits, tau=temperature, hard=self.hard, dim=-1)
        else:
            prob = F.gumbel_softmax(logits, tau=temperature, hard=not self.training, dim=-1)

        return prob @ self.embedding.weight, prob
