import torch
import torch.nn as nn
import torch.nn.functional as F
import jactorch
import numpy as np


class VectorQuantizer(nn.Module):
    def __init__(self, nr_embeddings, embedding_dim, beta, norm=None):
        super().__init__()
        self.nr_embeddings = nr_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.norm = norm

        self.embedding = nn.Embedding(self.nr_embeddings, self.embedding_dim)
        self.reset_parameters()
        self.saved_tensors = list()

    def reset_parameters(self):
        # self.embedding.weight.data.uniform_(-1.0 / self.nr_embeddings, 1.0 / self.nr_embeddings)
        self.embedding.weight.data.uniform_(-1.0 / 10, 1.0 / 10)

    def save_tensor(self, tensor):
        self.saved_tensors.append(tensor)

    def init_embeddings(self):
        saved = list()
        for tensor in self.saved_tensors:
            saved.append(tensor.reshape(-1, tensor.shape[-1]))
        saved = torch.cat(saved, dim=0)
        from sklearn.cluster import KMeans
        kmeans = KMeans(self.nr_embeddings)
        kmeans.fit(jactorch.as_numpy(saved))
        self.embedding.weight.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        self.saved_tensors = list()

    def normalize(self, x):
        if self.norm is None:
            pass
        elif self.norm == 'l2':
            x = jactorch.normalize(x, p=2)
        else:
            raise ValueError('Unknown normalization: {}.'.format(self.norm))
        return x

    @property
    def embedding_weight(self):
        return self.normalize(self.embedding.weight)

    def forward(self, z):
        embedding = self.embedding_weight
        z = self.normalize(z)

        z_shape = z.shape
        z = z.reshape(-1, z_shape[-1])

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        distance = (
            torch.sum(z ** 2, dim=-1, keepdim=True)
            + torch.sum(embedding ** 2, dim=1)
            - 2 * torch.matmul(z, embedding.t())
        )

        argmin = distance.argmin(dim=1)
        z_q = embedding[argmin]

        loss = None
        if self.training:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        z_q = z_q.reshape(z_shape)
        argmin = argmin.reshape(z_shape[:-1])

        if self.training:
            return z_q, argmin, loss
        return z_q, argmin

        # e_mean = torch.mean(min_encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

    def extra_repr(self) -> str:
        return f'nr_embeddings={self.nr_embeddings}, embedding_dim={self.embedding_dim}'