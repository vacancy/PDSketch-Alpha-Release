import torch
import torch.nn as nn
import torch.nn.functional as F
import hacl.pdsketch as pds

__all__ = ['IntEmbedding', 'FloatEmbedding', 'FallThroughEmbedding', 'ConcatIntEmbedding']


class IntEmbedding(nn.Module):
    def __init__(self, embedding_dim, input_dim=3, value_range=(0, 16), attach_input=False, concat_input=False):
        super().__init__()
        if isinstance(value_range, tuple):
            assert len(value_range) == 2
        elif isinstance(value_range, int):
            value_range = (0, value_range)
        else:
            raise TypeError('Value range should be either tuple or int, got {}.'.format(type(value_range)))

        self.input_dim = input_dim
        self.value_range = value_range
        self.embedding_dim = embedding_dim + int(concat_input) * input_dim
        self.embedding = nn.Embedding((self.value_range[1] - self.value_range[0]) ** self.input_dim, embedding_dim)
        self.attach_input = attach_input
        self.concat_input = concat_input

    def forward(self, input):
        input = pds.unwrap_value(input)
        input = input - self.value_range[0]  # shift the input first.

        if self.input_dim == 1:
            index = input[..., 0]
        elif self.input_dim == 2:
            x, y = input.split(1, dim=-1)
            index = (x * 16 + y).squeeze(-1)
        elif self.input_dim == 3:
            x, y, z = input.split(1, dim=-1)
            index = ((x * 16 + y) * 16 + z).squeeze(-1)
        elif self.input_dim == 4:
            x, y, z, t = input.split(1, dim=-1)
            index = ((((x * 16) + y) * 16 + z) * 16 + t).squeeze(-1)
        else:
            index = input[..., 0]
            for i in range(1, self.input_dim):
                index = index * 16 + input[..., i]

        # TODO:: Remove this hack after the bug is fixed.
        rv = self.embedding(index.long())
        if self.attach_input:
            rv[..., :self.input_dim] = input
        if self.concat_input:
            rv = torch.cat((rv, input), dim=-1)
        return rv


class FloatEmbedding(nn.Module):
    def __init__(self, embedding_dim, input_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.mapping = nn.Linear(input_dim, embedding_dim)

    def forward(self, input):
        input = pds.unwrap_value(input)
        return self.mapping(input)


class PadEmbedding(nn.Module):
    def __init__(self, embedding_dim, input_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        assert self.input_dim < self.embedding_dim

    def forward(self, input):
        input = pds.unwrap_value(input)
        return F.pad(input, (0, self.embedding_dim - input.shape[-1]))


class FallThroughEmbedding(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = input_dim

    def forward(self, input):
        input = pds.unwrap_value(input)
        if self.input_dim is not None:
            assert input.shape[-1] == self.input_dim
        return input


class ConcatIntEmbedding(nn.Module):
    """
    The dimension mapping is a dictionary from int to int. For example, if the mapping is:

    ```
    {
        3: IntEmbedding(64),
        2: IntEmbedding(32, value_range=(-1, 15)),
        1: IntEmbedding(32, value_range=(0, 4))
    }
    ```

    This mapping indicates that the input tensor has dimension 3+2+1=6.
    The first 3 dimensions will be embedded to a 64-dim latent vector.
    The next 2 dimensions will be embedded to a 32-dim latent vector.
    The last dimension will be embedded to a 32-dim latent vector.

    Thus, the total output dimension is 64+32+32 = 128.
    """
    def __init__(self, dimension_mapping: dict[int, nn.Module]):
        super().__init__()
        self.dimension_mapping = dimension_mapping
        self.embeddings = nn.ModuleList()

        self.input_dim = 0
        self.output_dim = 0
        for k, v in dimension_mapping.items():
            self.input_dim += k
            self.output_dim += v.embedding_dim
            self.embeddings.append(v)

    def forward(self, input):
        input = pds.unwrap_value(input)

        dims = list(self.dimension_mapping)
        input_splits = torch.split(input, dims, dim=-1)
        outputs = list()
        for v, e in zip(input_splits, self.embeddings):
            outputs.append(e(v))
        return torch.cat(outputs, dim=-1)
