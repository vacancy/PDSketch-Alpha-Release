import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib

__all__ = ['detach_grouplin_parameters', 'GroupLinear', 'GroupMLP']


@contextlib.contextmanager
def detach_grouplin_parameters():
    GroupLinear.DETACHED = True
    yield
    GroupLinear.DETACHED = False


class GroupLinear(nn.Module):
    DETACHED = False

    def __init__(self, in_features, out_features, bias=True, groups=1):
        super().__init__()
        self.conv1d = nn.Conv1d(in_features * groups, out_features * groups, kernel_size=(1, ), groups=groups, bias=bias)

    def reset_parameters(self):
        self.conv1d.reset_parameters()

    def forward(self, input: torch.Tensor):
        if GroupLinear.DETACHED:
            return F.conv1d(input.unsqueeze(-1), self.conv1d.weight.detach(), self.conv1d.bias.detach(), self.conv1d.stride, self.conv1d.padding, self.conv1d.dilation, self.conv1d.groups).squeeze(-1)
        else:
            return self.conv1d(input.unsqueeze(-1)).squeeze(-1)


class GroupMLP(nn.Sequential):
    def __init__(self, input_dim, output_dim, hidden_dim, groups):
        super().__init__(
            GroupLinear(input_dim, hidden_dim, groups=groups),
            nn.ReLU(),
            GroupLinear(hidden_dim, output_dim, groups=groups)
        )
