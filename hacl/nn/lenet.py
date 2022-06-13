import torch.nn as nn
import torch.nn.functional as F
import jactorch.nn as jacnn

__all__ = ['LeNet', 'LeNetRGB32']


class LeNet(nn.Module):
    def __init__(self, nr_classes=10):
        super().__init__()
        self.nr_classes = nr_classes
        self.conv1 = jacnn.Conv2dLayer(1, 10, kernel_size=5, batch_norm=True, activation='relu')
        self.conv2 = jacnn.Conv2dLayer(10, 20, kernel_size=5, batch_norm=True, dropout=False, activation='relu')
        self.fc1 = nn.Linear(320, 50)
        if self.nr_classes is not None:
            self.fc2 = nn.Linear(50, 10)
        else:
            self.add_module('fc2', None)

    def forward(self, x):
        assert x.dim() in (2, 4)
        if x.dim() == 2:
            x = x.reshape(x.shape[0], 1, 28, 28)

        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        if self.fc2 is not None:
            x = self.fc2(x)
        return x


class LeNetRGB32(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        self.conv1 = jacnn.Conv2dLayer(3, 10, kernel_size=5, batch_norm=False, activation='relu')
        self.conv2 = jacnn.Conv2dLayer(10, 20, kernel_size=5, batch_norm=False, dropout=False, activation='relu')
        self.fc1 = nn.Linear(500, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        assert x.dim() in (2, 4)
        if x.dim() == 2:
            x = x.reshape(x.shape[0], 3, 32, 32)
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
