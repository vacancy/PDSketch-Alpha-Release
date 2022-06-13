import torch.nn as nn


class BatchMaskedMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, input, target, mask):
        mask = mask.view(-1)
        l = self.mse(input, target)
        l = l.reshape((mask.size(0), -1)).sum(-1)
        l = (l * mask).sum() / mask.sum().clamp(min=1e-5)
        return l


class BatchMaskedBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, input, target, mask):
        mask = mask.view(-1)
        l = self.bce(input, target)
        l = l.reshape((mask.size(0), -1)).mean(-1)
        l = (l * mask).sum() / mask.sum().clamp(min=1e-5)
        return l


def batch_masked_value(tensor, mask):
    mask = mask.view(-1)
    tensor = tensor.reshape(mask.size(0), -1).mean(dim=-1)
    value = (tensor * mask).sum() / mask.sum().clamp(min=1e-5)
    return value
