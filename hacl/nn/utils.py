import torch.nn.init as init
import torch.nn as nn


def init_weights(net, init_type='normal', init_param=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_param)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_param)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orth':
                init.orthogonal_(m.weight.data, gain=init_param)
            elif init_type == 'pdb':
                init.constant_(m.weight.data, 1.0)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_param)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class TimesConstant(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x):
        return x * self.c


class AddConstant(nn.Module):
    def __init__(self, constant=0):
        super(AddConstant, self).__init__()
        self.constant = constant

    def forward(self, x):
        return x + self.constant

class Negative(nn.Module):
    def __init__(self):
        super(Negative, self).__init__()

    def forward(self, x):
        return -x
