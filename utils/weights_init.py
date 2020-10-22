import torch
import torch.nn as nn

from os.path import join
from constant import THIS_DIR

def normal_weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()

def pretrained_weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()

        if m.weight.data.shape == torch.Size([256, 3, 7, 7]):
            parameters = torch.load(join(THIS_DIR, '../../SAE/checkpoint.pth'))
            pretrained_weights = parameters['encoder.weight']
            pretrained_bias = parameters['encoder.bias']

            m.weight.data = nn.Parameter(pretrained_weights).cuda()
            m.bias.data = nn.Parameter(pretrained_bias).cuda()