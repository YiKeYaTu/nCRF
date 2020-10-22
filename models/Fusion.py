import torch
import torch.nn as nn
import sys

sys.path.append('models')

from Common import Fine, Medium, Coarse


class Fusion(nn.Module):

    def __init__(self):
        super(Fusion, self).__init__()

        self.fine = Fine()
        self.medium = Medium()
        self.coarse = Coarse()

        self.conv_fusion = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        fine = self.fine(x)
        medium = self.medium(x)
        coarse = self.coarse(x)

        y = torch.cat((fine, medium, coarse), dim=1)
        y = self.conv_fusion(y)
        y = [ torch.sigmoid(y) ]

        return y