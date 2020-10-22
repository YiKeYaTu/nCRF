import torch
import torch.nn as nn

class N_CRF(nn.Module):
    def __init__(self):
        super(N_CRF, self).__init__()

        self.conv_CRF = nn.Conv2d(3, 256, 7, padding=3)
        self.conv_NCRF = nn.Conv2d(256, 256, 21, padding=10, bias=False)
        self.conv_MF = nn.Conv2d(256, 1, 1, padding=0)

        self.relu = torch.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x):
        conv_CRF = self.conv_CRF(x)
        conv_NCRF = self.conv_NCRF(conv_CRF)

        conv_modulatory = self.relu(conv_CRF - conv_NCRF)

        conv_fusion = self.conv_MF(conv_modulatory)

        result = self.sigmoid(conv_fusion)
        results = [result]

        return results
