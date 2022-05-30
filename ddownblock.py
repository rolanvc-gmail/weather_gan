import torch
import torch.nn as nn
from spectral_norm import SpectralNorm


class DDownBlock(nn.Module):
    """
    DDownBlock
    """
    def __init__(self, n_channels_in, n_channels_out):
        super(DDownBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv2d(n_channels_in, n_channels_out, 1))
        self.conv3x3a = SpectralNorm(nn.Conv2d(n_channels_in, n_channels_in, 3, stride=1, padding=1))
        self.conv3x3b = SpectralNorm(nn.Conv2d(n_channels_in, n_channels_in, 3, stride=1, padding=1))
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool(x1)

        x2 = self.relu(x)
        x2 = self.conv3x3a(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)

        out = x1 + x2
        return out


