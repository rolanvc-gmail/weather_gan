import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from utils import get_conv_layer


class LBlock(nn.Module):
    """
    LBlock
    """
    def __init__(self, in_channels: int = 12, out_channels: int = 12, conv_type: str = "standard", kernel_size: int=3 ):
        """
        L-Block
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param conv_type: "standard", "coord", or "3d"
        :param kernel_size:
        """
        super(LBlock, self).__init__()
        self.relu = nn.ReLU()
        self.kernel_size = kernel_size
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.conv_type = conv_type
        convxd = get_conv_layer(conv_type)
        self.conv1x1 = convxd(in_channels=in_channels, out_channels=out_channels-in_channels, kernel_size=1)
        self.first_3x3_conv = convxd(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.second_3x3_conv = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1)

    def forward(self, x):
        x1 = x

        x2 = self.conv1x1()
        x12 = torch.cat([x1, x2], dim=1)

        x3 = self.relu(x)
        x3 = self.first_3x3_conv(x3)
        x3 = self.relu(x3)
        x3 = self.second_3x3_conv(x3)

        out = x12 + x3
        return out


def test_l_block():
    pass


def main():
    test_l_block()


if __name__ == "__main__":
    main()

