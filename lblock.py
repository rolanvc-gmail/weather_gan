import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from utils import get_conv_layer


class LBlock(nn.Module):
    """
    LBlock
    """
    def __init__(self, input_channels: int = 12, output_channels: int = 12, conv_type: str = "standard", kernel_size: int = 3):
        """
        L-Block
        :param input_channels: number of input channels
        :param output_channels: number of output channels
        :param conv_type: "standard", "coord", or "3d"
        :param kernel_size:
        """
        super(LBlock, self).__init__()
        self.relu = nn.ReLU()
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_type = conv_type
        convxd = get_conv_layer(conv_type)
        self.conv_1x1 = convxd(in_channels=input_channels, out_channels=output_channels - input_channels, kernel_size=1)
        self.first_3x3_conv = convxd(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.second_3x3_conv = convxd(in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size, stride=1, padding=1)

    def forward(self, x) -> torch.Tensor:
        if self.input_channels < self.output_channels:
            sc = self.conv_1x1(x)
            sc = torch.cat([x, sc], dim=1)
        else:
            sc = x

        x2 = self.relu(x)
        x2 = self.first_3x3_conv(x2)
        x2 = self.relu(x2)
        x2 = self.second_3x3_conv(x2)

        out = x2 + sc 
        return out


def test_l_block():
    pass


def main():
    test_l_block()


if __name__ == "__main__":
    main()

