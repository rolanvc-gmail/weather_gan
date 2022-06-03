import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from utils import get_conv_layer


class DBlock(nn.Module):
    """
    This is a generalized D-Block, that can be a:
    D Block
    3d Block
    D Block with downsampling
    D Block with first_relu
    """

    def __init__(self, n_channels_in: int = 12, n_channels_out: int = 12, conv_type: str = "standard", first_relu: bool = True,
                 keep_same_output:bool = False):
        """

        :param n_channels_in: number of input channels
        :param n_channels_out: number of output channels
        :param conv_type: 'standard', 'coord', or '3d'
        :param first_relu:
        :param keep_same_output:
        """
        super(DBlock, self).__init__()
        self.input_channels = n_channels_in
        self.output_channels = n_channels_out
        self.first_relu = first_relu
        self.keep_same_output = keep_same_output
        self.conv_type = conv_type
        convXD = get_conv_layer(conv_type)
        if conv_type == "3d":
            self.pooling = torch.nn.AvgPool3d(kernel_size=2, stride=2)
        else:
            self.pooling = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.conv_1x1 = SpectralNorm(convXD(n_channels_in, n_channels_out, kernel_size=1))
        self.first_conv_3x3 = SpectralNorm(convXD(n_channels_in, n_channels_in, kernel_size=3, stride=1, padding=1))
        self.second_conv_3x3 = SpectralNorm(nn.Conv2d(n_channels_in, n_channels_in, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        if self.input_channels != self.output_channels:
            x1 = self.conv_1x1(x)
            if not self.keep_same_output:
                x1 = self.pooling(x1)
        else:
            x1 = x
        if self.first_relu:
            x = self.relu(x)
        x = self.first_conv_3x3(x)
        x = self.relu(x)
        x = self.second_conv_3x3(x)

        if not self.keep_same_output:
            x1 = self.pooling(x)
        out = x1 + x
        return out


def test_d_block():
    model = DBlock(n_channels_in=12, n_channels_out=12, keep_same_output=True)
    x = torch.rand((2, 12, 128, 128))
    out = model(x)
    y = torch.rand((2, 12, 128, 128))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert out.size() == (2, 12, 128, 128)
    assert not torch.isnan(out).any(), "Output included NaNs"  # tests if any element of out if NaN.


def main():
    test_d_block()


if __name__ == "__main__":
    main()
