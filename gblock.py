import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from utils import get_conv_layer


class GBlock(nn.Module):
    """
    GBlock
    """
    def __init__(self, in_channels: int = 12, out_channels: int = 12, conv_type="standard", spectral_normalized_eps=0.0001):
        """

        :param in_channels:
        :param out_channels:
        :param conv_type:
        :param spectral_normalized_eps:
        """
        super(GBlock, self).__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        convxd = get_conv_layer(conv_type)
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_1x1 = SpectralNorm(convxd(in_channels, out_channels, kernel_size=1), eps = spectral_normalized_eps)
        self.first_3x3_conv = SpectralNorm(convxd(in_channels, out_channels, kernel_size=3, stride=1, padding=1), eps=spectral_normalized_eps)
        self.second_3x3_conv = SpectralNorm(convxd(in_channels, out_channels, kernel_size=3, stride=1, padding=1), eps=spectral_normalized_eps)

    def forward(self, x):
        x1 = self.conv_1x1(x)

        x2 = self.bn1(x)
        x2 = self.relu(x2)
        x2 = self.first_3x3_conv(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.second_3x3_conv(x2)

        out = x1 + x2
        return out


def test_g_block():
    model = GBlock()
    x = torch.rand((2, 12, 128, 128))
    out = model(x)
    y = torch.rand((2, 12, 128, 128))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert out.size() == (2, 12, 128, 128)
    assert not torch.isnan(out).any(), "Output included NaNs"  # tests if any element of out if NaN.


def main():
    test_g_block()


if __name__ == "__main__":
    main()
