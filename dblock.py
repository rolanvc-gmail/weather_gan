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

    def __init__(self, input_channels: int = 12, output_channels: int = 12, conv_type: str = "standard", first_relu: bool = True,
                 keep_same_output: bool = False):
        """

        :param input_channels: number of input channels
        :param output_channels: number of output channels
        :param conv_type: 'standard', 'coord', or '3d'
        :param first_relu:
        :param keep_same_output: False/Default if "Down".
        """
        super(DBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.first_relu = first_relu
        self.keep_same_output = keep_same_output
        self.conv_type = conv_type
        convXD = get_conv_layer(conv_type)
        if conv_type == "3d":
            self.pooling = torch.nn.AvgPool3d(kernel_size=2, stride=2)
        else:
            self.pooling = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.conv_1x1 = SpectralNorm(convXD(input_channels, output_channels, kernel_size=1))
        self.first_conv_3x3 = SpectralNorm(convXD(input_channels, input_channels, kernel_size=3, stride=1, padding=1))
        self.second_conv_3x3 = SpectralNorm(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        """

        :param x:
        :return:
        """

        """
        In Pancho's code, the DBlock is as in the diagram (without down arrows). DBlockDown is as in the diagram with 
        down-arrows as maxpool. DBlockDownFirst is DBlockDown without x2's first relu.
        
        In OpenClimateFix's code, DBlock generalizes Pancho's except: if the input and out channels are unequal, x1 is conv'ed otherwise, no conv.
        also, keep_same_output=False is the equivalent of "Down". Finally, for pooling, this uses ave.
        
        Therefore, OCF's DBlock is Pancho's DBlock if input and output channels differ.
        OCF's DBlock is Pancho's DBlockDown if keep_same_output is false.
        OCF's DBlock is Pancho's DBlockDownFirst if keep_same_output is false, input and output channels differ, and first_relu = True
        
        """
        if self.input_channels != self.output_channels:
            x1 = self.conv_1x1(x)
        else:
            x1 = self.conv_1x1(x)

        if not self.keep_same_output:
            x1 = self.pooling(x1)

        if self.first_relu:
            x = self.relu(x)
        x = self.first_conv_3x3(x)
        x = self.relu(x)
        x = self.second_conv_3x3(x)

        if not self.keep_same_output:
            x = self.pooling(x)
        out = x1 + x
        return out


def test_d_block_down():
    model = DBlock(input_channels=12, output_channels=12)
    x = torch.rand((2, 12, 128, 128))
    out = model(x)
    y = torch.rand((2, 12, 64, 64))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert out.size() == (2, 12, 64, 64)
    assert not torch.isnan(out).any(), "Output included NaNs"  # tests if any element of out if NaN.


def test_d_block():
    model = DBlock(input_channels=12, output_channels=12, keep_same_output=True)
    x = torch.rand((2, 12, 128, 128))
    out = model(x)
    y = torch.rand((2, 12, 128, 128))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert out.size() == (2, 12, 128, 128)
    assert not torch.isnan(out).any(), "Output included NaNs"  # tests if any element of out if NaN.


def main():
    try:
        test_d_block_down()
        print("DBlockDown ok")
    except Exception as e:
        print("DBlockDown exception: {}".format(str(e)))

    try:
        test_d_block()
        print("DBlock ok")
    except Exception as e:
        print("DBlock exception: {}".format(str(e)))
    print("DBlock....passed")


if __name__ == "__main__":
    main()
