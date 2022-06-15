import torch
from torch import nn
from utils import get_conv_layer
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm

class UpsampleGBlock(nn.Module):
    """

    """
    def __init__(self, input_channels: int = 12, output_channels: int = 12, conv_type: str = "standard", spectral_normalized_eps: float = 0.0001):
        """

        :param input_channels:
        :param output_channels:
        :param conv_type:
        :param spectral_normalized_eps:
        """
        super().__init__()
        self.output_channels = output_channels
        self.bn1 = torch.nn.BatchNorm2d(input_channels)
        self.bn2 = torch.nn.BatchNorm2d(input_channels)
        self.relu = torch.nn.ReLU()

        convXD = get_conv_layer(conv_type)
        self.conv_1x1 = SpectralNorm(convXD(input_channels=input_channels,
                                            output_channels=output_channels,
                                            kernel_size=1
                                            )
                                     , eps=spectral_normalized_eps
                                     )
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.first_conv_3x3 = SpectralNorm(convXD(input_channels=input_channels,
                                                  output_channels=output_channels,
                                                  kernel_size=3,
                                                  padding=1),
                                           eps=spectral_normalized_eps)
        self.last_conv_3x3 = SpectralNorm(convXD(input_channels=input_channels,
                                                 output_channels=output_channels,
                                                 kernel_size=3,
                                                 padding=1),
                                          eps=spectral_normalized_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sc = self.upsample(x)
        sc = self.conv_1x1(sc)

        x2 = self.bn1(x)
        x2 = self.relu(x2)

        x2 = self.upsample(x2)
        x2 = self.first_conv_3x3(x2)
        x2 = self.bn2(x2)
        x2 = self.last_conv_3x3(x2)

        x = x2 + sc
        return x








