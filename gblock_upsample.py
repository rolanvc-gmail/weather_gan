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
        self.bn2 = torch.nn.BatchNorm2d(input_channels//2)  # originally, just input_channels
        self.relu = torch.nn.ReLU()

        convxd = get_conv_layer(conv_type)
        self.conv_1x1 = SpectralNorm(convxd(in_channels=input_channels,
                                            out_channels=output_channels,
                                            kernel_size=1
                                            )
                                     , eps=spectral_normalized_eps
                                     )
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.first_conv_3x3 = SpectralNorm(convxd(in_channels=input_channels,
                                                  out_channels=output_channels,
                                                  kernel_size=3,
                                                  padding=1),
                                           eps=spectral_normalized_eps)
        self.last_conv_3x3 = SpectralNorm(convxd(in_channels=input_channels//2,  # originally just input_channels
                                                 out_channels=output_channels,
                                                 kernel_size=3,
                                                 padding=1),
                                          eps=spectral_normalized_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is Tensor(2,768,8,8)
        sc = self.upsample(x)
        # sc is Tensor(2,768,16,16)
        sc = self.conv_1x1(sc)
        # sc is Tensor(2,384,16,16)
        
        x2 = self.bn1(x)
        # x2 is Tensor(2,768,8,8)
        x2 = self.relu(x2)
        # x2 is Tensor(2,768,8,8)
        x2 = self.upsample(x2)
        # x2 is Tensor(2,768,16,16)
        x2 = self.first_conv_3x3(x2)
        # x2 is Tensor(2, 384, 16, 16)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        # x2 is Tensor(2, 384, 16, 16)

        x2 = self.last_conv_3x3(x2)
        # x2 is Tensor(2, 768, 16, 16)
        x = x2 + sc
        return x








