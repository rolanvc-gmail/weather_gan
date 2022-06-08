import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from utils import get_conv_layer
from torch.nn.modules.pixelshuffle import PixelUnshuffle
from dblock import DBlock
from typing import Tuple
import numpy as np


class ConditioningStack(nn.Module):
    """
    ConditioningStack
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 768, conv_type: str = "standard", **kwargs):
        super(ConditioningStack, self).__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        input_channels = self.config["input_channels"]
        output_channels = self.config["output_channels"]
        conv_type = self.config["conv_type"]
        num_context_steps = self.config["num_context_steps"]

        convxd = get_conv_layer(conv_type)
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        self.d1 = DBlock(
            input_channels=4*input_channels,
            output_channels=((output_channels // 4)*input_channels) // num_context_steps,
            conv_type=conv_type
        )
        self.conv1 = SpectralNorm(
            convxd(
                in_channels=((output_channels // 4) * input_channels),
                out_channels=(output_channels // 8) * input_channels,
                kernel_size=3,
                padding=1
            )
        )
        self.d2 = DBlock(
            input_channels=((output_channels // 4)*input_channels) // num_context_steps,
            output_channels=((output_channels // 2)*input_channels) // num_context_steps,
            conv_type=conv_type
        )
        self.conv2 = SpectralNorm(
            convxd(
                in_channels=((output_channels // 2) * input_channels),
                out_channels=(output_channels // 4) * input_channels,
                kernel_size=3,
                padding=1
            )
        )
        self.d3 = DBlock(
            input_channels=((output_channels // 2)*input_channels) // num_context_steps,
            output_channels=(output_channels * input_channels) // num_context_steps,
            conv_type=conv_type
        )
        self.conv3 = SpectralNorm(
            convxd(
                in_channels=(output_channels * input_channels),
                out_channels=(output_channels // 2) * input_channels,
                kernel_size=3,
                padding=1
            )
        )
        self.d4 = DBlock(
            input_channels=(output_channels * input_channels) // num_context_steps,
            output_channels=(output_channels * 2 * input_channels) // num_context_steps,
            conv_type=conv_type
        )
        self.conv4 = SpectralNorm(
            convxd(
                in_channels=(output_channels * 2 * input_channels),
                out_channels=output_channels * input_channels,
                kernel_size=3,
                padding=1
            )
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data_list = []
        for i in range(x.shape[1]):
            x_new = x[:, i, :, :, :]
            x_new = self.space2depth(x_new,2)
            x_new = np.squeeze(x_new)
            x_new = self.d1(x_new)

            if i == 0:
                data_0 = x_new
            else:
                data_0 = torch.cat((data_0, x_new), 1)
                if i == 3:
                    data_0 = self.conv3(data_0)
                    data_list.append(data_0)

            x_new = self.d2(x_new)

