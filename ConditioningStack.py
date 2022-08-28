import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from utils import get_conv_layer
from torch.nn.modules.pixelshuffle import PixelUnshuffle
from dblock import DBlock
from typing import Tuple
import numpy as np
import einops


def _mixing_layer(inputs: torch.Tensor, conv_block: nn.Module) -> torch.Tensor:
    # Convert from [batch_size, time, h, w, c] -> [batch_size, h, w, c * time]
    # then perform convolution on the output while preserving number of c.
    stacked_inputs = einops.rearrange(inputs, "b t c h w -> b (c t) h w")
    out = F.relu(conv_block(stacked_inputs))
    return out


class ConditioningStack(nn.Module):
    """
    ConditioningStacks (or ContextConditioningStack)
    """

    def __init__(self, input_channels: int = 1, output_channels: int = 768, num_context_steps: int = 4, conv_type: str = "standard", **kwargs):
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x is Tensor(2, 4, 1, 256,256)
        x = self.space2depth(x)
        # is Tensor(2, 4, 4, 64, 64)
        steps = x.size(1)
        # steps = 4
        scale_1 = []
        scale_2 = []
        scale_3 = []
        scale_4 = []
        for i in range(steps):
            s1 = self.d1(x[:, i, :, :, :])
            # s1 is Tensor(2,48,64,64)
            s2 = self.d2(s1)
            # s2 is Tensor(2,96,32,32)
            s3 = self.d3(s2)
            # s3 is Tensor(2,192,16,16)
            s4 = self.d4(s3)
            # s4 is Tensor(2,384,8,8)
            scale_1.append(s1)
            scale_2.append(s2)
            scale_3.append(s3)
            scale_4.append(s4)

            # scale_1 is list of Tensor(2,48,64,64)
            # scale_2 is list of Tensor(2,96,32,32)
            # scale_3 is list of Tensor(2,192,16,16)
            # scale_4 is list of Tensor(2,384,8,8)

        scale_1 = torch.stack(scale_1, dim=1)  # B, T, C, H, W and want along C dimension
        scale_2 = torch.stack(scale_2, dim=1)  # B, T, C, H, W and want along C dimension
        scale_3 = torch.stack(scale_3, dim=1)  # B, T, C, H, W and want along C dimension
        scale_4 = torch.stack(scale_4, dim=1)  # B, T, C, H, W and want along C dimension
        # Mixing layer
        # scale_1 is Tensor(2,4,48,64,64)
        # scale_2 is Tensor(2,4,96,32,32)
        # scale_3 is Tensor(2,4,192,16,16)
        # scale_4 is Tensor(2,4,384,8,8)
        scale_1 = _mixing_layer(scale_1, self.conv1)
        scale_2 = _mixing_layer(scale_2, self.conv2)
        scale_3 = _mixing_layer(scale_3, self.conv3)
        scale_4 = _mixing_layer(scale_4, self.conv4)
        return scale_1, scale_2, scale_3, scale_4


def test_condition_stack():

    input_channels: int = 1  # Number of input channels per image
    conv_type: str = "standard"  # type of convolution to use
    latent_channels: int = 768  # Number of channels the latent space should be reshaped to.
    context_channels: int = 384

    conditioning_stack = ConditioningStack(input_channels=input_channels, conv_type=conv_type, output_channels=context_channels)
    model = ConditioningStack().cuda()
    batch_size = 4
    x = torch.rand((batch_size, 22, 1, 256, 256))
    out = model(x)
    y = torch.rand((2, 96, 32, 32))
    loss = F.mse_loss(y, out[0])
    loss.backward()
    assert len(out) == 4
    assert out[0].size() == (2, 96, 32, 32)
    assert out[1].size() == (2, 192, 16, 16)
    assert out[2].size() == (2, 384, 8, 8)
    assert out[3].size() == (2, 768, 4, 4)
    assert not all(torch.isnan(out[i]).any() for i in range(len(out))), "Output included NaNs"


def main():
    test_condition_stack()
    print("Conditioning Stack....passed")


if __name__ == "__main__":
    main()


