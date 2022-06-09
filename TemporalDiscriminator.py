import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from utils import get_conv_layer
from dblock import DBlock
from torch.nn.modules.pixelshuffle import PixelUnshuffle
import einops


class TemporalDiscriminator(nn.Module):
    """
    TemporalDiscriminator
    """
    def __init__(self, input_channels: int = 12, num_layers: int = 3, conv_type: str = "standard", **kwargs):
        """

        :param input_channels: number of channels per time_step.
        :param num_layers: Number of intermediate DBlock layers to use.
        :param conv_type: Type of convolutions to use. See utils/get_conv_layer()
        :param kwargs:
        """
        super(TemporalDiscriminator, self).__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        input_channels = self.config["input_channels"]
        num_layers = self.config["num_layers"]
        conv_type = self.config["conv_type"]
        self.downsample = torch.nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        internal_channels = 48
        self.first_3d = DBlock(input_channels=4*input_channels,
                               output_channels=internal_channels * input_channels,
                               conv_type="3d",
                               first_relu=False,
                               )
        self.second_3d = DBlock(input_channels=internal_channels * input_channels,
                                output_channels=2 * internal_channels*input_channels,
                                conv_type="3d")
        self.intermediate_d_blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            internal_channels *= 2
            self.intermediate_d_blocks.append(
                DBlock( input_channels= internal_channels * input_channels,
                        output_channels=2 * internal_channels*input_channels,
                        conv_type=conv_type
                        )

            )
        self.last_d = DBlock(input_channels=2 * internal_channels * input_channels,
                             output_channels=2 * internal_channels * input_channels,
                             keep_same_output=True,
                             conv_type=conv_type)
        self.fc = SpectralNorm(torch.nn.Linear(2 * internal_channels * input_channels, 1))
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(2 * internal_channels * input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is Tensor(2,22,1,256,256)
        x = self.downsample(x)
        # x is Tensor(2,22,1,128,128)
        x = self.space2depth(x)
        # x is Tensor(2,22,4,64,64)
        # From (B, T, C, H, W) convert to (B, C, T, H, W)
        x = torch.permute(x, dims=(0, 2, 1, 3, 4))
        # x is Tensor(2,4,22,64,64)
        x = self.first_3d(x)
        # x is Tensor(2, 48, 11, 32,32) -- this looks correct according to paper.
        x = self.second_3d(x) # ---this is supposed to double the channels
        # x is Tensor(2, 96, 5, 16,16)
        x = torch.permute(x, dims=(0, 2, 1, 3, 4))
        # x is Tensor(2, 5, 96, 16,16)
        representations = []
        for idx in range(x.size(1)):
            rep = x[:, idx, :, :, :]
            for d in self.intermediate_d_blocks:
                rep = d(rep)

            rep = self.last_d(rep)

            rep = torch.sum(F.relu(rep), dim=[2, 3])
            rep = self.bn(rep)
            rep = self.fc(rep)

            representations.append(rep)

        x = torch.stack(representations, dim=1)

        x = torch.sum(x, keepdim=True, dim=1)
        return x


def test_temporal_discriminator():
    model = TemporalDiscriminator(input_channels=1)
    x = torch.rand(2, 22, 1, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 1)
    y = torch.rand((2, 1, 1,))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert not torch.isnan(out).any()


def main():
    test_temporal_discriminator()
    print("Temporal Discriminator....passed")


if __name__ == "__main__":
    main()