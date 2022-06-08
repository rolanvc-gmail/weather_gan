import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from utils import get_conv_layer
from torch.nn.modules.pixelshuffle import PixelUnshuffle
from dblock import DBlock

# todo: pass the unit test in main()


class SpatialDiscriminator(nn.Module):
    """
    SpatialDiscriminator
    """
    def __init__(self, input_channels: int = 1, num_time_steps: int = 8, num_layers: int = 4, conv_type: str = "standard", **kwargs):
        """

        :param input_channels: input_channels
        :param num_time_steps: number of time steps to use. The paper uses 8/18 time steps.
        :param num_layers: Number of intermediate DBlock layers to use. The paper uses 5.
        :param conv_type: type of convolution to use. See utils/get_conv_layer
        :param kwargs:
        """
        super(SpatialDiscriminator, self).__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        input_channels = self.config["input_channels"]
        num_time_steps = self.config["num_time_steps"]
        num_layers = self.config["num_layers"]
        conv_type = self.config["conv_type"]
        self.num_time_steps = num_time_steps
        self.mean_pool = torch.nn.AvgPool2d(2)
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        internal_channels = 24
        # assume input_channels = 1
        self.d1 = DBlock(
            input_channels=4 * input_channels,
            output_channels=2 * internal_channels * input_channels,
            first_relu=False,
            conv_type=conv_type,
        ) # input_channels = 4; output_channels = 48
        self.intermediate_d_blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            internal_channels *= 2
            self.intermediate_d_blocks.append(
                DBlock(
                    input_channels=internal_channels * input_channels,
                    output_channels=2 * internal_channels * input_channels,
                    conv_type=conv_type
                )
            )
        self.d6 = DBlock(
            input_channels=2 * internal_channels * input_channels,
            output_channels=2 * internal_channels * input_channels,
            keep_same_output=True,
            conv_type=conv_type,
        )

        # Spectrally normalized layer for binary classification
        self.fc = SpectralNorm(torch.nn.Linear(2*internal_channels*input_channels, 1))
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(2*internal_channels * input_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x should be the chosen 8 or so
        indxs = torch.randint(low=0, high=x.size()[1], size=(self.num_time_steps,))
        representations = []
        for idx in indxs:
            # x = Tensor(2, 18,1, 256,256)
            rep = self.mean_pool(x[:, idx, :, :, :])
            # rep is Tensor(2, 1,128,128)
            rep = self.space2depth(rep)
            # rep is Tensor(2,4,64,64)
            rep = self.d1(rep)
            for d in self.intermediate_d_blocks:
                rep = d(rep)
            # rep is Tensor(2,768,2,2)
            rep = self.d6(rep)
            # rep is Tensor(2, 768,2,2)
            rep = torch.sum(F.relu(rep), dim=[2, 3])
            # rep is Tensor(2,768)
            rep = self.bn(rep)
            rep = self.fc(rep)
            representations.append(rep)

        x = torch.stack(representations, dim=1)

        x = torch.sum(x, keepdim=True, dim=1)
        return x


def test_spatial_discriminator():
    model = SpatialDiscriminator(input_channels=1)
    x = torch.rand((2, 18, 1, 256, 256))
    out = model(x)
    assert out.shape == (2, 1, 1)
    y = torch.rand((2, 1, 1))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert not torch.isnan(out).any()


def main():
    test_spatial_discriminator()
    print("Spatial Discriminator...passed")


if __name__ == "__main__":
    main()
