import torch
import torch.nn as nn
import torch.nn.functional as F
from TemporalDiscriminator import TemporalDiscriminator
from SpatialDiscriminator import SpatialDiscriminator


class Discriminator(nn.Module):
    """
    The Discriminator has 2 components: the SpatialDiscriminator, and the TemporalDiscriminator. When data is passed through,
    the data is input to both separately, then results are concatenated. The shape of the output is batch x 2 x 1.
    """
    def __init__(self, input_channels: int = 12, num_spatial_frames: int = 8, conv_type: str = "standard", **kwargs):
        super(Discriminator, self).__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        input_channels = self.config["input_channels"]
        num_spatial_frames = self.config["num_spatial_frames"]
        conv_type = self.config["conv_type"]

        self.spatial_discriminator = SpatialDiscriminator(input_channels=input_channels, num_time_steps=num_spatial_frames, conv_type=conv_type)
        self.temporal_discriminator = TemporalDiscriminator(input_channels=input_channels, conv_type=conv_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_loss = self.spatial_discriminator(x)
        temporal_loss = self.temporal_discriminator(x)

        # spatial_loss and temporal_loss are both Tensor(2,1,1)
        return torch.cat([spatial_loss, temporal_loss], dim=1)


def test_discriminator():
    model = Discriminator(input_channels=1)
    x = torch.rand((2, 18, 1, 256, 256))
    out = model(x)
    assert out.shape == (2, 2, 1)
    y = torch.rand((2, 2, 1))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert not torch.isnan(out).any()


def main():
    test_discriminator()
    print("Discriminator passed unit test")


if __name__ == "__main__":
    main()



