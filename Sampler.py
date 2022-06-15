import torch
from torch import nn
import torch.nn.functional as F
from ConvGRU import ConvGRU
from gblock import GBlock
from gblock_upsample import UpsampleGBlock
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from torch.nn.modules.pixelshuffle import PixelShuffle
from typing import List
import einops
from ConditioningStack import ConditioningStack
from LatentConditioningStack import LatentConditioningStack


class Sampler(nn.Module):
    def __init__(self, forecast_steps: int = 18,
                 latent_channels: int = 768,
                 context_channels: int = 384,
                 output_channels: int = 1,
                 **kwargs):
        """

        :param forecast_steps:
        :param latent_channels:
        :param context_channels:
        :param output_channels:
        :param kwargs:
        """
        super(Sampler, self).__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        self.forecast_steps = self.config["forecast_steps"]
        latent_channels = self.config["latent_channels"]
        context_channels = self.config["context_channels"]
        output_channels = self.config["output_channels"]

        self.convGRU1 = ConvGRU(
            input_channels=latent_channels + context_channels,
            output_channels=context_channels,
            kernel_size=3
        )
        self.gru_conv_1x1 = SpectralNorm(
            torch.nn.Conv2d(in_channels=context_channels,
                            out_channels=latent_channels,
                            kernel_size=(1, 1)
                            )
        )
        self.g1 = GBlock(input_channels=latent_channels, output_channels=latent_channels)
        self.up_g1 = UpsampleGBlock(input_channels=latent_channels, output_channels=latent_channels // 2)

        self.convGRU2 = ConvGRU(input_channels=latent_channels // 2 + context_channels//2,
                                output_channels=context_channels // 2,
                                kernel_size=3)
        self.gru_conv_1x1_2 = SpectralNorm(
            torch.nn.Conv2d(in_channels=context_channels//2,
                            out_channels=latent_channels//2,
                            kernel_size=(1, 1),
                            )
        )
        self.g2 = GBlock(input_channels=latent_channels//2, output_channels=latent_channels//2)
        self.up_g2 = UpsampleGBlock(input_channels=latent_channels//2,
                                    output_channels=latent_channels//4)

        self.convGRU3 = ConvGRU(input_channels=latent_channels // 4 + context_channels // 4,
                                output_channels=context_channels // 4,
                                kernel_size=3
                                )
        self.gru_conv_1x1_3 = SpectralNorm(
            torch.nn.Conv2d(in_channels=context_channels // 4,
                            out_channels=latent_channels // 4,
                            kernel_size=(1, 1)
                            )
        )

        self.g3 = GBlock(input_channels=latent_channels // 4, output_channels=latent_channels // 4)
        self.up_g3 = UpsampleGBlock(input_channels=latent_channels // 4,
                                    output_channels=latent_channels // 8)

        self.convGRU4 = ConvGRU(input_channels=latent_channels // 8 + context_channels // 8,
                                output_channels=context_channels // 8,
                                kernel_size=3
                                )
        self.gru_conv_1x1_4 = SpectralNorm(
            torch.nn.Conv2d(in_channels=context_channels // 8,
                            out_channels=latent_channels // 8,
                            kernel_size=(1, 1)
                            )
        )
        self.g4 = GBlock(input_channels=latent_channels // 8, output_channels=latent_channels // 8)
        self.up_g4 = UpsampleGBlock(input_channels=latent_channels // 8,
                                    output_channels=latent_channels // 16)

        self.bn = torch.nn.BatchNorm2d(latent_channels // 16)
        self.relu = torch.nn.ReLU()
        self.conv_1x1 = SpectralNorm(
            torch.nn.Conv2d(
                in_channels=latent_channels // 16,
                out_channels=4 * output_channels,
                kernel_size=(1, 1)
            )
        )
        self.depth2space = PixelShuffle(upscale_factor=2)

    def forward(self, conditioning_states: List[torch.Tensor], latent_dim: torch.Tensor) -> torch.Tensor:
        init_states = conditioning_states

        latent_dim = einops.repeat(latent_dim, "b c h w -> (repeat b) c h w", repeat=init_states[0].shape[0])
        hidden_states = [latent_dim] * self.forecast_steps

        # Layer 4 (bottom most)
        hidden_states = self.convGRU1(hidden_states, init_states[3])
        hidden_states = [self.gru_conv_1x1_1(h) for h in hidden_states]
        hidden_states = [self.g1(h) for h in hidden_states]
        hidden_states = [self.up_g1(h) for h in hidden_states]

        # Layer 3
        hidden_states = self.convGRU2(hidden_states, init_states[2])
        hidden_states = [self.gru_conv_1x1_2(h) for h in hidden_states]
        hidden_states = [self.g2(h) for h in hidden_states]
        hidden_states = [self.up_g2(h) for h in hidden_states]

        # Layer 2
        hidden_states = self.convGRU3(hidden_states, init_states[1])
        hidden_states = [self.gru_conv_1x1_3(h) for h in hidden_states]
        hidden_states = [self.g3(h) for h in hidden_states]
        hidden_states = [self.up_g3(h) for h in hidden_states]

        # Layer 1 (top most)
        hidden_states = self.convGRU4(hidden_states, init_states[0])
        hidden_states = [self.gru_conv_1x1_4(h) for h in hidden_states]
        hidden_states = [self.g4(h) for h in hidden_states]
        hidden_states = [self.up_g4(h) for h in hidden_states]

        # Output layer.
        hidden_states = [F.relu(self.bn(h)) for h in hidden_states]
        hidden_states = [self.conv_1x1(h) for h in hidden_states]
        hidden_states = [self.depth2space(h) for h in hidden_states]

        forecasts = torch.stack(hidden_states, dim=1)

        return forecasts


def test_sampler():
    input_channels = 1
    conv_type = "standard"
    context_channels = 384
    latent_channels = 768
    forecast_steps = 18
    output_shape = 256
    conditioning_stack = ConditioningStack(
        input_channels=input_channels,
        conv_type=conv_type,
        output_channels=context_channels,
    )
    latent_stack = LatentConditioningStack(
        shape=(8 * input_channels, output_shape // 32, output_shape // 32),
        output_channels=latent_channels,
    )
    sampler = Sampler(
        forecast_steps=forecast_steps,
        latent_channels=latent_channels,
        context_channels=context_channels,
    )
    latent_stack.eval()
    conditioning_stack.eval()
    sampler.eval()
    x = torch.rand((2, 4, 1, 256, 256))
    with torch.no_grad():
        latent_dim = latent_stack(x)
        assert not torch.isnan(latent_dim).any()
        init_states = conditioning_stack(x)
        assert not all(torch.isnan(init_states[i]).any() for i in range(len(init_states)))
        # Expand latent dim to match batch size
        latent_dim = einops.repeat(
            latent_dim, "b c h w -> (repeat b) c h w", repeat=init_states[0].shape[0]
        )
        assert not torch.isnan(latent_dim).any()
        hidden_states = [latent_dim] * forecast_steps
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = sampler.convGRU1(hidden_states, init_states[3])
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.gru_conv_1x1(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.g1(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.up_g1(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        # Layer 3.
        hidden_states = sampler.convGRU2(hidden_states, init_states[2])
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.gru_conv_1x1_2(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.g2(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.up_g2(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))

        # Layer 2.
        hidden_states = sampler.convGRU3(hidden_states, init_states[1])
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.gru_conv_1x1_3(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.g3(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.up_g3(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))

        # Layer 1 (top-most).
        hidden_states = sampler.convGRU4(hidden_states, init_states[0])
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.gru_conv_1x1_4(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.g4(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.up_g4(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))

        # Output layer.
        hidden_states = [F.relu(sampler.bn(h)) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.conv_1x1(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.depth2space(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))

def main():
    test_sampler()
    print("Sampler passed Unit Test")


if __name__ == "__main__":
    main()



