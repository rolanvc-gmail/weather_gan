import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from lblock import LBlock
from attention import AttentionLayer


class LatentConditioningStack(nn.Module):
    def __init__(self,
                 shape: (int, int, int) = (8, 8, 8),
                 output_channels: int = 768,
                 use_attention: bool = True,
                 **kwargs):
        super(LatentConditioningStack, self).__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        shape = self.config["shape"]
        output_channels = self.config["output_channels"]
        use_attention = self.config["use_attention"]

        self.shape = shape
        self.use_attention = use_attention
        self.distribution = normal.Normal(loc=torch.Tensor([0.0]), scale=torch.Tensor([1.0]))
        self.conv_3x3 = SpectralNorm(
            torch.nn.Conv2d(
                in_channels=shape[0],
                out_channels=shape[0],
                kernel_size=(3, 3),
                padding=1
            )
        )
        self.l_block1 = LBlock(input_channels=shape[0], output_channels=output_channels // 32)
        self.l_block2 = LBlock(input_channels=output_channels // 32, output_channels=output_channels // 16)
        self.l_block3 = LBlock(input_channels=output_channels // 16, output_channels=output_channels // 4)

        if self.use_attention:
            self.attention_block = AttentionLayer(input_channels=output_channels//4, output_channels=output_channels//4)

        self.l_block4 = LBlock(input_channels=output_channels // 4, output_channels=output_channels)

    def forward(self, x: torch.Tensor):
        # independent draws from Normal distribution
        z = self.distribution.sample(self.shape)
        # Batch is at end for some reason, reshape
        z = torch.permute(z, (3, 0, 1, 2)).type_as(x)

        # 3 LBlocks
        # z is Tensor(1, 8, 8, 8)
        z = self.conv_3x3(z)
        # z is Tensor(1, 8, 8, 8)
        z = self.l_block1(z)
        # z is Tensor(1, 24, 8, 8)
        z = self.l_block2(z)
        # z is Tensor(1, 48, 8, 8)
        z = self.l_block3(z)
        # z is Tensor(1, 192, 8, 8)

        # ATT block
        z = self.attention_block(z)

        # final L Block
        z = self.l_block4(z)
        return z


def test_latent_conditioning_stack():
    model = LatentConditioningStack()
    batch_sz = 4
    x = torch.rand((1, 4, 1, 128, 128))
    out = model(x)
    assert out.size() == (1, 768, 8, 8)
    y = torch.rand((1, 768, 8, 8))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert not torch.isnan(out).any(), "Output included NaNs"


def main():
    test_latent_conditioning_stack()
    print("Latent Conditioning Stack....passed")


if __name__ == "__main__":
    main()
