import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from typing import Tuple


class ConvGRUCell(nn.Module):
    """
    ConvGRUCell
    """
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3, sn_eps: float = 0.0001, **kwargs):
        """

        :param input_channels:
        :param output_channels:
        :param kernel_size: kernel_size for the convolutions. Default: 3.
        :param sn_eps: constant for Spectral Normalization. Default: 1e-4
        """
        super(ConvGRUCell, self).__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        self.kernel_size = self.config["kernel_size"]
        self.sn_eps = self.config["sn_eps"]
        self.read_gate_conv = SpectralNorm(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(kernel_size, kernel_size),
                padding=1,
            ),
            eps=sn_eps,
        )
        self.update_gate_conv = SpectralNorm(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(kernel_size, kernel_size),
                padding=1,
            ),
            eps=sn_eps,
        )

        self.output_conv = SpectralNorm(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(kernel_size, kernel_size),
                padding=1,
            ),
            eps=sn_eps,
        )

    def forward(self, x: torch.Tensor, prev_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param x: Input tensor
        :param prev_state: the previous state
        :return: new_tensor, new_state

        """

        # Concatenate teh input and the previous state along the  channel axis.
        xh = torch.cat([x, prev_state], dim=1)

        # Read gate of the GRU.
        read_gate = torch.sigmoid(self.read_gate_conv(xh))

        # Update gate of the GRU.
        update_gate = torch.sigmoid(self.update_gate_conv(xh))

        # Gate the inputs
        gated_input = torch.cat([x, read_gate * prev_state], dim=1)

        # Gate the cell and state/outputs.
        c = F.relu(self.output_conv(gated_input))
        out = update_gate * prev_state + (1.0 - update_gate)*c
        new_state = out

        return out, new_state


def test_conv_gru():
    model = ConvGRUCell(input_channels=768 + 384,
                        output_channels=384,
                        kernel_size=3)
    x = torch.rand((2, 768, 32, 32))
    out, hidden = model(x, torch.rand(2, 384, 32, 32))
    y = torch.rand((2, 384, 32, 32))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert out.size() == (2, 384, 32, 32)
    assert not torch.isnan(out).any(), "Output included NaNs"


def main():
    test_conv_gru()
    print("ConvGRU passed unit test")


if __name__ == "__main__":
    main()
