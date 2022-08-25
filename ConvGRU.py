import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvGRUCell import ConvGRUCell


class ConvGRU(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3, sn_eps: float = 0.0001):
        super(ConvGRU, self).__init__()
        self.cell = ConvGRUCell(input_channels, output_channels, kernel_size, sn_eps)

    def forward(self, x:torch.Tensor, hidden_state=None) -> torch.Tensor:
        outputs = []
        for step in range(len(x)):
            output, hidden_state = self.cell(x[step], hidden_state)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        return outputs


def test_convgru():
    model = ConvGRU(
        input_channels=768 + 384,
        output_channels=384,
        kernel_size=3,
    )
    init_states = [torch.rand((2, 384, 32, 32)) for _ in range(4)]
    # Expand latent dim to match batch size
    x = torch.rand((2, 768, 32, 32))
    hidden_states = [x] * 18
    out = model(hidden_states, init_states[3])
    y = torch.rand((18, 2, 384, 32, 32))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert out.size() == (18, 2, 384, 32, 32)
    assert not torch.isnan(out).any(), "Output included NaNs"


def main():
    test_convgru()
    print("ConvGRU passed unit test")


if __name__ == "__main__":
    main()
