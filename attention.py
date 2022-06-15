import torch
import torch.nn as nn
from torch.nn import functional as F
import einops


def attention_einsum(q, k, v):
    """ Apply the attention operator to tensors of shape [h, w, c]."""

    # Reshape 3D tensors to 2D tensors with first dimension L = h x w.
    k = einops.rearrange(k, "h w c -> (h w) c")  # [h, w, c] -> [L, c]
    v = einops.rearrange(v, "h w c -> (h w) c")  # [h, w, c] -> [L, c]

    beta = F.softmax(torch.einsum("hwc, Lc->hwL", q, k), dim=-1)

    out = torch.einsum("hwL, Lc->hwc", beta, v)
    return out


class AttentionLayer(torch.nn.Module):
    """
    Attention Module
    """
    def __init__(self, input_channels: int, output_channels: int, ratio_kq=8, ratio_v=8):
        super(AttentionLayer, self).__init__()
        self.ratio_kq = ratio_kq
        self.ratio_v = ratio_v
        self.output_channels = output_channels
        self.input_channels = input_channels

        self.query = torch.nn.Conv2d(in_channels=input_channels,
                                     out_channels=self.output_channels // self.ratio_kq,
                                     kernel_size=(1, 1),
                                     padding="valid",
                                     bias=False)

        self.key = torch.nn.Conv2d(in_channels=input_channels,
                                   out_channels=self.output_channels // self.ratio_kq,
                                   kernel_size=(1, 1),
                                   padding="valid",
                                   bias=False)
        self.value = torch.nn.Conv2d(in_channels=input_channels,
                                     out_channels=self.output_channels // self.ratio_v,
                                     kernel_size=(1, 1),
                                     padding="valid",
                                     bias=False)

        self.last_conv = torch.nn.Conv2d(in_channels=self.output_channels // 8,
                                         out_channels=self.output_channels,
                                         kernel_size=(1, 1),
                                         padding="valid",
                                         bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        out = []
        for b in range(x.shape[0]):
            out.append(attention_einsum(query[b], key[b], value[b]))

        out = torch.stack(out, dim=0)
        out = self.gamma * self.last_conv(out)

        return out + x


def test_attention():
    input_channels = 768 
    output_channels = 768
    x = torch.rand((1, 768, 8, 8))
    model = AttentionLayer(input_channels=input_channels, output_channels=output_channels)
    out = model(x)
    y = torch.rand((1, 768, 8, 8))
    assert out.size() == (1, 768, 8, 8)
    loss = F.mse_loss(y, out)
    loss.backward()
    assert not torch.isnan(out).any(), "Output included NaNs"
    


def main():
    test_attention()
    print("AttentionLayer passed unit test")

    
if __name__ == "__main__":
    main()