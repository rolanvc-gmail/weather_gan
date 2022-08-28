import torch
from torch import nn
import torch.nn.functional as F
from al_lblock import AlLBlock, SpatialAttention
import numpy as np
from torch.autograd import Variable
cuda = False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class AlLCStack(nn.Module):
    def __init__(self, ):
        super(AlLCStack, self).__init__()
        self.conv3_1 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.LBlock_1 = AlLBlock(8, 24)
        self.LBlock_2 = AlLBlock(24, 48)
        self.LBlock_3 = AlLBlock(48, 192)
        self.LBlock_4 = AlLBlock(192, 768)
        self.mask = SpatialAttention()

    def forward(self, x):
        x = self.conv3_1(x)
        x = self.LBlock_1(x)
        x = self.LBlock_2(x)
        x = self.LBlock_3(x)
        mask = self.mask(x)
        x = x * mask
        out = self.LBlock_4(x)
        return out


def test_latent_conditioning_stack():
    model = AlLCStack()
    batch_sz = 1
    z = Variable(Tensor(np.random.normal(0, 1, (batch_sz, 8, 8, 8))))  # latent variable input for latent conditioning stack
    assert z.shape == (batch_sz, 8, 8, 8)
    out = model(z)
    assert out.size() == (batch_sz, 768, 8, 8)
    y = torch.rand((batch_sz, 768, 8, 8))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert not torch.isnan(out).any(), "Output included NaNs"


def main():
    test_latent_conditioning_stack()
    print("AlLCStack....passed")


if __name__ == "__main__":
    main()
