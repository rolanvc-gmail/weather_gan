import torch
from torch import nn
from torch.nn import functional as F
from al_spectral_norm import AlSpectralNorm
from al_dblock_down import AlDBlockDown
import numpy as np
from al_space_to_depth import space_to_depth_1 as al_space_to_depth


class AlConditioningStack(nn.Module):
    def __init__(self, in_channels):
        super(AlConditioningStack, self).__init__()
        self.DBlockDown_1 = AlDBlockDown(4, 24)
        self.DBlockDown_2 = AlDBlockDown(24, 48)
        self.DBlockDown_3 = AlDBlockDown(48, 96)
        self.DBlockDown_4 = AlDBlockDown(96, 192)
        self.relu = nn.ReLU()
        self.conv3_1 = AlSpectralNorm(nn.Conv2d(96, 48, 3, stride=1, padding=1))
        self.conv3_2 = AlSpectralNorm(nn.Conv2d(192, 96, 3, stride=1, padding=1))
        self.conv3_3 = AlSpectralNorm(nn.Conv2d(384, 192, 3, stride=1, padding=1))
        self.conv3_4 = AlSpectralNorm(nn.Conv2d(768, 384, 3, stride=1, padding=1))

    def forward(self, x):
        dataList = []

        for i in range(x.shape[1]):
            x_new = x[:, i, :, :, :]
            x_new = al_space_to_depth(x_new, 2)
            x_new = np.squeeze(x_new)
            x_new = self.DBlockDown_1(x_new)

            if i == 0:
                data_0 = x_new
            else:
                data_0 = torch.cat((data_0, x_new), 1)
                if i == 3:
                    data_0 = self.conv3_1(data_0)
                    dataList.append(data_0)
            x_new = self.DBlockDown_2(x_new)

            if i == 0:
                data1 = x_new
            else:
                data1 = torch.cat((data1, x_new), 1)
                if i == 3:
                    data1 = self.conv3_2(data1)
                    dataList.append(data1)
            x_new = self.DBlockDown_3(x_new)

            if i == 0:
                data2 = x_new
            else:
                data2 = torch.cat((data2, x_new), 1)
                if i == 3:
                    data2 = self.conv3_3(data2)
                    dataList.append(data2)
            x_new = self.DBlockDown_4(x_new)

            if i == 0:
                data3 = x_new
            else:
                data3 = torch.cat((data3, x_new), 1)
                if i == 3:
                    data3 = self.conv3_4(data3)
                    dataList.append(data3)

        return dataList  # should equal around 4 stacks with 4 elements per stack


def test_conditioning_stack():
    model = AlConditioningStack(24)
    batch_sz = 4
    x = torch.rand((batch_sz, 4, 1, 256, 256))
    out = model(x)
    assert len(out) == 4
    assert out[0].size() == (batch_sz, 48, 64, 64)
    assert out[1].size() == (batch_sz, 96, 32, 32)
    assert out[2].size() == (batch_sz, 192, 16, 16)
    assert out[3].size() == (batch_sz, 384, 8, 8)
    assert not all(torch.isnan(out[i]).any() for i in range(len(out))), "Output included NaNs"
    y = torch.rand((batch_sz, 96, 32, 32))
    loss = F.mse_loss(y, out[1])
    loss.backward()


def main():
    test_conditioning_stack()
    print("AlConditioningStack passed")


if __name__ == "__main__":
    main()
