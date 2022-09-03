import torch
from torch import nn
from al_dblock import AlDBlockDown, AlDBlockDownFirst, AlDBlock
from al_spectral_norm import AlSpectralNorm
from al_space_to_depth import space_to_depth_1 as space_to_depth


class AlSpatialDiscriminator(nn.Module):
    def __init__(self,):
        super(AlSpatialDiscriminator, self).__init__()
        self.DBlockDown_1 = AlDBlockDownFirst(4, 48)
        self.DBlockDown_2 = AlDBlockDown(48, 96)
        self.DBlockDown_3 = AlDBlockDown(96, 192)
        self.DBlockDown_4 = AlDBlockDown(192, 384)
        self.DBlockDown_5 = AlDBlockDown(384, 768)
        self.DBlock_6 = AlDBlock(768, 768)
        self.sum_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=2)  # I ADDED THIS
        self.linear = AlSpectralNorm(nn.Linear(in_features=768 * 1 *1, out_features=1))

    def forward(self, x):
        x = self.avgPool(x)  # used avg pool instead of random crop sampling

        for i in range(x.shape[1]):
            x_temp = x[:, i]
            x_temp = x_temp.view(x_temp.shape[0] ,1 ,x_temp.shape[1] ,x_temp.shape[2])
            x_temp = space_to_depth(x_temp ,2)
            x_temp = torch.squeeze(x_temp)
            x_temp = self.DBlockDown_1(x_temp)
            x_temp = self.DBlockDown_2(x_temp)
            x_temp = self.DBlockDown_3(x_temp)
            x_temp = self.DBlockDown_4(x_temp)
            x_temp = self.DBlockDown_5(x_temp)
            x_temp = self.DBlock_6(x_temp)
            x_temp = self.sum_pool(x_temp)
            x_temp = x_temp.view(x_temp.shape[0] ,x_temp.shape[1])
            x_temp = x_temp * 4
            out = self.linear(x_temp)

            if i == 0:
                data = out
            else:
                data = data + out

        data = torch.squeeze((data))
        return data