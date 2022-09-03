import torch
from torch import nn
from al_dblock import AlDBlock, AlDBlockDown, AlDBlock3D_1, AlDBlock3D_2
from al_space_to_depth import space_to_depth_1 as space_to_depth
import random


class AlTemporalDiscriminator(nn.Module):
    def __init__(self, ):
        super(AlTemporalDiscriminator, self).__init__()
        self.DBlock3D_1 = AlDBlock3D_1(4, 48)
        self.DBlock3D_2 = AlDBlock3D_2(48, 96)
        self.DBlockDown_3 = AlDBlockDown(96, 192)
        self.DBlockDown_4 = AlDBlockDown(192, 384)
        self.DBlockDown_5 = AlDBlockDown(384, 768)
        self.DBlock_6 = AlDBlock(768, 768)
        self.sum_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(in_features=768 * 1 * 1, out_features=1)

    def forward(self, x):
        T_H = random.sample(range(0, 128), 1)  # I ADDED THIS
        T_W = random.sample(range(0, 128), 1)  # I ADDED THIS
        x = x[:, :, T_H[0]:T_H[0] + 128, T_W[0]:T_W[0] + 128]  # I ADDED THIS
        x = space_to_depth(x, 2)
        x = self.DBlock3D_1(x)
        x = self.DBlock3D_2(x)

        for i in range(0, x.shape[2]):
            x_temp = x[:, :, i, :, :]
            x_temp = self.DBlockDown_3(x_temp)
            x_temp = self.DBlockDown_4(x_temp)
            x_temp = self.DBlockDown_5(x_temp)
            x_temp = self.DBlock_6(x_temp)
            x_temp = self.sum_pool(x_temp)
            x_temp = x_temp.view(x_temp.shape[0], x_temp.shape[1])
            x_temp = x_temp * 4
            out = self.linear(x_temp)

            if i == 0:
                data = out
            else:
                data = data + out

        data = torch.squeeze((data))
        return data
