from torch import nn
from al_lblock import AlLBlock, SpatialAttention


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


