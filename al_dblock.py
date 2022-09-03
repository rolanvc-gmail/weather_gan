from torch import nn
from al_spectral_norm import AlSpectralNorm


class AlDBlockDown(nn.Module):  ### for sampler ####
    def __init__(self, in_channels, out_channels):
        super(AlDBlockDown, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = AlSpectralNorm( nn.Conv2d(in_channels, out_channels, 1))
        self.conv3_1 = AlSpectralNorm(nn.Conv2d(in_channels, in_channels, 3,stride = 1,padding = 1))
        self.conv3_2 = AlSpectralNorm(nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1))
        self.maxpool = nn.MaxPool2d(2, stride = 2, return_indices = False, ceil_mode = False)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool(x1)
        x2 = self.relu(x)
        x2 = self.conv3_1(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool(x2)
        out = x1 + x2
        return out


class AlDBlockDownFirst(nn.Module):  ### for sampler #### (only difference is Relu in forward ?)
    def __init__(self, in_channels, out_channels):
        super(AlDBlockDownFirst, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = AlSpectralNorm(nn.Conv2d(in_channels, out_channels, 1))
        self.conv3_1 = AlSpectralNorm(nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1))
        self.conv3_2 = AlSpectralNorm(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool(x1)

        x2 = self.conv3_1(x)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool(x2)
        out = x1 + x2
        return out


class AlDBlock(nn.Module):  ### for both spatial and temporal discriminators
    def __init__(self, in_channels, out_channels):
        super(AlDBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = AlSpectralNorm(nn.Conv2d(in_channels, out_channels, 1))
        self.conv3 = AlSpectralNorm(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.relu(x)
        x2 = self.conv3(x2)
        x2 = self.relu(x2)
        x2 = self.conv3(x2)

        out = x1 + x2
        return out


class AlDBlock3D_1(nn.Module):  ### for temporal discriminators
    def __init__(self, in_channels, out_channels):
        super(AlDBlock3D_1, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = AlSpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)))
        self.conv3_1 = AlSpectralNorm(nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=1, stride=1))
        self.conv3_2 = AlSpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1, stride=1))
        self.maxpool_3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool_3d(x1)
        x2 = self.conv3_1(x)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool_3d(x2)
        out = x1 + x2
        return out


class AlDBlock3D_2(nn.Module):  ### for temporal discriminators ### (only difference is Relu in forward ?)
    def __init__(self, in_channels, out_channels):
        super(AlDBlock3D_2, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = AlSpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)))
        self.conv3_1 = AlSpectralNorm(nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=1, stride=1))
        self.conv3_2 = AlSpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1, stride=1))
        self.maxpool_3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool_3d(x1)
        x2 = self.relu(x)
        x2 = self.conv3_1(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool_3d(x2)
        out = x1 + x2
        return out
