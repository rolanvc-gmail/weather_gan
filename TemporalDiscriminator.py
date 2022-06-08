import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from utils import get_conv_layer
from dblock import DBlock
from torch.nn.modules.pixelshuffle import PixelUnshuffle


class TemporalDiscriminator(nn.Module):
    """
    TemporalDiscriminator
    """
    def __init__(self, input_channels: int = 12, num_layers: int = 3, conv_type: str = "standard", **kwargs):
        super(TemporalDiscriminator, self).__init__()
        config = locals()
        config.pop("__class__")
        config.pop("__self__")
        self.config = kwargs.get("config", config)
        input_channels = self.config["input_channels"]
        num_layers = self.config["num_layers"]
        conv_type = self.config["conv_type"]
        self.first_3d = DBlock(conv_type="3d")
        self.second_3d = DBlock(conv_type="3d")
        self.first_d = DBlock(input_channels )
        self.second_d = DBlock(input_channels )
        self.space2depth = PixelUnshuffle(downscale_factor=2)

    def forward(self, x):
        x = self.space2depth(x)
        x = self.first_3d(x)
        y  = self.second_3d(x)


