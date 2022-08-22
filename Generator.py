import torch
from torch import nn
from torch.nn import functional as F
from ConditioningStack import ConditioningStack
from LatentConditioningStack import LatentConditioningStack
from Sampler import Sampler


class Generator(nn.Module):
    def __init__(self, conditioning_stack: nn.Module, latent_stack: nn.Module, sampler: nn.Module):
        """
            The Generator expects input of (batch, 4, 1, 256, 256) and outputs (batch, 18, 1, 256, 256)
        :param conditioning_stack: 
        :param latent_stack: 
        :param sampler: 
        """
        super(Generator, self).__init__()
        self.conditioning_stack = conditioning_stack
        self.latent_stack = latent_stack
        self.sampler = sampler

    def forward(self, x):
        conditioning_states = self.conditioning_stack(x)
        latent_dim = self.latent_stack(x)
        x = self.sampler(conditioning_states, latent_dim)
        return x


def test_generator():
    input_channels = 1
    conv_type = "standard"
    context_channels = 384
    latent_channels = 768
    forecast_steps = 18
    output_shape = 256
    conditioning_stack = ConditioningStack(input_channels=input_channels, conv_type=conv_type, output_channels=context_channels)
    latent_stack = LatentConditioningStack(shape=(8 * input_channels, output_shape // 32, output_shape // 32), output_channels=latent_channels)
    sampler = Sampler(forecast_steps=forecast_steps, latent_channels=latent_channels, context_channels=context_channels)

    model = Generator(conditioning_stack=conditioning_stack, latent_stack=latent_stack, sampler=sampler).cuda()
    x = torch.rand((4, 4, 1, 256, 256)).cuda()
    out = model(x)
    assert out.shape == (4, 18, 1, 256, 256)
    y = torch.rand((4, 18, 1, 256, 256)).cuda()
    loss = F.mse_loss(y, out).cuda()
    loss.backward()
    assert not torch.isnan(out).any()


def main():
    if torch.cuda.is_available():
        print("Device:{}".format(torch.cuda.get_device_properties(0)))
    test_generator()
    print("Generator passed unit test.")


if __name__ == "__main__":
    main()
