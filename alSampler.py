import torch
from torch import nn
from al_spectral_norm import AlSpectralNorm
from al_depth_to_space import depth_to_space
from al_ConvGRU import AlConvGRU
import numpy as np
from torch.autograd import Variable
from al_conditioning_stack import AlConditioningStack
from al_latent_conditioning_stack import AlLCStack
# cuda = True if torch.cuda.is_available() else False
cuda = False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

"""
AlOutputStack
"""


class AlOutputStack(nn.Module):
    def __init__(self, ):
        super(AlOutputStack, self).__init__()
        self.BN = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.conv1 = AlSpectralNorm(nn.Conv2d(48, 4, 1))

    def forward(self, x):
        x = self.BN(x)
        x = self.relu(x)
        x = self.conv1(x)
        out = depth_to_space(x, 2)
        return out


class AlSampler(nn.Module):
    """
    This module
    """
    def __init__(self):
        super(AlSampler, self).__init__()
        self.outputStack = AlOutputStack()
        self.ConvGRU = AlConvGRU(x_dim=[768, 384, 192, 96],
                                 h_dim=[384, 192, 96, 48],
                                 kernel_sizes=3,
                                 num_layers=4,
                                 gb_hidden_size=[384, 192, 96, 48])

    def forward(self, LCS_outputs, CD_output):

        for i in range(len(LCS_outputs)):
            if i == 0:
                LCS_outputs_data = LCS_outputs[i]
            else:
                LCS_outputs_data = torch.cat((LCS_outputs_data, LCS_outputs[i]), 1)  # create list of Z from latent conditioning stack

        gru_output = self.ConvGRU(LCS_outputs_data, CD_output)

        for i in range(gru_output.shape[1]):
            out = gru_output[:, i]
            out = self.outputStack(out)
            if i == 0:
                pred = out
            else:
                pred = torch.cat((pred, out), dim=1)
        return pred


def main():
    input_channel = 24
    conditioning_stack = AlConditioningStack(input_channel)
    l_c_stack = AlLCStack()
    convgru = AlConvGRU(x_dim=[768, 384, 192, 96],
                             h_dim=[384, 192, 96, 48],
                             kernel_sizes=3,
                             num_layers=4,
                             gb_hidden_size=[384, 192, 96, 48])

    batch_sz = 4
    sampler = AlSampler()
    z = Variable(Tensor(np.random.normal(0, 1, (batch_sz, 8, 8, 8))))  # latent variable input for latent conditioning stack
    x = torch.rand((batch_sz, 4, 1, 256, 256))
    CD_input = torch.unsqueeze(x, 2)
    LCS_output = l_c_stack(z)
    CD_output = conditioning_stack(CD_input)
    CD_output.reverse()  # to make the largest first
    LCS_output = torch.unsqueeze(LCS_output, 1)
    LCS_outputs = [LCS_output] * 18

    out = sampler(LCS_outputs, CD_output)
    print("sampler shape is {}".format(out.shape))

if __name__ == "__main__":
    main()

