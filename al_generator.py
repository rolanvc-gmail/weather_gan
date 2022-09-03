import torch
from torch import nn
import numpy as np
from al_ConvGRU import AlConvGRU
from torch.autograd import Variable
from al_conditioning_stack import AlConditioningStack
from al_spectral_norm import AlSpectralNorm
from al_depth_to_space import depth_to_space_1 as depth_to_space
from al_latent_conditioning_stack import AlLCStack
from alSampler import AlSampler, AlOutputStack
# cuda = True if torch.cuda.is_available() else False
cuda = False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


"""
Generator
"""


class AlGenerator(nn.Module):
    def __init__(self, input_channel):
        super(AlGenerator, self).__init__()
        self.conditioningStack = AlConditioningStack(input_channel)
        self.LCStack = AlLCStack()
        self.ConvGRU = AlConvGRU(x_dim=[768, 384, 192, 96],
                                 h_dim=[384, 192, 96, 48],
                                 kernel_sizes=3,
                                 num_layers=4,
                                 gb_hidden_size=[384, 192, 96, 48])
        self.sampler = AlSampler()

    def forward(self, CD_input, LCS_input):
        # LCS_input is Tensor(batch_size, 8, 8, 8). this was the sample from the Gaussian Distn
        # CD_input is Tensor(batch_size, 4, 1, 256, 256)
        CD_input = torch.unsqueeze(CD_input, 2)
        #  torch.unsqueeze returns a new tensor with a dimension of size one inserted at the specified position.
        #  CD_input is Tensor(batch_size, 4, 1, 1, 256, 256).
        LCS_output = self.LCStack(LCS_input)
        # LCS_output is Tensor(batch_size, 768, 8, 8)
        CD_output = self.conditioningStack(CD_input)
        # CD_output is [ [batch_sz, 48, 64, 64], [batch_sz, 96, 32,32], [batch_sz, 192, 16, 16], [batch_sz, 384, 8, 8]
        CD_output.reverse()  # to make the largest first
        # CD_output is [[batch_sz, 384, 8, 8],  [batch_sz, 192, 16, 16],[batch_sz, 96, 32,32], [ batch_sz, 48, 64, 64]
        LCS_output = torch.unsqueeze(LCS_output, 1)
        # LCS_output is Tensor(batch_size, 1, 768, 8,8)
        LCS_outputs = [LCS_output] * 18
        output = self.sampler(LCS_outputs, CD_output)
        # output = torch.unsqueeze(output, 2)
        return output



def test_generator():
    batch_sz = 4
    z = Variable(Tensor(np.random.normal(0, 1, (batch_sz, 8, 8, 8))))  # latent variable input for latent conditioning stack
    x = torch.rand((batch_sz, 4, 1, 256, 256))
    generator = AlGenerator(24)
    out = generator(x, z)
    print("genenerator out has shape:{}".format(out.shape))
    # assert out.shape == (batch_sz, 18, 1, 256, 256)


def main():
    test_generator()


if __name__ == "__main__":
    main()
