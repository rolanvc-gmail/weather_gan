import torch
from torch import nn
from torch.nn import Parameter
import numpy as np


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def space_to_depth(tensor, scale_factor):
    num, ch, height, width = tensor.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError('height and width of tensor must be divisible by scale_factor.')
    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor
    tensor = tensor.reshape([num, ch, new_height, scale_factor, new_width, scale_factor])  # divide by 2 the height and width then create four stacks
    tensor = tensor.permute([0, 1, 3, 5, 2, 4])
    tensor = tensor.reshape([num, scale_factor*scale_factor, ch, new_height, new_width])
    return tensor


def depth_to_space(tensor, scale_factor):
    num, ch, height, width = tensor.shape
    if ch % (scale_factor * scale_factor) != 0:
        raise ValueError('channel of tensor must be divisible by (scale_factor * scale_factor).')
    new_ch = ch // (scale_factor * scale_factor)
    new_height = height * scale_factor
    new_width = width * scale_factor
    tensor = tensor.reshape(
        [num, new_ch, scale_factor, scale_factor, height, width])
    tensor = tensor.permute([0, 1, 4, 2, 5, 3])
    tensor = tensor.reshape([num, new_ch, new_height, new_width])
    return tensor


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))
        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


"""
GBlockUp
"""


class GBlockUp(nn.Module):  # for upsampling in GRU
    def __init__(self, in_channels, out_channels):
        super(GBlockUp, self).__init__()
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3_1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = self.conv1(x1)
        x2 = self.BN(x)
        x2 = self.relu(x2)
        x2 = self.upsample(x2)
        x2 = self.conv3_1(x2)
        x2 = self.BN(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        out = x1 + x2
        return out


class GBlock(nn.Module):  # take GRU output then feed to GBlockUp (no upsampling)
    def __init__(self, in_channels, out_channels):
        super(GBlock, self).__init__()
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3_1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.BN(x)
        x2 = self.relu(x2)
        x2 = self.conv3_1(x2)
        x2 = self.BN(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_1(x2)
        out = x1 + x2
        return out


###################### L BLOCK ######################

class LBlock(nn.Module):  ### for latent conditioning
    def __init__(self, in_channels, out_channels):
        super(LBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels - in_channels, 1)
        self.conv3_1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.relu(x)
        x1 = self.conv3_1(x1)
        x1 = self.relu(x1)
        x1 = self.conv3_2(x1)
        x2 = self.conv1(x)
        x3 = x
        x23 = torch.cat([x2, x3], axis=1)
        out = x1 + x23
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        mask = self.sigmoid(x)
        return mask


class LCStack(nn.Module):
    def __init__(self, ):
        super(LCStack, self).__init__()
        self.conv3_1 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.LBlock_1 = LBlock(8, 24)
        self.LBlock_2 = LBlock(24, 48)
        self.LBlock_3 = LBlock(48, 192)
        self.LBlock_4 = LBlock(192, 768)
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


class DBlockDown(nn.Module):  ### for sampler ####
    def __init__(self, in_channels, out_channels):
        super(DBlockDown, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm( nn.Conv2d(in_channels, out_channels, 1))
        self.conv3_1 = SpectralNorm(nn.Conv2d(in_channels, in_channels, 3,stride = 1,padding = 1))
        self.conv3_2 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1))
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


class conditioningStack(nn.Module):
    def __init__(self, in_channels):
        super(conditioningStack, self).__init__()
        self.DBlockDown_1 = DBlockDown(4,24)
        self.DBlockDown_2 = DBlockDown(24,48)
        self.DBlockDown_3 = DBlockDown(48,96)
        self.DBlockDown_4 = DBlockDown(96,192)
        self.relu = nn.ReLU()
        self.conv3_1 = SpectralNorm(nn.Conv2d(96, 48, 3,stride = 1,padding = 1))
        self.conv3_2 = SpectralNorm(nn.Conv2d(192, 96, 3, stride = 1, padding = 1))
        self.conv3_3 = SpectralNorm(nn.Conv2d(384, 192, 3, stride = 1, padding = 1))
        self.conv3_4 = SpectralNorm(nn.Conv2d(768, 384, 3, stride = 1, padding = 1))

    def forward(self, x):
        dataList=[]

        for i in range(x.shape[1]):
            x_new = x[:,i,:,:,:]
            x_new = space_to_depth(x_new,2)
            x_new = np.squeeze(x_new)
            x_new = self.DBlockDown_1(x_new)

            if i == 0:
                data_0 = x_new
            else:
                data_0 = torch.cat((data_0,x_new),1)
                if i ==3:
                    data_0 = self.conv3_1(data_0)
                    dataList.append(data_0)
            x_new = self.DBlockDown_2(x_new)

            if i == 0:
                data1 = x_new
            else:
                data1 = torch.cat((data1,x_new),1)
                if i == 3:
                    data1 = self.conv3_2(data1)
                    dataList.append(data1)
            x_new = self.DBlockDown_3(x_new)

            if i == 0:
                data2 = x_new
            else:
                data2 = torch.cat((data2,x_new),1)
                if i == 3:
                    data2 = self.conv3_3(data2)
                    dataList.append(data2)
            x_new = self.DBlockDown_4(x_new)

            if i == 0:
                data3 = x_new
            else:
                data3 = torch.cat((data3,x_new),1)
                if i == 3:
                    data3 = self.conv3_4(data3)
                    dataList.append(data3)

        return dataList ## should equal around 4 stacks with 4 elements per stack

class SequenceGRU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv = SpectralNorm(nn.Conv2d(input_size, input_size, kernel_size=3, padding=1, stride=1))
        self.GBlock = GBlock(input_size, input_size)
        self.GBlockUp = GBlockUp(input_size, input_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.GBlock(x)
        out = self.GBlockUp(x)
        return out


class ConvGRUCell(nn.Module):  # modified GRU cell from original code
    def __init__(self, x_dim, h_dim, kernel_size, activation=torch.sigmoid):
        super().__init__()
        padding = kernel_size // 2
        self.x_dim = x_dim  # [768, 384, 192, 96],
        self.h_dim = h_dim  # [384, 192, 96, 48]
        self.activation = activation
        self.reset_gate_x = nn.Conv2d(self.x_dim, self.h_dim, kernel_size, padding=padding, stride=1)  # x input reset gate
        self.reset_gate_h = nn.Conv2d(self.h_dim, self.h_dim, kernel_size, padding=padding, stride=1)  # h input reset gate
        self.update_gate_x = nn.Conv2d(self.x_dim, self.h_dim, kernel_size, padding=padding, stride=1)  # x input update gate
        self.update_gate_h = nn.Conv2d(self.h_dim, self.h_dim, kernel_size, padding=padding, stride=1)  # h input update gate
        self.new_gate_x = nn.Conv2d(self.x_dim, self.h_dim, kernel_size, padding=padding, stride=1)  # x input update gate
        self.new_gate_h = nn.Conv2d(self.h_dim, self.h_dim, kernel_size, padding=padding, stride=1)  # h input update gate

        self.sqrt_k = np.sqrt(1 / self.h_dim)

        nn.init.uniform_(self.reset_gate_x.weight, - self.sqrt_k, + self.sqrt_k)
        nn.init.uniform_(self.reset_gate_h.weight, - self.sqrt_k, + self.sqrt_k)
        nn.init.uniform_(self.update_gate_x.weight, - self.sqrt_k, + self.sqrt_k)
        nn.init.uniform_(self.update_gate_h.weight, - self.sqrt_k, + self.sqrt_k)
        nn.init.uniform_(self.new_gate_x.weight, - self.sqrt_k, + self.sqrt_k)
        nn.init.uniform_(self.new_gate_h.weight, - self.sqrt_k, + self.sqrt_k)
        nn.init.constant_(self.reset_gate_x.bias, 0.)
        nn.init.constant_(self.reset_gate_h.bias, 0.)
        nn.init.constant_(self.update_gate_x.bias, 0.)
        nn.init.constant_(self.update_gate_h.bias, 0.)
        nn.init.constant_(self.new_gate_x.bias, 0.)
        nn.init.constant_(self.new_gate_h.bias, 0.)

        self.reset_gate_x = SpectralNorm(self.reset_gate_x)
        self.reset_gate_h = SpectralNorm(self.reset_gate_h)
        self.update_gate_x = SpectralNorm(self.update_gate_x)
        self.update_gate_h = SpectralNorm(self.update_gate_h)
        self.new_gate_x = SpectralNorm(self.new_gate_x)
        self.new_gate_h = SpectralNorm(self.new_gate_h)

    def forward(self, x, prev_state=None):  # prev_state: bs x 768 x 8 x 8; x : bs x 384 x 8 x 8
        if prev_state is None:
            batch_size = x.data.size()[0]  # number of samples
            spatial_size = x.data.size()[2:]  # width x height --> 8 x 8, 16 x 16, 32 x 32, 64 x 64
            state_size = [batch_size, self.cs_dim] + list(spatial_size)  # [batch size, hidden size, height, width]
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).cuda()
            else:
                prev_state = torch.zeros(state_size).cuda()

        r_t = self.activation(self.reset_gate_x(x) + self.reset_gate_h(prev_state))
        z_t = self.activation(self.update_gate_x(x) + self.update_gate_h(prev_state))
        n_t = torch.tanh(self.new_gate_x(x) + (r_t * self.new_gate_h(prev_state)))
        h_t = ((1 - z_t) * n_t) + (z_t * prev_state)
        return h_t


class ConvGRU(nn.Module):
    def __init__(self, x_dim, h_dim, kernel_sizes, num_layers, gb_hidden_size):  # ls_dim is [768, 384, 192, 96]; cs_dim is [384, 192, 96, 48]
        super().__init__()

        if type(x_dim) != list:
            self.x_dim = [x_dim] * num_layers
        else:
            assert len(x_dim) == num_layers
            self.x_dim = x_dim

        if type(h_dim) != list:
            self.h_dim = [h_dim] * num_layers
        else:
            assert len(h_dim) == num_layers
            self.h_dim = h_dim

        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes] * num_layers
        else:
            assert len(kernel_sizes) == num_layers
            self.kernel_sizes = kernel_sizes

        self.n_layers = num_layers  # 4 layers
        cells = nn.ModuleList()
        squenceCells = nn.ModuleList()

        for i in range(self.n_layers):
            cell = ConvGRUCell(self.x_dim[i], self.h_dim[i], 3)
            cells.append(cell)
        self.cells = cells

        for i in range(self.n_layers):
            squenceCell = SequenceGRU(gb_hidden_size[i])
            squenceCells.append(squenceCell)
        self.squenceCells = squenceCells

    def forward(self, x, h):  # x is from latent conditioning stack, h is from conditioning stack
        seq_len = x.size(1)  # 18
        prev_state_list = []

        for t in range(seq_len):
            if t == 0:
                curr_state_list = []
                for layer_idx in range(self.n_layers):
                    cell = self.cells[layer_idx]
                    h_input = h[layer_idx]  # hidden is from conditioning stack: bs x 384 x 8 x 8; bs x 192 x 16 x 16; bs x 96 x 32 x 32; bs x 48 x 64 x 64
                    squenceCell = self.squenceCells[layer_idx]

                    if layer_idx == 0:
                        x_input = x[:, t, :, :, :]  # a is from latent conditioning stack: bs x 1 x 768 x 8 x 8
                        new_state = cell(x_input, h_input)
                        upd_new_state = squenceCell(new_state)
                        curr_state_list.append(new_state)
                    else:
                        x_input = upd_new_state  # get lower upscaled hidden state
                        new_state = cell(x_input, h_input)
                        upd_new_state = squenceCell(new_state)
                        curr_state_list.append(new_state)

                upd_new_state = torch.unsqueeze(upd_new_state, dim=1)
                output = upd_new_state  # get upper output at t = 0
                prev_state_list = curr_state_list

            else:
                curr_state_list = []
                for layer_idx in range(self.n_layers):
                    cell = self.cells[layer_idx]
                    h_input = prev_state_list[layer_idx]
                    squenceCell = self.squenceCells[layer_idx]

                    if layer_idx == 0:
                        x_input = x[:, t, :, :, :]  # a is from latent conditioning stack: bs x 1 x 768 x 8 x 8
                        new_state = cell(x_input, h_input)
                        upd_new_state = squenceCell(new_state)
                        curr_state_list.append(new_state)
                    else:
                        x_input = upd_new_state  # get lower upscaled hidden state
                        new_state = cell(x_input, h_input)
                        upd_new_state = squenceCell(new_state)
                        curr_state_list.append(new_state)

                upd_new_state = torch.unsqueeze(upd_new_state, dim=1)
                output = torch.cat((output, upd_new_state), dim=1)  # get upper output
                prev_state_list = curr_state_list
        return output


###################### OUTPUT STACK ######################
class outputStack(nn.Module):
    def __init__(self, ):
        super(outputStack, self).__init__()
        self.BN = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv2d(48, 4, 1))

    def forward(self, x):
        x = self.BN(x)
        x = self.relu(x)
        x = self.conv1(x)
        out = depth_to_space(x, 2)
        return out


###################### GENERATOR ######################
class AlGenerator(nn.Module):
    def __init__(self, input_channel):
        super(AlGenerator, self).__init__()
        self.conditioningStack = conditioningStack(input_channel)
        self.LCStack = LCStack()
        self.ConvGRU = ConvGRU(x_dim=[768, 384, 192, 96],
                               h_dim=[384, 192, 96, 48],
                               kernel_sizes=3,
                               num_layers=4,
                               gb_hidden_size=[384, 192, 96, 48])
        self.outputStack = outputStack()

    def forward(self, CD_input, LCS_input):
        CD_input = torch.unsqueeze(CD_input, 2)
        LCS_output = self.LCStack(LCS_input)
        CD_output = self.conditioningStack(CD_input)
        CD_output.reverse()  # to make the largest first
        LCS_output = torch.unsqueeze(LCS_output, 1)
        LCS_outputs = [LCS_output] * 18

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

