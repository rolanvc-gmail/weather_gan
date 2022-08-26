import torch
from torch import nn
import numpy as np
from al_spectral_norm import AlSpectralNorm
from al_gblock_up import AlGBlockUp, AlGBlock


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

        self.reset_gate_x = AlSpectralNorm(self.reset_gate_x)
        self.reset_gate_h = AlSpectralNorm(self.reset_gate_h)
        self.update_gate_x = AlSpectralNorm(self.update_gate_x)
        self.update_gate_h = AlSpectralNorm(self.update_gate_h)
        self.new_gate_x = AlSpectralNorm(self.new_gate_x)
        self.new_gate_h = AlSpectralNorm(self.new_gate_h)

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


class SequenceGRU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv = AlSpectralNorm(nn.Conv2d(input_size, input_size, kernel_size=3, padding=1, stride=1))
        self.GBlock = AlGBlock(input_size, input_size)
        self.GBlockUp = AlGBlockUp(input_size, input_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.GBlock(x)
        out = self.GBlockUp(x)
        return out


class AlConvGRU(nn.Module):
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


def test_convgru():
    model = AlConvGRU(x_dim=[768, 384, 192, 96],
                      h_dim=[384, 192, 96, 48],
                      kernel_sizes=3,
                      num_layers=4,
                      gb_hidden_size=[384, 192, 96, 48])
    init_states = [torch.rand((2, 384, 32, 32)) for _ in range(4)]
    # Expand latent dim to match batch size
    x = torch.rand((2, 768, 32, 32))
    hidden_states = [x] * 18
    out = model(hidden_states, init_states[3])
    y = torch.rand((18, 2, 384, 32, 32))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert out.size() == (18, 2, 384, 32, 32)
    assert not torch.isnan(out).any(), "Output included NaNs"


def main():
    test_convgru()
    print("ConvGRU passed unit test")


if __name__ == "__main__":
    main()
