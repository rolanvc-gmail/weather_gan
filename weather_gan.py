import torch
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
from torch.nn import init
cuda = True if torch.cuda.is_available() else False
#cuda = False
import numpy as np
import os
import random


os.environ["CUDA_VISIBLE_DEVICES"] = '2'


###################### UTILITIES ######################

def space_to_depth(tensor, scale_factor):
    num, ch, height, width = tensor.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError('height and width of tensor must be divisible by scale_factor.')
    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor
    tensor = tensor.reshape([num, ch, new_height, scale_factor, new_width, scale_factor]) # divide by 2 the height and width then create four stacks
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

def Norm_1_numpy(y):
    sum1 = 0
    for i in range(y.shape[0]):
      sum1 = sum1+np.linalg.norm(y[i], ord=1, keepdims=True)
    return sum1/y.shape[0]
    
def Norm_1_torch(y):
    sum = 0
    for i in range(y.shape[0]):
      sum = sum+torch.max(torch.norm(y[i],  p=1, dim=0))
    return sum/y.shape[0]
    
def Norm_1(y):
    sum = 0
    for i in range(y.shape[0]):
      sum = sum+np.linalg.norm(y[i], ord=1, keepdims=True)
    return sum/y.shape[0]

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

###################### SPECTRAL NORM ###################### 

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
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))
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

###################### D BLOCK ###################### 

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

class DBlockDownFirst(nn.Module):  ### for sampler #### (only difference is Relu in forward ?)
    def __init__(self, in_channels, out_channels):
        super(DBlockDownFirst, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 1))
        self.conv3_1 = SpectralNorm(nn.Conv2d(in_channels, in_channels, 3,stride = 1,padding = 1))
        self.conv3_2 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1))
        self.maxpool = nn.MaxPool2d(2, stride = 2, return_indices = False, ceil_mode = False)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool(x1)
        x2 = self.conv3_1(x)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool(x2)
        out = x1 + x2
        return out

class DBlock(nn.Module): ### for both spatial and temporal discriminators
    def __init__(self, in_channels, out_channels):
        super(DBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 1))
        self.conv3 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 3,stride = 1,padding = 1))
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.relu(x)
        x2 = self.conv3(x2)
        x2 = self.relu(x2)
        x2 = self.conv3(x2)
        out = x1 + x2
        return out
        
class DBlock3D_1(nn.Module): ### for temporal discriminators
    def __init__(self, in_channels, out_channels):
        super(DBlock3D_1, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size = (1, 1, 1)))
        self.conv3_1 = SpectralNorm(nn.Conv3d(in_channels, in_channels, kernel_size = (3, 3, 3), padding = 1,stride = 1))
        self.conv3_2 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size = (3, 3, 3), padding = 1, stride = 1))
        self.maxpool_3d = nn.MaxPool3d(kernel_size = (2, 2, 2), stride = (2, 2, 2))
        
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool_3d(x1)
        x2 = self.conv3_1(x)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool_3d(x2)
        out = x1 + x2
        return out
        
class DBlock3D_2(nn.Module): ### for temporal discriminators ### (only difference is Relu in forward ?)
    def __init__(self, in_channels, out_channels):
        super(DBlock3D_2, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size = (1, 1, 1)))
        self.conv3_1 = SpectralNorm(nn.Conv3d(in_channels, in_channels, kernel_size = (3, 3, 3), padding = 1,stride = 1))
        self.conv3_2 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size = (3, 3, 3), padding = 1, stride = 1))
        self.maxpool_3d = nn.MaxPool3d(kernel_size = (2, 2, 2), stride = (2, 2, 2))

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

###################### L BLOCK ###################### 

class LBlock(nn.Module):  ### for latent conditioning 
    def __init__(self, in_channels, out_channels):
        super(LBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels,out_channels-in_channels, 1)
        self.conv3_1 = nn.Conv2d(in_channels, in_channels, 3, stride = 1, padding = 1)
        self.conv3_2 = nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1)

    def forward(self, x):
        x1 = self.relu(x)
        x1 = self.conv3_1(x1)
        x1 = self.relu(x1)
        x1 = self.conv3_2(x1)
        x2 = self.conv1(x)
        x3 = x
        x23 = torch.cat([x2,x3],axis = 1)
        out = x1 + x23
        return out
        
###################### G BLOCK ######################

class GBlockUp(nn.Module):  ### for upsampling in GRU
    def __init__(self, in_channels, out_channels):
        super(GBlockUp, self).__init__()
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3_1 = nn.Conv2d(in_channels, in_channels, 3, stride = 1, padding = 1)
        self.conv3_2 = nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1)

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

class GBlock(nn.Module): ### take GRU output then feed to GBlockUp (no upsampling)
    def __init__(self, in_channels, out_channels):
        super(GBlock, self).__init__()
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3_1 = nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1)

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
        
###################### LATENT CONDITIONING STACK ######################

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 3, stride=1,padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        mask=self.sigmoid(x)
        return mask
        
class LCStack(nn.Module):
    def __init__(self,):
        super(LCStack, self).__init__()
        self.conv3_1 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.LBlock_1 = LBlock(8, 24)
        self.LBlock_2 = LBlock(24, 48)
        self.LBlock_3 = LBlock(48, 192)
        self.LBlock_4 = LBlock(192, 768)
        self.mask = SpatialAttention()
        
    def forward(self,x):
        x = self.conv3_1(x)
        x = self.LBlock_1(x)
        x = self.LBlock_2(x)
        x = self.LBlock_3(x)
        mask = self.mask(x)
        x = x * mask
        out = self.LBlock_4(x)
        return out
        
###################### SPATIAL DISCRIMINATOR ######################

class spaDiscriminator(nn.Module):
    def __init__(self,):
        super(spaDiscriminator, self).__init__()
        self.DBlockDown_1 = DBlockDownFirst(4, 48)
        self.DBlockDown_2 = DBlockDown(48, 96)
        self.DBlockDown_3 = DBlockDown(96, 192)
        self.DBlockDown_4 = DBlockDown(192, 384)
        self.DBlockDown_5 = DBlockDown(384, 768)
        self.DBlock_6 = DBlock(768, 768)
        self.sum_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=2) # I ADDED THIS
        self.linear =  SpectralNorm(nn.Linear(in_features = 768*1*1, out_features = 1))

    def forward(self,x):
        x = self.avgPool(x) # used avg pool instead of random crop sampling
        
        for i in range(x.shape[1]):
          x_temp = x[:,i]
          x_temp = x_temp.view(x_temp.shape[0],1,x_temp.shape[1],x_temp.shape[2])
          x_temp = space_to_depth(x_temp,2)
          x_temp = torch.squeeze(x_temp)
          x_temp = self.DBlockDown_1(x_temp)
          x_temp = self.DBlockDown_2(x_temp)
          x_temp = self.DBlockDown_3(x_temp)
          x_temp = self.DBlockDown_4(x_temp)
          x_temp = self.DBlockDown_5(x_temp)
          x_temp = self.DBlock_6(x_temp)
          x_temp = self.sum_pool(x_temp)
          x_temp = x_temp.view(x_temp.shape[0],x_temp.shape[1])
          x_temp = x_temp * 4
          out = self.linear(x_temp)
          
          if i == 0:
           data = out
          else:
           data = data + out
        
        data = torch.squeeze((data))
        return data
        
###################### TEMPORAL DISCRIMINATOR ######################

class temDiscriminator(nn.Module):
    def __init__(self,):
        super(temDiscriminator, self).__init__()
        self.DBlock3D_1 =DBlock3D_1(4, 48)
        self.DBlock3D_2 = DBlock3D_2(48, 96)
        self.DBlockDown_3 = DBlockDown(96, 192)
        self.DBlockDown_4 = DBlockDown(192, 384)
        self.DBlockDown_5 = DBlockDown(384, 768)
        self.DBlock_6 = DBlock(768, 768)
        self.sum_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(in_features=768 * 1 * 1, out_features=1)

    def forward(self,x):
        T_H = random.sample(range(0, 128), 1) # I ADDED THIS
        T_W = random.sample(range(0, 128), 1) # I ADDED THIS
        x = x[:, :, T_H[0]:T_H[0] + 128, T_W[0]:T_W[0] + 128] # I ADDED THIS
        x = space_to_depth(x,2)
        x = self.DBlock3D_1(x)
        x = self.DBlock3D_2(x)
        
        for i in range(0,x.shape[2]):
          x_temp = x[:,:,i,:,:]
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
        
###################### CONDITIONING STACK ######################

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

###################### CONV GRU ######################     
    
class SequenceGRU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv = SpectralNorm(nn.Conv2d(input_size, input_size, kernel_size = 3, padding = 1, stride = 1))
        self.GBlock = GBlock(input_size, input_size)
        self.GBlockUp = GBlockUp(input_size, input_size)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.GBlock(x)
        out = self.GBlockUp(x)
        return out

class ConvGRUCell(nn.Module): # modified GRU cell from original code
    def __init__(self, x_dim, h_dim, kernel_size, activation = torch.sigmoid):
        super().__init__()
        padding = kernel_size//2
        self.x_dim = x_dim # [768, 384, 192, 96],
        self.h_dim = h_dim # [384, 192, 96, 48]
        self.activation = activation
        self.reset_gate_x = nn.Conv2d(self.x_dim, self.h_dim, kernel_size, padding = padding, stride = 1) # x input reset gate
        self.reset_gate_h = nn.Conv2d(self.h_dim, self.h_dim, kernel_size, padding = padding, stride = 1) # h input reset gate
        self.update_gate_x = nn.Conv2d(self.x_dim, self.h_dim, kernel_size, padding = padding, stride = 1) # x input update gate
        self.update_gate_h = nn.Conv2d(self.h_dim, self.h_dim, kernel_size, padding = padding, stride = 1) # h input update gate
        self.new_gate_x = nn.Conv2d(self.x_dim, self.h_dim, kernel_size, padding = padding, stride = 1) # x input update gate
        self.new_gate_h = nn.Conv2d(self.h_dim, self.h_dim, kernel_size, padding = padding, stride = 1) # h input update gate
        
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

    def forward(self, x, prev_state = None): # prev_state: bs x 768 x 8 x 8; x : bs x 384 x 8 x 8
        if prev_state is None: 
            batch_size = x.data.size()[0] # number of samples
            spatial_size = x.data.size()[2:] # width x height --> 8 x 8, 16 x 16, 32 x 32, 64 x 64
            state_size = [batch_size, self.cs_dim] + list(spatial_size) # [batch size, hidden size, height, width]
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
    def __init__(self, x_dim, h_dim, kernel_sizes, num_layers, gb_hidden_size): # ls_dim is [768, 384, 192, 96]; cs_dim is [384, 192, 96, 48]
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

        self.n_layers = num_layers # 4 layers
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


    def forward(self, x, h): # x is from latent conditioning stack, h is from conditioning stack
        seq_len = x.size(1) # 18
        prev_state_list = []
        
        for t in range(seq_len):
            if t == 0:
                curr_state_list = []
                for layer_idx in range(self.n_layers):
                    cell = self.cells[layer_idx]
                    h_input = h[layer_idx] # hidden is from conditioning stack: bs x 384 x 8 x 8; bs x 192 x 16 x 16; bs x 96 x 32 x 32; bs x 48 x 64 x 64
                    squenceCell = self.squenceCells[layer_idx]
                    
                    if layer_idx == 0:
                        x_input = x[:, t, :, :, :] # a is from latent conditioning stack: bs x 1 x 768 x 8 x 8
                        new_state = cell(x_input, h_input)
                        upd_new_state = squenceCell(new_state)
                        curr_state_list.append(new_state)
                    else:
                        x_input = upd_new_state # get lower upscaled hidden state
                        new_state = cell(x_input, h_input)
                        upd_new_state = squenceCell(new_state)
                        curr_state_list.append(new_state)

                upd_new_state = torch.unsqueeze(upd_new_state, dim = 1)
                output = upd_new_state # get upper output at t = 0
                prev_state_list = curr_state_list
            
            else:
                curr_state_list = []
                for layer_idx in range(self.n_layers):
                    cell = self.cells[layer_idx]
                    h_input = prev_state_list[layer_idx]
                    squenceCell = self.squenceCells[layer_idx]
                    
                    if layer_idx == 0:
                        x_input = x[:, t, :, :, :] # a is from latent conditioning stack: bs x 1 x 768 x 8 x 8
                        new_state = cell(x_input, h_input)
                        upd_new_state = squenceCell(new_state)
                        curr_state_list.append(new_state)
                    else:
                        x_input = upd_new_state # get lower upscaled hidden state
                        new_state = cell(x_input, h_input)
                        upd_new_state = squenceCell(new_state)
                        curr_state_list.append(new_state)
                    
                upd_new_state = torch.unsqueeze(upd_new_state, dim = 1)
                output = torch.cat((output, upd_new_state), dim = 1) # get upper output
                prev_state_list = curr_state_list
        return output

###################### OUTPUT STACK ######################
        
class outputStack(nn.Module):
    def __init__(self,):
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

class generator(nn.Module):
    def __init__(self, input_channel):
        super(generator, self).__init__()
        self.conditioningStack = conditioningStack(input_channel)
        self.LCStack = LCStack()
        self.ConvGRU = ConvGRU(x_dim = [768, 384, 192, 96],
                               h_dim = [384, 192, 96, 48],
                               kernel_sizes = 3,
                               num_layers = 4,
                               gb_hidden_size = [384, 192, 96, 48])
        self.outputStack = outputStack()

    def forward(self, CD_input, LCS_input):
        CD_input = torch.unsqueeze(CD_input, 2)
        LCS_output = self.LCStack(LCS_input)
        CD_output = self.conditioningStack(CD_input)
        CD_output.reverse() # to make the largest first
        LCS_output = torch.unsqueeze(LCS_output, 1)
        LCS_outputs = [LCS_output] * 18

        for i in range(len(LCS_outputs)):
            if i == 0:
               LCS_outputs_data = LCS_outputs[i]
            else:
               LCS_outputs_data = torch.cat((LCS_outputs_data , LCS_outputs[i]), 1) # create list of Z from latent conditioning stack

        gru_output = self.ConvGRU(LCS_outputs_data, CD_output)
        
        for i in range(gru_output.shape[1]):
            out = gru_output[:,i]
            out = self.outputStack(out)
            if i == 0:
                pred = out
            else:
                pred = torch.cat((pred,out), dim = 1)  
        return pred
        
        
###############################################################################################
###############################################################################################
###############################################################################################


import cv2
import glob as glob
import matplotlib.pyplot as plt
import matplotlib
import random

BATCHSIZE = 16
REP = 4
M = 4
N = 22
H = 256
W = 256
Lambda = 20
num_epoch = 5
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
RELU = nn.ReLU()

vmin = -30
vmax = 75
norm = plt.Normalize(vmin, vmax)
cmap = matplotlib.cm.get_cmap('jet')
sm = matplotlib.cm.ScalarMappable(norm = norm, cmap = cmap)
sm.set_array([])

train_mos = ['01', '03', '05', '07', '08', '09']
test_mos = ['02', '04', '06']

def create_dummy_real_sequence():
    data = []
    for i in range(BATCHSIZE):
        img_set = []
        mos = random.choice(train_mos)
        root = '/home/rolan/Weather-Datasets/npy-data/' + str(mos) + '/'
        dirs = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
        day = random.choice(dirs)
        d = '/home/rolan/Weather-Datasets/npy-data/' + str(mos) + '/' + str(day) + '/'
        files = glob.glob(d + '*.npy')
        start = np.random.randint(0, len(files) - 22)
        for j in range(22):
            im = np.load(files[start + j])
            im = im[0]
            pic = np.zeros((256, 256)) + im
            pic = np.expand_dims(pic, axis = 2)
            img_set.append(pic)
        data.append(img_set)       
    return np.array(data)[:,:,:,:,0].astype(np.float32)
    
def create_dummy_real_sequence_for_gen():
    data = []
    for i in range(int(BATCHSIZE / REP)):
        img_set = []
        mos = random.choice(train_mos)
        root = '/home/rolan/Weather-Datasets/npy-data/' + str(mos) + '/'
        dirs = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
        day = random.choice(dirs)
        d = '/home/rolan/Weather-Datasets/npy-data/' + str(mos) + '/' + str(day) + '/'
        files = glob.glob(d + '*.npy')
        start = np.random.randint(0, len(files) - 22)
        for j in range(22):
            im = np.load(files[start + j])
            im = im[0]
            pic = np.zeros((256, 256)) + im
            pic = np.expand_dims(pic, axis = 2)
            img_set.append(pic)
        for k in range(REP):
            data.append(img_set)       
    return np.array(data)[:,:,:,:,0].astype(np.float32)
    
###############################################################################################
###############################################################################################
###############################################################################################

if __name__ == "__main__":

    sd = spaDiscriminator()
    td = temDiscriminator()
    g = generator(24)
    sig = nn.Sigmoid()
    
    if torch.cuda.is_available():
        sd = sd.cuda()
        td = td.cuda()
        g = g.cuda()
    
    if os.path.exists('./sd_.dict') and os.path.exists('./td_.dict') and os.path.exists('./g_.dict'):
        print("loading saved model")
        sd.load_state_dict(torch.load('./sd_.dict'))
        td.load_state_dict(torch.load('./td_.dict'))
        g.load_state_dict(torch.load('./g_.dict'))
        print("saved model loaded")
        print("")
    
    sd_optimizer = torch.optim.Adam(sd.parameters(), betas=(0.0, 0.999), lr=0.0002)
    td_optimizer = torch.optim.Adam(td.parameters(), betas=(0.0, 0.999), lr=0.0002)
    g_optimizer = torch.optim.Adam(g.parameters(), betas=(0.0, 0.999), lr=0.00005)
    #sd_optimizer = torch.optim.RMSprop(sd.parameters(), lr=0.000001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    #td_optimizer = torch.optim.RMSprop(td.parameters(), lr=0.000001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    #g_optimizer = torch.optim.RMSprop(g.parameters(), lr=0.000001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    real_label = Variable(torch.ones(BATCHSIZE)).cuda()
    fake_label = Variable(torch.zeros(BATCHSIZE)).cuda()
    
    for e in range(5000000):
        print("iteration " + str(e))
        
        for j in range(2):
            S = random.sample(range(0, 18), 8) # spatial discriminator picks uniformly at random 8 out of 18 lead times
            #S.sort()
            
            ##### train discriminators on real inputs
            data = create_dummy_real_sequence() ### FETCH DATA HERE
            input_real_1sthalf = Variable(torch.from_numpy(data[:,:4])).cuda() # 2 x 4 x 256 x 256 x 1 (for generator)
            input_real_2ndhalf = Variable(torch.from_numpy(data[:,4:])).cuda() # 2 x 18 x 256 x 256 x 1 (for spatial discriminator)
            input_real_whole = Variable(torch.from_numpy(data)).cuda() # 2 x 22 x 256 x 256 x 1 (for temporal discriminator)
            input_real_2ndhalf_sd = input_real_2ndhalf[:,S]       
            sd_pred_real = sd(input_real_2ndhalf_sd) # output of spatial discriminator for real lead times x 18
            td_pred_real = td(input_real_whole) # output of temporal discriminator for entire sequence x 22

            ##### train discriminators on fake inputs
            z = Variable(Tensor(np.random.normal(0, 1, (BATCHSIZE ,8 ,8 ,8)))) # latent variable input for latent conditioning stack
            fake_img = g(input_real_1sthalf, z).detach() # fake output of generator x 18
            print("shape of generator output is: {}".format(fake_img.shape))
            fake_img_2ndhalf_sd = fake_img[:,S] # get input to spatial discriminator from fake images
            sd_pred_fake = sd(fake_img_2ndhalf_sd)
            fake_img_whole_td = torch.cat((input_real_1sthalf, fake_img), dim = 1) # create input to temporal discriminator from fake images
            td_pred_fake = td(fake_img_whole_td)
            
            sd_loss = torch.mean(RELU(1 - (sd_pred_real)) + RELU(1 + (sd_pred_fake)))
            td_loss = torch.mean(RELU(1 - (td_pred_real)) + RELU(1 + (td_pred_fake)))
            #sd_loss_real = torch.log(sig(sd_pred_real) + 1e-5)
            #td_loss_real = torch.log(sig(td_pred_real) + 1e-5)
            #sd_loss_fake = torch.log(1-sig(sd_pred_fake) + 1e-5)
            #td_loss_fake = torch.log(1-sig(td_pred_fake) + 1e-5)
            #sd_loss = -torch.mean(sd_loss_real + sd_loss_fake)
            #td_loss = -torch.mean(td_loss_real + td_loss_fake)
            #sd_loss = torch.mean(sig(sd_pred_real)) - torch.mean(sig(sd_pred_fake))
            #td_loss = torch.mean(sig(td_pred_real)) - torch.mean(sig(td_pred_fake))
            
            d_loss = (sd_loss + td_loss) 
            sd_optimizer.zero_grad() 
            td_optimizer.zero_grad() 
            d_loss.backward() 
            sd_optimizer.step()
            td_optimizer.step()
            
            #with torch.no_grad():
            #    for v in sd.parameters():
            #        v[:] = v.clip(-0.01, +0.01)
            #    for v in td.parameters():
            #        v[:] = v.clip(-0.01, +0.01)
                
        print("discriminator loss: " + str(np.mean(d_loss.detach().cpu().numpy())))
        
        ##### train generator
        data = create_dummy_real_sequence_for_gen()
        
        input_real_1sthalf = Variable(torch.from_numpy(data[:,:4])).cuda() # 2 x 4 x 256 x 256 x 1 (for generator)
        input_real_2ndhalf = Variable(torch.from_numpy(data[:,4:])).cuda() # 2 x 18 x 256 x 256 x 1 (for generator true values)
        z = Variable(Tensor(np.random.normal(0, 1, (BATCHSIZE ,8 ,8 ,8))))
        g_output = g(input_real_1sthalf, z) # input is real stack of 4 images
 
        fake_img_2ndhalf_sd_g = g_output[:,S]
        sd_pred_fake_g = sd(fake_img_2ndhalf_sd_g) # get prediction of spatial discriminator
        
        fake_img_whole_td_g = torch.cat((input_real_1sthalf, g_output), dim = 1) 
        td_pred_fake_g = td(fake_img_whole_td_g) # get prediction of temporal discriminator
        
        r_loss_sum = 0 # compute r loss for generator
        for i in range(BATCHSIZE):
            result = torch.mul((g_output[i] - input_real_2ndhalf[i]), input_real_2ndhalf)
            r_loss = (1 / H * W * N) * Lambda * Norm_1_torch(result)
            r_loss_sum = r_loss_sum + r_loss

        #g_loss_sum = - (torch.mean(sig(sd_pred_fake_g) + sig(td_pred_fake_g)))
        #g_loss_sum = - (torch.mean(sig(sd_pred_fake_g) + sig(td_pred_fake_g)))
        g_loss_sum = -(torch.mean((sd_pred_fake_g)) + torch.mean((td_pred_fake_g))) + (r_loss_sum / BATCHSIZE)
        
        if e % 25 == 0:
            pred = g_output.detach().cpu().numpy()[0]
            
            for i in range(data[0].shape[0]):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                cbar = fig.colorbar(sm)
                cbar.ax.set_title("dbz")
                ax.imshow(data[0][i], cmap = 'jet', vmin = vmin, vmax = vmax)
                plt.savefig('actual_'+ str(i) + '.png')
            
            for i in range(pred.shape[0]):
                gg = pred[i]
                #gg = gg.reshape(256,256,1) * 255
                #gg = gg.astype(np.uint8)
                #cv2.imwrite(str(i)+'_output.jpg',gg)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                cbar = fig.colorbar(sm)
                cbar.ax.set_title("dbz")
                ax.imshow(gg, cmap = 'jet', vmin = vmin, vmax = vmax)
                plt.savefig('fake_'+ str(i) + '.png')

        g_optimizer.zero_grad()
        g_loss_sum.backward()
        g_optimizer.step()
        
        if e % 1000 == 0:
            print("saving model for iteration: " + str(e))
            torch.save(sd.state_dict(), './sd_.dict')
            torch.save(td.state_dict(), './td_.dict')
            torch.save(g.state_dict(), './g_.dict')
            print("model saved for iteration: " + str(e))
            print("")
        
        #with torch.no_grad():
        #    for v in g.parameters():
        #        v[:] = v.clamp(-0.01, +0.01)
        
        print("generator loss: " + str(np.mean(g_loss_sum.detach().cpu().numpy())))
        pred_sd = sig(sd_pred_fake)
        pred_td = sig(td_pred_fake)
        
        print("sd sample prediction: " + str(pred_sd.detach().cpu().numpy()))
        print("td sample prediction: " + str(pred_td.detach().cpu().numpy()))
        print("")