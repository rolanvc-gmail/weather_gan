import torch
import torch.nn as nn
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    return v/(v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Spectral Normalization based on Miyato's paper.
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _made_params(self):
        """
        Check if the u, v, and w parameters of the module have been set.
        :return: Bool
        """
        try:
            u = getattr(self.module, self.name+"_u")
            v = getattr(self.module, self.name+"_v")
            w = getattr(self.module, self.name+"_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        """
        Initialize the u,v with random vector sampled from isotropic distribution.
        Then store these as parameters to the  module.
        :return: None
        """
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = Parameter(w.data.new(height).normal_(0,1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0,1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def _update_u_v(self):
        """
        Retrieve, update, and store the module's parameters.
        :return:
        """

        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        height = w.data.shape[0]
        for i in range(self.power_iterations):
            v_data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u_data = l2normalize(torch.mv(w.view(height, -1).data), u.data)
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)










