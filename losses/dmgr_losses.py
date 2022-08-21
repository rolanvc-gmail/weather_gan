import torch
from torch import nn
from torch.nn import functional as F


class NowcastingLoss(nn.Module):
    def __init__(self):
        super(NowcastingLoss, self).__init__()

    def forward(self, x, real_flag):
        if real_flag:
            x = -x
        return F.relu(1.0+x).mean()


class GridCellLoss(nn.Module):
    """
    Grid Cell Regularizer
    """
    def __init__(self, weight_fn=None):
        self.weight_fn = weight_fn  # In Paper weight_fn is max(y+1,24)

    def forward(self, generated_images, targets):
        """

        :param generated_images: mean generated images from the generator
        :param targets: ground truth future frames
        :return: Grid Cell Regularizer term

        """
        difference = generated_images - targets
        if self.weight_fn is not None:
            difference *= self.weight_fn(targets)

        difference /= targets.size(1)*targets.size(3) * targets.size(4)  # 1/HWN
        return difference.mean()


def loss_hinge_disc(score_generated, score_real):
    """ Discriminator hinge loss."""
    l1 = F.relu(1.0 - score_real)
    loss = torch.mean(l1)
    l2 = F.relu(1.0 + score_generated)
    loss += torch.mean(l2)
    return loss


def grid_cell_regularizer(generated_samples, batch_targets):
    """
    Grid cell regularizer
    :param generated_samples: Tensor of size [n_samples, batch_size, 18, 256,256, 1].
    :param batch_targets: Tensor of size [batch_size, 18, 256, 256, 1].
    :return loss: a tensor of shape [batch_size].
    """
    gen_mean = torch.mean(generated_samples, dim=0)
    weights = torch.clip(batch_targets, 0.0, 24.0)
    loss = torch.mean(torch.abs(gen_mean - batch_targets) * weights)
    return loss


def loss_hinge_gen(score_generated):
    """
    Generator hinge loss
    """
    loss = -torch.mean(score_generated)
    return loss