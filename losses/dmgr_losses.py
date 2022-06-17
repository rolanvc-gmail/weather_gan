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
