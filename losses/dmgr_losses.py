import torch
from torch import nn
from torch.nn import functional as F
from data_modules import RadarDataset
from torch.utils.data import DataLoader
from ConditioningStack import ConditioningStack
from LatentConditioningStack import LatentConditioningStack
from Sampler import Sampler
from Generator import Generator


class NowcastingLoss(nn.Module):
    def __init__(self):
        super(NowcastingLoss, self).__init__()

    def forward(self, x, real_flag):
        if real_flag:
            x = -x
        return F.relu(1.0 + x).mean()


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

        difference /= targets.size(1) * targets.size(3) * targets.size(4)  # 1/HWN
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
    gen_mean = torch.mean(generated_samples, dim=0).cuda()
    weights = torch.clip(batch_targets, 0.0, 24.0).cuda()
    loss = torch.mean(torch.abs(gen_mean - batch_targets) * weights)
    return loss


def loss_hinge_gen(score_generated):
    """
    Generator hinge loss
    """
    loss = -torch.mean(score_generated)
    return loss


def main():
    radar_dataset = RadarDataset()

    train_dataloader = DataLoader(radar_dataset, batch_size=16, shuffle=True)

    input_channels = 1
    conv_type = "standard"
    context_channels = 384
    latent_channels = 768
    forecast_steps = 18
    output_shape = 256
    conditioning_stack = ConditioningStack(input_channels=input_channels, conv_type=conv_type, output_channels=context_channels)
    latent_stack = LatentConditioningStack(shape=(8 * input_channels, output_shape // 32, output_shape // 32), output_channels=latent_channels)
    sampler = Sampler(forecast_steps=forecast_steps, latent_channels=latent_channels, context_channels=context_channels)

    generator = Generator(conditioning_stack=conditioning_stack, latent_stack=latent_stack, sampler=sampler).cuda()

    for b in range(1):
        print("Step # {}".format(b))
        batch_data = next(iter(train_dataloader))
        images_data, target_images = batch_data  # images_data should be 16x4x256x256x1, target_images should be 16x18x256x256x1
        batch_sz = 4
        x = torch.rand((batch_sz, 4, 1, 256, 256)).cuda()
        gen_predictions = generator(x)
        target_images = target_images.cuda()
        grid_cell_reg = grid_cell_regularizer(gen_predictions, target_images)
        print("grid_cell_regularizer: {}".format(grid_cell_reg))


if __name__ == "__main__":
    main()
