import torch
from torch import nn
from losses.dmgr_losses import grid_cell_regularizer
from ConditioningStack import ConditioningStack
from LatentConditioningStack import LatentConditioningStack
from Sampler import Sampler
from Generator import Generator
from SpatialDiscriminator import SpatialDiscriminator
from TemporalDiscriminator import TemporalDiscriminator
import random


class DGMR:
    """
    Deep Generative Model of Radar
    """
    def __init__(self,
                 forecast_steps: int = 18,  # Number of steps to predict in the future
                 input_channels: int = 1,  # Number of input channels per image
                 output_shape: int = 256,  # Shape of the output predictions, generally should be the same as the input shape
                 gen_lr: float = 5e5,  # learning rate for the generator,
                 disc_lr: float = 2e-4,  # learning rate for the discriminator
                 conv_type: str = "standard",  # type of convolution to use
                 num_samples: int = 6,  # Number of samples of the latent space to sample for training/validation
                 grid_lambda: float = 20.0,  # Lambda for the grid regularization loss
                 beta1: float = 0.0,  # Beta1 for adam optimizer
                 beta2: float = 0.999,  # Beta2 for Adam optimizer
                 latent_channels: int = 768,  # Number of channels the latent space should be reshaped to.
                 context_channels: int = 384,
                 **kwargs
                 ):
        super(DGMR, self).__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        input_channels = self.config["input_channels"]
        forecast_steps = self.config["forecast_steps"]
        output_shape = self.config["output_shape"]
        gen_lr = self.config["gen_lr"]
        disc_lr = self.config["disc_lr"]
        conv_type = self.config["conv_type"]
        num_samples = self.config["num_samples"]
        grid_lambda = self.config["grid_lambda"]
        beta1 = self.config["beta1"]
        beta2 = self.config["beta2"]
        latent_channels = self.config["latent_channels"]
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.grid_lambda = grid_lambda
        self.num_samples = num_samples
        self.latent_channels = latent_channels
        self.context_channels = context_channels
        self.input_channels = input_channels
        self.conditioning_stack = ConditioningStack(input_channels=input_channels,
                                                    conv_type=conv_type,
                                                    output_channels=self.context_channels,
                                                    )
        self.latent_stack = LatentConditioningStack(shape=(8 * self.input_channels, output_shape // 32, output_shape // 32),
                                                    output_channels=self.latent_channels)
        self.sampler = Sampler(forecast_steps=forecast_steps,
                               latent_channels=self.latent_channels,
                               contex_channels=self.context_channels)

        self.generator = Generator(self.conditioning_stack, self.latent_stack, self.sampler)
        num_spatial_frames = 8
        self.spatial_discriminator = SpatialDiscriminator(input_channels=input_channels, num_time_steps=num_spatial_frames, conv_type=conv_type)
        self.temporal_discriminator = TemporalDiscriminator(input_channels=input_channels, conv_type=conv_type)

        self.automatic_optimization = False  # Use PyLightning's Manual Optimization.
        torch.autograd.set_detect_anomaly(True)

    def forward(self, x):
        x = self.generator(x)
        return x

    def training_step(self, batch):
        images_data, target_images = batch  # images_data should be 16x4x256x256x1, target_images should be 16x18x256x256x1
        images_data = images_data.float()
        target_images = target_images.float()
        g_opt, sd_opt, td_opt = self.configure_optimizers()

        # Two discriminator steps per generator step
        for _ in range(2):
            # compute spatial discriminator loss
            s_sd = random.sample(range(0, 18), 8)
            predictions = self.generator(images_data)  # predictions should be 16x18x256x256x1
            sd_score_predictions = self.spatial_discriminator(predictions[:, s_sd])  # we only use 8 of 18 images to get sd_score, sd_score should be 16x1x1
            sd_score_target_images = self.spatial_discriminator(target_images[:, s_sd])
            sd_loss = torch.mean(nn.ReLU(1-sd_score_target_images) + nn.ReLU(1+sd_score_predictions))

            # compute temporal discriminator loss
            sequence_whole_real = torch.cat((images_data, target_images), dim=1)
            td_score_whole_real = self.temporal_discriminator(sequence_whole_real)
            sequence_generated = torch.cat((images_data, predictions), dim=1)
            td_score_generated = self.temporal_discriminator(sequence_generated)
            td_loss = torch.mean(nn.ReLU(1-td_score_whole_real) + nn.ReLU(1+td_score_generated))

            # compute discriminator loss
            d_loss = (sd_loss + td_loss)

            sd_opt.zero_grad()
            td_opt.zero_grad()
            d_loss.backward()
            sd_opt.step()
            td_opt.step()

        # Optimize generator
        gen_predictions = self.generator(images_data)
        sd_fake_predictions = self.spatial_discriminator(gen_predictions)
        gen_td_data = torch.cat([images_data, gen_predictions], dim=1)
        td_predictions = self.temporal_discriminator(gen_td_data)

        # R_Loss
        grid_cell_reg = grid_cell_regularizer(torch.stack(gen_predictions, dim=0), target_images)

        g_loss = -(torch.mean(sd_fake_predictions) + torch.mean(td_predictions)) + self.grid_lambda * grid_cell_reg
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

    def configure_optimizers(self):
        b1 = self.beta1
        b2 = self.beta2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr, betas=(b1, b2))
        opt_sd = torch.optim.Adam(self.spatial_discriminator.parameters(), lr=self.disc_lr, betas=(b1, b2))
        opt_td = torch.optim.Adam(self.spatial_discriminator.parameters(), lr=self.disc_lr, betas=(b1, b2))

        return [opt_g, opt_sd, opt_td]
