import torch
from torch import nn
from torch.nn import functional as F
from losses.dmgr_losses import NowcastingLoss, GridCellLoss
from ConditioningStack import ConditioningStack
from LatentConditioningStack import LatentConditioningStack
from Sampler import Sampler
from Generator import Generator
from Discriminator import Discriminator


class DGMR(nn.Module):
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
        self.discriminator_loss = NowcastingLoss()
        self.grid_regularizer = GridCellLoss()
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
        self.discriminator = Discriminator(input_channels)
        self.save_hyperparameters()

        self.automatic_optimization = False
        torch.autograd.set_detect_anomaly(True)

    def forward(self,x):
        x = self.generator(x)
        return x

    def training_step(self, batch, batch_idx):
        


