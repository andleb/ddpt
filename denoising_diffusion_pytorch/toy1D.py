import numpy as np
import torch
from torch import nn

from tqdm.auto import tqdm

# Define the U-Net architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, padding=1)
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define the Gaussian diffusion model
class GaussianDiffusion(nn.Module):
    def __init__(self, beta):
        super(GaussianDiffusion, self).__init__()
        self.beta = beta

    # FIXME: check & add schedule!
    def forward(self, x, t):
        noise = torch.randn_like(x) * self.beta * (1 - t)
        return x + noise

# Define the denoising model
class DenoisingModel(nn.Module):
    def __init__(self, in_channels, out_channels, beta):
        super(DenoisingModel, self).__init__()
        self.unet = UNet(in_channels, out_channels)
        self.diffusion = GaussianDiffusion(beta)

    def forward(self, x):
        noisy_x = self.diffusion(x)
        return self.unet(noisy_x)


class ConditionalDenoisingModel(nn.Module):
    def __init__(self, in_channels, out_channels, beta):
        super(ConditionalDenoisingModel, self).__init__()
        self.unet = UNet(in_channels + 2, out_channels)  # +1 for the condition channel, + 1 for the time
        self.diffusion = GaussianDiffusion(beta)

    def forward(self, x, condition):
        # Expand the condition tensor to have the same shape as x
        condition = condition.unsqueeze(2).expand(-1, -1, x.shape[2])
        t = torch.rand_like(x, device=x.device)
        noisy_x = self.diffusion(x, t)


        # Concatenate the condition tensor with x along the channel dimension
        noisy_x = torch.cat([noisy_x, condition, t], dim=1)

        model_out = self.unet(noisy_x)

        loss = nn.MSELoss()(model_out, x)

        return loss.mean()

    def sample(self, x, condition, nsteps=100):

        condition = condition.unsqueeze(2).expand(-1, -1, x.shape[2])
        x_start = torch.randn(x.shape, device=x.device)
        # x_start = None

        for t in reversed(np.linspace(0, 1, nsteps)):
            tt = torch.Tensor([t]).to(x.device).expand_as(condition)
            x_start = self.unet(torch.cat([x_start, condition, tt], dim=1))

        # img = unnormalize_to_zero_to_one(img)
        # return img
        # Expand the condition tensor to have the same shape as x
        # t = torch.rand_like(x, device=x.device)
        # noisy_x = self.diffusion(x, t)

        # Concatenate the condition tensor with x along the channel dimension
        # noisy_x = torch.cat([noisy_x, condition, t], dim=1)

        # model_out = self.unet(noisy_x)

        return x_start

