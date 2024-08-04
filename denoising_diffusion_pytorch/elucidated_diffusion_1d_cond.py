# NOTE: this is EDM - Karras 2022

from collections import deque
from math import sqrt
from random import random

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from tqdm.auto import tqdm


# helpers
def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


# tensor helpers
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


# normalization functions
def normalize_to_neg_one_to_one(seq):
    return seq * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# main class
class ElucidatedDiffusion1Dcond(nn.Module):
    def __init__(self, net, *, seq_length, channels=1, num_sample_steps=32,  # number of sampling steps
                 auto_normalize=False,  # we have unbounded data #TODO: check what engression does
                 sigma_min=0.002,  # min noise level
                 sigma_max=80,  # max noise level
                 # NOTE: if standardizing the data, adjust this
                 sigma_data=0.5,  # standard deviation of data distribution
                 rho=7,  # controls the sampling schedule
                 P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
                 P_std=1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
                 S_churn=80,  # parameters for stochastic sampling - depends on dataset, Table 5 in paper
                 S_tmin=0.05, S_tmax=50, S_noise=1.003):

        super().__init__()
        # TODO: karras doesn't have these, not used in this file
        # assert net.random_or_learned_sinusoidal_cond

        self.net = net
        self.self_condition = net.self_condition

        # sequence dimensions
        self.channels = channels
        self.seq_length = seq_length

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

        # whether to autonormalize
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return next(self.net.parameters()).device

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # NOTE:  preconditioned network output, equation (7) in the paper
    def preconditioned_network_forward(self, noised_seq, sigma, condition=None, self_cond=None, clamp=False):

        batch, device = noised_seq.shape[0], noised_seq.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        # NOTE: reduced this by one dim
        padded_sigma = rearrange(sigma, 'b -> b 1 1')

        # NOTE: only model call, param format currently for karras1dcont
        net_out = self.net(self.c_in(padded_sigma) * noised_seq, self.c_noise(sigma), condition=condition,
                           self_cond=self_cond)

        # TODO: figure out which of these would relate to the data for engression
        out = self.c_skip(padded_sigma) * noised_seq + self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    # sampling

    # sample schedule
    # equation (5) in the paper
    def sample_schedule(self, num_sample_steps=None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (
                self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.)  # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def sample(self, condition=None, batch_size=16, num_sample_steps=None, clamp=False):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        shape = (batch_size, self.channels, self.seq_length)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where((sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
                             min(self.S_churn / num_sample_steps, sqrt(2) - 1), 0.)

        # Interpolating between sigmas
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # sequence is noise at the beginning, regardless of the conditioning
        init_sigma = sigmas[0]
        seqs = init_sigma * torch.randn(shape, device=self.device)

        # for self conditioning
        x_start = None

        # gradually denoise
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc='sampling time step'):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device=self.device)  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            seqs_hat = seqs + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            self_cond = x_start if self.self_condition else None

            # NOTE: condition passed here
            model_output = self.preconditioned_network_forward(seqs_hat, sigma_hat, condition, self_cond, clamp=clamp)
            # kind of a relative estimated error
            diff_over_sigma = (seqs_hat - model_output) / sigma_hat

            seqs_next = seqs_hat + (sigma_next - sigma_hat) * diff_over_sigma

            # second order correction, if not the last timestep
            if sigma_next != 0:
                self_cond = model_output if self.self_condition else None

                # NOTE: condition passed here
                model_output_next = self.preconditioned_network_forward(seqs_next, sigma_next, condition, self_cond,
                                                                        clamp=clamp)
                diff_prime_over_sigma = (seqs_next - model_output_next) / sigma_next
                seqs_next = seqs_hat + 0.5 * (sigma_next - sigma_hat) * (diff_over_sigma + diff_prime_over_sigma)

            seqs = seqs_next
            # remember: this is x-prediction, after all
            x_start = model_output_next if sigma_next != 0 else model_output

        if clamp:
            seqs = seqs.clamp(-1., 1.)

        return self.unnormalize(seqs)

    @torch.no_grad()
    def sample_using_dpmpp(self, condition=None, batch_size=16, num_sample_steps=None, clamp=False):
        """
        thanks to Katherine Crowson (https://github.com/crowsonkb) for figuring it all out!
        https://arxiv.org/abs/2211.01095
        """

        device, num_sample_steps = self.device, default(num_sample_steps, self.num_sample_steps)

        sigmas = self.sample_schedule(num_sample_steps)

        shape = (batch_size, self.channels, self.seq_length)
        seqs = sigmas[0] * torch.randn(shape, device=device)

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        old_denoised = None
        for i in tqdm(range(len(sigmas) - 1)):
            denoised = self.preconditioned_network_forward(seqs, sigmas[i].item(), condition=condition)
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t

            if not exists(old_denoised) or sigmas[i + 1] == 0:
                denoised_d = denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                gamma = - 1 / (2 * r)
                denoised_d = (1 - gamma) * denoised + gamma * old_denoised

            seqs = (sigma_fn(t_next) / sigma_fn(t)) * seqs - (-h).expm1() * denoised_d
            old_denoised = denoised

        if clamp:
            seqs = seqs.clamp(-1., 1.)
        return self.unnormalize(seqs)

    # training

    # as in the original paper, wtilde
    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)).exp()

    def forward(self, seqs, condition=None):
        batch_size, c, n, device, seq_length, channels = *seqs.shape, seqs.device, self.seq_length, self.channels
        assert n == seq_length, f'seq length must be {seq_length}'
        assert c == channels, 'mismatch of image channels'

        # unnormalized - in sampling
        seqs = self.normalize(seqs)

        sigmas = self.noise_distribution(batch_size)
        # NOTE: reduced this by one dim
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1')

        noise = torch.randn_like(seqs)

        noised_seqs = seqs + padded_sigmas * noise  # alphas are 1. in the paper

        self_cond = None
        if self.self_condition and random() < 0.5:
            # from hinton's group's bit diffusion paper
            with torch.no_grad():
                # NOTE: the logic is that we pass conditional again, since
                # each cond. diffusion process an independent diffusion process in itself
                self_cond = self.preconditioned_network_forward(noised_seqs, sigmas, condition=condition)
                self_cond.detach_()

        denoised = self.preconditioned_network_forward(noised_seqs, sigmas, condition=condition, self_cond=self_cond)

        # this is just weighted x-prediction, as in paper
        losses = F.mse_loss(denoised, seqs, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        losses = losses * self.loss_weight(sigmas)

        return losses.mean()


################################################################################
# TODO: add the dataset etc. files if needed
# TODO: re-work to batching

def train(diffusion: ElucidatedDiffusion1Dcond, seqs, conditions, num_epochs=100, batch_size=32, lr=1e-4,
          adam_betas=(0.9, 0.999), ema_decay=0.999, log_interval=10, save_interval=1000,
          save_path='elucidated_diffusion.pt',
          return_loss=False,
          test_loss=False,
          early_stopping=None,
           **kwargs):

    device = diffusion.device

    optimizer = Adam(diffusion.parameters(), lr=lr, betas=adam_betas)

    # TODO: EMA decay


    if return_loss:
        train_losses = []
        if test_loss:
            test_seqs = kwargs.get('test_seqs')
            test_conditions = kwargs.get('test_conditions')
            test_losses = []


    if early_stopping is not None:
        early_stopping_buffer = deque(maxlen=int(early_stopping))
        early_stopping_threshold = kwargs.get('early_stopping_threshold', 0.5)

    pbar = tqdm(range(num_epochs), desc='Loss: N/A')
    for epoch in pbar:
        optimizer.zero_grad()
        loss = diffusion(seqs=seqs, condition=conditions)
        loss.backward()
        # print(loss.item())
        optimizer.step()
        pbar.set_description("Loss: %.4f" % loss.item())

        if return_loss:
            train_losses.append(loss.item())
            if test_loss:
                with torch.no_grad():
                    test_loss = diffusion(seqs=test_seqs, condition=test_conditions)
                    test_losses.append(test_loss.item())

        if early_stopping is not None:
            early_stopping_buffer.append(loss.item())
            if len(early_stopping_buffer) == early_stopping:
                # NOTE: need to add abs if checking every step
                if np.abs(early_stopping_buffer[0] - early_stopping_buffer[-1]) < early_stopping_threshold:
                    print(f'Early stopping at epoch {epoch}! Prev loss: {early_stopping_buffer[0]}, curr. loss: {early_stopping_buffer[-1]}')
                    break


    if return_loss:
        if test_loss:
            return train_losses, test_losses
        return train_losses


#%%
