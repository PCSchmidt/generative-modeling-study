"""A 2-D MLP VAE and a small DDPM, each with a proper negative-log-likelihood bound.

VAE:  `nll` is an importance-weighted (IWAE) estimate, an upper bound on -log p(x) in expectation.
DDPM: `nll` is the full variational bound (Ho et al. 2020, eq. 5), summing all T terms, with a
      Gaussian decoder at t=1. `dsm` is the plain denoising error, a cheaper score.
Both are checked against the exact data density in the tests.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

Array = npt.NDArray[np.float64]
LOG_2PI = math.log(2.0 * math.pi)


def _mlp(i: int, h: int, o: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def _t(x: Array) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


# ------------------------------------------------------------------------------------------- VAE


class VAE(nn.Module):
    def __init__(self, latent: int = 2, hidden: int = 64) -> None:
        super().__init__()
        self.latent = latent
        self.enc = _mlp(2, hidden, 2 * latent)
        self.dec = _mlp(latent, hidden, 4)  # mean (2) + log-variance (2)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.enc(x).chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 4.0)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.dec(z).chunk(2, dim=-1)
        return mean, logvar.clamp(-10.0, 4.0)

    def neg_elbo(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        mean, dlv = self.decode(z)
        rec = 0.5 * (LOG_2PI * 2 + dlv.sum(-1) + ((x - mean) ** 2 / dlv.exp()).sum(-1))
        kl = 0.5 * (mu**2 + logvar.exp() - 1.0 - logvar).sum(-1)
        return rec + kl

    @torch.no_grad()
    def nll(self, x: Array, k: int = 128, seed: int = 0) -> Array:
        """Importance-weighted NLL estimate in nats, one value per row."""
        gen = torch.Generator().manual_seed(seed)
        xt = _t(x)
        mu, logvar = self.encode(xt)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn((k, *mu.shape), generator=gen)
        z = mu + eps * std
        mean, dlv = self.decode(z)
        log_px_z = -0.5 * (LOG_2PI * 2 + dlv.sum(-1) + ((xt - mean) ** 2 / dlv.exp()).sum(-1))
        log_pz = -0.5 * (z**2 + LOG_2PI).sum(-1)
        log_qz = -0.5 * (eps**2 + LOG_2PI + logvar).sum(-1)
        lw = log_px_z + log_pz - log_qz
        return (-(torch.logsumexp(lw, dim=0) - math.log(k))).numpy().astype(np.float64)

    @torch.no_grad()
    def recon_error(self, x: Array) -> Array:
        """Squared error between x and its deterministic reconstruction."""
        xt = _t(x)
        mu, _ = self.encode(xt)
        mean, _ = self.decode(mu)
        return ((xt - mean) ** 2).sum(-1).numpy().astype(np.float64)

    @torch.no_grad()
    def latent_mean(self, x: Array) -> Array:
        return self.encode(_t(x))[0].numpy().astype(np.float64)


def train_vae(x: Array, seed: int, steps: int = 6000, batch: int = 256, lr: float = 2e-3) -> VAE:
    torch.manual_seed(seed)
    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    data = _t(x)
    for _ in range(steps):
        xb = data[torch.randint(0, len(data), (batch,))]
        loss = model.neg_elbo(xb).mean()
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        sched.step()
    model.eval()
    return model


# ------------------------------------------------------------------------------------------- DDPM


class DDPM(nn.Module):
    betas: torch.Tensor
    alphas: torch.Tensor
    abar: torch.Tensor
    post_var: torch.Tensor

    def __init__(self, steps: int = 200, beta_max: float = 0.05, hidden: int = 128) -> None:
        super().__init__()
        self.T = steps
        betas = torch.linspace(1e-4, beta_max, steps)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, 0)
        abar_prev = torch.cat([torch.ones(1), abar[:-1]])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)
        # posterior variance beta~_t; at t=1 (index 0) the decoder variance is beta_1 instead
        post = betas * (1.0 - abar_prev) / (1.0 - abar)
        post[0] = betas[0]
        self.register_buffer("post_var", post)
        self.net = _mlp(2 + 16, hidden, 2, depth=3)

    def _embed(self, t_idx: torch.Tensor) -> torch.Tensor:
        freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), 8))
        ang = (t_idx.float() / self.T)[..., None] * freqs
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def eps(self, xt: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([xt, self._embed(t_idx)], dim=-1))  # type: ignore[no-any-return]

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.T, (len(x0),))
        eps = torch.randn_like(x0)
        ab = self.abar[t][:, None]
        xt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps
        return ((eps - self.eps(xt, t)) ** 2).sum(-1).mean()

    @torch.no_grad()
    def dsm(self, x: Array, seed: int = 0) -> Array:
        """Mean denoising error over all T timesteps with one fixed noise draw each."""
        gen = torch.Generator().manual_seed(seed)
        x0 = _t(x)
        n = len(x0)
        eps = torch.randn((self.T, n, 2), generator=gen)
        t = torch.arange(self.T)[:, None].expand(self.T, n)
        ab = self.abar[:, None, None]
        xt = ab.sqrt() * x0[None] + (1.0 - ab).sqrt() * eps
        err = ((eps - self.eps(xt.reshape(-1, 2), t.reshape(-1)).reshape(self.T, n, 2)) ** 2).sum(-1)
        return err.mean(0).numpy().astype(np.float64)

    @torch.no_grad()
    def nll(self, x: Array, seed: int = 0) -> Array:
        """Full variational bound on -log p(x), one value per row, in nats."""
        gen = torch.Generator().manual_seed(seed)
        x0 = _t(x)
        n = len(x0)
        T = self.T
        eps = torch.randn((T, n, 2), generator=gen)
        t = torch.arange(T)[:, None].expand(T, n)
        ab = self.abar[:, None, None]
        xt = ab.sqrt() * x0[None] + (1.0 - ab).sqrt() * eps
        pred = self.eps(xt.reshape(-1, 2), t.reshape(-1)).reshape(T, n, 2)
        sq = ((eps - pred) ** 2).sum(-1)  # (T, n)
        beta, alpha, abar, pv = self.betas, self.alphas, self.abar, self.post_var
        w = beta**2 / (2.0 * pv * alpha * (1.0 - abar))  # weight of ||eps - eps_theta||^2, t >= 2
        terms = w[:, None] * sq
        # t = 1: Gaussian decoder N(x0; mu_theta(x_1, 1), beta_1 I)
        mu1 = (xt[0] - beta[0] / (1.0 - abar[0]).sqrt() * pred[0]) / alpha[0].sqrt()
        l1 = 0.5 * (LOG_2PI * 2 + 2.0 * pv[0].log() + ((x0 - mu1) ** 2).sum(-1) / pv[0])
        # t = T: KL(q(x_T | x_0) || N(0, I))
        a_t = abar[-1]
        l_t = 0.5 * (-2.0 * a_t + a_t * (x0**2).sum(-1) - 2.0 * (1.0 - a_t).log())
        total = l_t + terms[1:].sum(0) + l1
        return total.numpy().astype(np.float64)


def train_ddpm(x: Array, seed: int, steps: int = 12000, batch: int = 512, lr: float = 2e-3) -> DDPM:
    torch.manual_seed(seed)
    model = DDPM()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    data = _t(x)
    for _ in range(steps):
        xb = data[torch.randint(0, len(data), (batch,))]
        loss = model.loss(xb)
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        sched.step()
    model.eval()
    return model
