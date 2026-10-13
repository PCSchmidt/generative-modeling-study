"""d-dimensional VAE, DDPM and noise-contrastive EBM, with the scores used in docs/PROTOCOL_TABULAR.md.

Every score is "higher = more unusual". The 2-D versions in `src/calibration/generators.py` are untouched.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

Array = npt.NDArray[np.float64]
LOG_2PI = math.log(2.0 * math.pi)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mlp(i: int, h: int, o: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def _t(x: Array) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)


def _np(x: torch.Tensor) -> Array:
    return x.detach().cpu().numpy().astype(np.float64)


def _batches(n: int, size: int) -> list[slice]:
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


# ------------------------------------------------------------------------------------------- VAE


class VAE(nn.Module):
    def __init__(self, dim: int, latent: int, hidden: int = 256) -> None:
        super().__init__()
        self.dim, self.latent = dim, latent
        self.enc = _mlp(dim, hidden, 2 * latent, 2)
        self.dec = _mlp(latent, hidden, 2 * dim, 2)

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
        rec = 0.5 * (
            LOG_2PI * self.dim + dlv.sum(-1) + ((x - mean) ** 2 / dlv.exp()).sum(-1)
        )
        kl = 0.5 * (mu**2 + logvar.exp() - 1.0 - logvar).sum(-1)
        return rec + kl

    @torch.no_grad()
    def nll(self, x: Array, k: int = 64, seed: int = 0) -> Array:
        """Importance-weighted NLL estimate in nats, one value per row."""
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        out = []
        for sl in _batches(len(x), 1000):
            xt = _t(x[sl])
            mu, logvar = self.encode(xt)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn((k, *mu.shape), generator=gen, device=DEVICE)
            z = mu + eps * std
            mean, dlv = self.decode(z)
            log_px_z = -0.5 * (
                LOG_2PI * self.dim
                + dlv.sum(-1)
                + ((xt - mean) ** 2 / dlv.exp()).sum(-1)
            )
            log_pz = -0.5 * (z**2 + LOG_2PI).sum(-1)
            log_qz = -0.5 * (eps**2 + LOG_2PI + logvar).sum(-1)
            lw = log_px_z + log_pz - log_qz
            out.append(-(torch.logsumexp(lw, dim=0) - math.log(k)))
        return _np(torch.cat(out))

    @torch.no_grad()
    def recon_error(self, x: Array) -> Array:
        xt = _t(x)
        mu, _ = self.encode(xt)
        mean, _ = self.decode(mu)
        return _np(((xt - mean) ** 2).sum(-1))


def _fit(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    data: torch.Tensor,
    seed: int,
    steps: int,
    batch: int,
    lr: float,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    for _ in range(steps):
        xb = data[torch.randint(0, len(data), (batch,), generator=gen, device=DEVICE)]
        loss = loss_fn(xb)
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        sched.step()
    model.eval()


def train_vae(
    x: Array,
    seed: int,
    latent: int,
    steps: int = 8000,
    batch: int = 512,
    lr: float = 2e-3,
) -> VAE:
    torch.manual_seed(seed)
    model = VAE(x.shape[1], latent).to(DEVICE)
    _fit(model, lambda xb: model.neg_elbo(xb).mean(), _t(x), seed, steps, batch, lr)
    return model


# ------------------------------------------------------------------------------------------- DDPM


class DDPM(nn.Module):
    betas: torch.Tensor
    abar: torch.Tensor

    def __init__(
        self, dim: int, hidden: int, steps: int = 200, beta_max: float = 0.05
    ) -> None:
        super().__init__()
        self.dim, self.T = dim, steps
        betas = torch.linspace(1e-4, beta_max, steps)
        self.register_buffer("betas", betas)
        self.register_buffer("abar", torch.cumprod(1.0 - betas, 0))
        self.net = _mlp(dim + 16, hidden, dim, 3)

    def _embed(self, t_idx: torch.Tensor) -> torch.Tensor:
        freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), 8, device=t_idx.device))
        ang = (t_idx.float() / self.T)[..., None] * freqs
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def eps(self, xt: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([xt, self._embed(t_idx)], dim=-1))  # type: ignore[no-any-return]

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.T, (len(x0),), device=x0.device)
        eps = torch.randn_like(x0)
        ab = self.abar[t][:, None]
        xt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps
        return ((eps - self.eps(xt, t)) ** 2).sum(-1).mean()

    @torch.no_grad()
    def dsm(self, x: Array, seed: int = 0) -> Array:
        """Mean denoising error over all T timesteps with one fixed noise draw each."""
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        out = []
        for sl in _batches(len(x), 500):
            x0 = _t(x[sl])
            n = len(x0)
            eps = torch.randn((self.T, n, self.dim), generator=gen, device=DEVICE)
            t = torch.arange(self.T, device=DEVICE)[:, None].expand(self.T, n)
            ab = self.abar[:, None, None]
            xt = ab.sqrt() * x0[None] + (1.0 - ab).sqrt() * eps
            pred = self.eps(xt.reshape(-1, self.dim), t.reshape(-1)).reshape(
                self.T, n, self.dim
            )
            out.append(((eps - pred) ** 2).sum(-1).mean(0))
        return _np(torch.cat(out))


def train_ddpm(
    x: Array,
    seed: int,
    hidden: int,
    steps: int = 20000,
    batch: int = 512,
    lr: float = 2e-3,
) -> DDPM:
    torch.manual_seed(seed)
    model = DDPM(x.shape[1], hidden).to(DEVICE)
    _fit(model, model.loss, _t(x), seed, steps, batch, lr)
    return model


# ------------------------------------------------------------------------------------------- EBM


class EBM(nn.Module):
    """Unnormalised energy E(x), trained by noise-contrastive estimation against a covariance-matched Gaussian."""

    mean: torch.Tensor
    chol: torch.Tensor
    log_det: torch.Tensor

    def __init__(self, x_train: Array, hidden: int, noise_scale: float = 2.0) -> None:
        super().__init__()
        d = x_train.shape[1]
        self.dim = d
        self.energy = _mlp(d, hidden, 1, 3)
        self.log_c = nn.Parameter(torch.zeros(()))  # learnable log-normaliser
        cov = np.cov(x_train, rowvar=False) * noise_scale + 1e-6 * np.eye(d)
        self.register_buffer(
            "mean", torch.as_tensor(x_train.mean(0), dtype=torch.float32)
        )
        self.register_buffer(
            "chol", torch.as_tensor(np.linalg.cholesky(cov), dtype=torch.float32)
        )
        self.register_buffer(
            "log_det", torch.as_tensor(np.linalg.slogdet(cov)[1], dtype=torch.float32)
        )

    def log_noise(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.linalg.solve_triangular(self.chol, (x - self.mean).T, upper=False).T
        out: torch.Tensor = -0.5 * (z**2).sum(-1) - 0.5 * (
            self.log_det + self.dim * LOG_2PI
        )
        return out

    def sample_noise(self, n: int, gen: torch.Generator | None = None) -> torch.Tensor:
        eps = torch.randn((n, self.dim), generator=gen, device=self.mean.device)
        out: torch.Tensor = self.mean + eps @ self.chol.T
        return out

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = -self.energy(x).squeeze(-1) - self.log_c - self.log_noise(x)
        return out

    def nce_loss(self, xb: torch.Tensor) -> torch.Tensor:
        noise = self.sample_noise(len(xb))
        return (
            nn.functional.softplus(-self.logit(xb)).mean()
            + nn.functional.softplus(self.logit(noise)).mean()
        )

    @torch.no_grad()
    def score(self, x: Array) -> Array:
        return np.concatenate(
            [_np(self.energy(_t(x[sl])).squeeze(-1)) for sl in _batches(len(x), 5000)]
        )

    @torch.no_grad()
    def val_loss(self, x: Array, seed: int = 0) -> float:
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        xv = _t(x)
        noise = self.sample_noise(len(xv), gen)
        pos = nn.functional.softplus(-self.logit(xv)).mean()
        neg = nn.functional.softplus(self.logit(noise)).mean()
        return float((pos + neg).item())


def train_ebm(
    x: Array,
    seed: int,
    hidden: int,
    steps: int = 10000,
    batch: int = 512,
    lr: float = 2e-3,
) -> EBM:
    torch.manual_seed(seed)
    model = EBM(x, hidden).to(DEVICE)
    _fit(model, model.nce_loss, _t(x), seed, steps, batch, lr)
    return model


def ddpm_val_loss(model: DDPM, x: Array, seed: int = 0) -> float:
    return float(model.dsm(x, seed).mean())
