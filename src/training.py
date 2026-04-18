"""Training utilities for β-VAE."""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import BetaVAE


@dataclass
class TrainingMetrics:
    """Container for per-epoch training metrics."""

    epoch: list[int] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)
    recon_loss: list[float] = field(default_factory=list)
    kl_loss: list[float] = field(default_factory=list)


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the β-VAE ELBO loss.

    L = BCE(recon, target) + β · KL(q(z|x) || p(z))

    Returns (total_loss, recon_loss, kl_loss) averaged over the batch.
    """
    recon_loss = F.binary_cross_entropy(recon, target, reduction="sum") / target.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


def train_epoch(
    model: BetaVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
) -> tuple[float, float, float]:
    """Train for one epoch. Returns (avg_total, avg_recon, avg_kl)."""
    model.train()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    n_batches = 0

    for images, _ in loader:
        images = images.to(device)
        recon, mu, logvar = model(images)
        loss, recon_l, kl_l = vae_loss(recon, images, mu, logvar, beta=model.beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_l.item()
        total_kl += kl_l.item()
        n_batches += 1

    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


@torch.no_grad()
def evaluate(
    model: BetaVAE,
    loader: DataLoader,
    device: torch.device | str = "cpu",
) -> tuple[float, float, float]:
    """Evaluate on a data loader. Returns (avg_total, avg_recon, avg_kl)."""
    model.eval()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    n_batches = 0

    for images, _ in loader:
        images = images.to(device)
        recon, mu, logvar = model(images)
        loss, recon_l, kl_l = vae_loss(recon, images, mu, logvar, beta=model.beta)
        total_loss += loss.item()
        total_recon += recon_l.item()
        total_kl += kl_l.item()
        n_batches += 1

    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


def train_model(
    model: BetaVAE,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
) -> TrainingMetrics:
    """Full training loop with optional validation tracking."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metrics = TrainingMetrics()

    for epoch in range(1, epochs + 1):
        train_total, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device)
        metrics.epoch.append(epoch)
        metrics.total_loss.append(train_total)
        metrics.recon_loss.append(train_recon)
        metrics.kl_loss.append(train_kl)

    return metrics
