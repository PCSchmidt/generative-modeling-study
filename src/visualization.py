"""Visualization utilities for β-VAE study."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .models import BetaVAE
from .training import TrainingMetrics


def plot_loss_curves(
    metrics: TrainingMetrics,
    title: str = "Training Loss",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot total, reconstruction, and KL loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(metrics.epoch, metrics.total_loss, "b-")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(metrics.epoch, metrics.recon_loss, "g-")
    axes[1].set_title("Reconstruction Loss")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(metrics.epoch, metrics.kl_loss, "r-")
    axes[2].set_title("KL Divergence")
    axes[2].set_xlabel("Epoch")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_reconstructions(
    model: BetaVAE,
    images: torch.Tensor,
    n_show: int = 8,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Show original images vs. their reconstructions."""
    recon = model.reconstruct(images[:n_show])
    originals = images[:n_show].cpu().numpy()
    reconstructed = recon.cpu().numpy()

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 1.5, 3))
    for i in range(n_show):
        axes[0, i].imshow(originals[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i, 0], cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Recon", fontsize=10)
    plt.suptitle(f"Reconstructions (β={model.beta})", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_generations(
    model: BetaVAE,
    n_samples: int = 16,
    device: torch.device | str = "cpu",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Generate and display random samples from the prior."""
    samples = model.generate(n_samples, device=device).cpu().numpy()
    nrow = int(np.ceil(np.sqrt(n_samples)))
    ncol = int(np.ceil(n_samples / nrow))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.5, nrow * 1.5))
    axes_flat = axes.flatten()
    for i in range(n_samples):
        axes_flat[i].imshow(samples[i, 0], cmap="gray")
        axes_flat[i].axis("off")
    for i in range(n_samples, len(axes_flat)):
        axes_flat[i].axis("off")

    plt.suptitle(f"Generated Samples (β={model.beta})", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_latent_space(
    model: BetaVAE,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device | str = "cpu",
    max_points: int = 5000,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Scatter plot of first two latent dimensions, colored by digit class."""
    model.eval()
    all_mu, all_labels = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            mu, _ = model.encoder(images.to(device))
            all_mu.append(mu.cpu())
            all_labels.append(labels)
            if sum(m.size(0) for m in all_mu) >= max_points:
                break

    mu_cat = torch.cat(all_mu)[:max_points].numpy()
    labels_cat = torch.cat(all_labels)[:max_points].numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(mu_cat[:, 0], mu_cat[:, 1], c=labels_cat, cmap="tab10", s=4, alpha=0.6)
    plt.colorbar(scatter, ax=ax, label="Digit class")
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.set_title(f"Latent Space (β={model.beta})")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_latent_interpolation(
    model: BetaVAE,
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    n_steps: int = 10,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Interpolate between two latent vectors and decode each step."""
    alphas = torch.linspace(0, 1, n_steps).unsqueeze(1)
    z_interp = z_start * (1 - alphas) + z_end * alphas

    with torch.no_grad():
        images = model.decoder(z_interp).cpu().numpy()

    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 1.5, 1.5))
    for i in range(n_steps):
        axes[i].imshow(images[i, 0], cmap="gray")
        axes[i].axis("off")
    plt.suptitle("Latent Interpolation", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_beta_comparison(
    metrics_dict: dict[float, TrainingMetrics],
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Compare loss curves across different β values."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for beta_val, metrics in metrics_dict.items():
        axes[0].plot(metrics.epoch, metrics.total_loss, label=f"β={beta_val}")
        axes[1].plot(metrics.epoch, metrics.recon_loss, label=f"β={beta_val}")
        axes[2].plot(metrics.epoch, metrics.kl_loss, label=f"β={beta_val}")

    axes[0].set_title("Total Loss")
    axes[1].set_title("Reconstruction Loss")
    axes[2].set_title("KL Divergence")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.suptitle("β Sweep Comparison", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
