"""β-VAE model architecture for MNIST generative modeling."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Convolutional encoder mapping 28x28 images to latent distribution parameters."""

    def __init__(self, latent_dim: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(64 * 7 * 7, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    """Transposed-conv decoder mapping latent vectors to 28x28 images."""

    def __init__(self, latent_dim: int = 10):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc(z))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), 64, 7, 7)
        x = F.relu(self.deconv1(x))
        return torch.sigmoid(self.deconv2(x))


class BetaVAE(nn.Module):
    """β-VAE: Variational autoencoder with tunable β for disentanglement.

    When β=1 this is a standard VAE. β>1 encourages a more disentangled
    latent representation at the cost of reconstruction fidelity.
    """

    def __init__(self, latent_dim: int = 10, beta: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, sigma^2) using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def generate(self, n_samples: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Generate new images by sampling from the prior N(0, I)."""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        with torch.no_grad():
            return self.decoder(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input images (deterministic — uses mean, no sampling)."""
        mu, _ = self.encoder(x)
        with torch.no_grad():
            return self.decoder(mu)
