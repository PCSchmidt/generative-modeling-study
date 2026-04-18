"""Tests for β-VAE model architecture."""

import pytest
import torch

from src.models import BetaVAE, Encoder, Decoder


@pytest.fixture
def batch():
    """A small batch of fake 28x28 images."""
    torch.manual_seed(0)
    return torch.rand(4, 1, 28, 28)


class TestEncoder:
    def test_output_shapes(self, batch):
        enc = Encoder(latent_dim=10)
        mu, logvar = enc(batch)
        assert mu.shape == (4, 10)
        assert logvar.shape == (4, 10)

    def test_different_latent_dims(self, batch):
        for dim in [2, 10, 32]:
            enc = Encoder(latent_dim=dim)
            mu, logvar = enc(batch)
            assert mu.shape == (4, dim)
            assert logvar.shape == (4, dim)


class TestDecoder:
    def test_output_shape(self):
        dec = Decoder(latent_dim=10)
        z = torch.randn(4, 10)
        out = dec(z)
        assert out.shape == (4, 1, 28, 28)

    def test_output_range(self):
        """Decoder uses sigmoid — output should be in [0, 1]."""
        dec = Decoder(latent_dim=10)
        z = torch.randn(8, 10)
        out = dec(z)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestBetaVAE:
    def test_forward_shapes(self, batch):
        model = BetaVAE(latent_dim=10, beta=1.0)
        recon, mu, logvar = model(batch)
        assert recon.shape == batch.shape
        assert mu.shape == (4, 10)
        assert logvar.shape == (4, 10)

    def test_beta_attribute(self):
        model = BetaVAE(latent_dim=10, beta=4.0)
        assert model.beta == 4.0

    def test_generate_shape(self):
        model = BetaVAE(latent_dim=10, beta=1.0)
        samples = model.generate(n_samples=8)
        assert samples.shape == (8, 1, 28, 28)

    def test_generate_range(self):
        model = BetaVAE(latent_dim=10, beta=1.0)
        samples = model.generate(n_samples=4)
        assert samples.min() >= 0.0
        assert samples.max() <= 1.0

    def test_reconstruct_shape(self, batch):
        model = BetaVAE(latent_dim=10, beta=1.0)
        recon = model.reconstruct(batch)
        assert recon.shape == batch.shape

    def test_reparameterize_shape(self):
        mu = torch.zeros(4, 10)
        logvar = torch.zeros(4, 10)
        z = BetaVAE.reparameterize(mu, logvar)
        assert z.shape == (4, 10)

    def test_reparameterize_stochastic(self):
        mu = torch.zeros(4, 10)
        logvar = torch.zeros(4, 10)
        z1 = BetaVAE.reparameterize(mu, logvar)
        z2 = BetaVAE.reparameterize(mu, logvar)
        assert not torch.allclose(z1, z2), "Reparameterize should be stochastic"
