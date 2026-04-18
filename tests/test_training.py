"""Tests for β-VAE training utilities."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import BetaVAE
from src.training import vae_loss, train_epoch, evaluate, train_model, TrainingMetrics


@pytest.fixture
def fake_loader():
    """DataLoader with small random images and labels."""
    torch.manual_seed(42)
    images = torch.rand(32, 1, 28, 28)
    labels = torch.randint(0, 10, (32,))
    return DataLoader(TensorDataset(images, labels), batch_size=8)


@pytest.fixture
def model():
    return BetaVAE(latent_dim=4, beta=1.0)


class TestVAELoss:
    def test_loss_shapes(self):
        recon = torch.sigmoid(torch.randn(4, 1, 28, 28))
        target = torch.rand(4, 1, 28, 28)
        mu = torch.randn(4, 10)
        logvar = torch.randn(4, 10)
        total, recon_l, kl_l = vae_loss(recon, target, mu, logvar, beta=1.0)
        assert total.shape == ()
        assert recon_l.shape == ()
        assert kl_l.shape == ()

    def test_kl_zero_at_prior(self):
        """KL should be ~0 when q(z|x) = N(0, I)."""
        recon = torch.sigmoid(torch.randn(4, 1, 28, 28))
        target = torch.rand(4, 1, 28, 28)
        mu = torch.zeros(4, 10)
        logvar = torch.zeros(4, 10)
        _, _, kl_l = vae_loss(recon, target, mu, logvar)
        assert kl_l.item() == pytest.approx(0.0, abs=1e-5)

    def test_beta_scales_kl(self):
        recon = torch.sigmoid(torch.randn(4, 1, 28, 28))
        target = torch.rand(4, 1, 28, 28)
        mu = torch.randn(4, 10)
        logvar = torch.randn(4, 10)
        total_b1, recon_b1, kl_b1 = vae_loss(recon, target, mu, logvar, beta=1.0)
        total_b4, recon_b4, kl_b4 = vae_loss(recon, target, mu, logvar, beta=4.0)
        # Recon should be the same
        assert recon_b1.item() == pytest.approx(recon_b4.item(), rel=1e-5)
        # Total should differ by 3 * KL
        expected_diff = 3.0 * kl_b1.item()
        actual_diff = total_b4.item() - total_b1.item()
        assert actual_diff == pytest.approx(expected_diff, rel=1e-4)

    def test_losses_positive(self):
        recon = torch.sigmoid(torch.randn(4, 1, 28, 28))
        target = torch.rand(4, 1, 28, 28)
        mu = torch.randn(4, 10)
        logvar = torch.randn(4, 10)
        total, recon_l, kl_l = vae_loss(recon, target, mu, logvar)
        assert total.item() > 0
        assert recon_l.item() > 0


class TestTraining:
    def test_train_epoch_returns_floats(self, model, fake_loader):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        total, recon, kl = train_epoch(model, fake_loader, optimizer)
        assert isinstance(total, float)
        assert isinstance(recon, float)
        assert isinstance(kl, float)

    def test_evaluate_returns_floats(self, model, fake_loader):
        total, recon, kl = evaluate(model, fake_loader)
        assert isinstance(total, float)

    def test_train_model_metrics(self, model, fake_loader):
        metrics = train_model(model, fake_loader, epochs=2, lr=1e-3)
        assert isinstance(metrics, TrainingMetrics)
        assert len(metrics.epoch) == 2
        assert len(metrics.total_loss) == 2
        assert metrics.epoch == [1, 2]

    def test_loss_decreases(self, fake_loader):
        model = BetaVAE(latent_dim=4, beta=1.0)
        metrics = train_model(model, fake_loader, epochs=5, lr=1e-3)
        # Loss should generally decrease over 5 epochs
        assert metrics.total_loss[-1] < metrics.total_loss[0]
