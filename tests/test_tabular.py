from __future__ import annotations

import numpy as np
import torch

from src.calibration.metrics import auroc
from src.tabular import models as M
from src.tabular.data import DEV_BYTE_LIMIT, _split, pool_of


def test_pool_assignment_is_deterministic_and_about_thirty_percent() -> None:
    a, b = pool_of(5000), pool_of(5000)
    assert (a == b).all()
    assert 0.27 < a.mean() < 0.33 and DEV_BYTE_LIMIT == 77


def test_split_parts_are_disjoint_and_respect_caps() -> None:
    rows = np.arange(1000)
    parts = _split(rows, np.random.default_rng(0), (0.6, 0.1, 0.1, 0.2), (300, 50, 50, 100))
    sizes = [len(p) for p in parts]
    assert sizes == [300, 50, 50, 100]
    allr = np.concatenate(parts)
    assert len(set(allr.tolist())) == len(allr)


def test_split_without_caps_covers_the_requested_fractions() -> None:
    parts = _split(np.arange(1000), np.random.default_rng(1), (0.5, 0.1, 0.2, 0.2), (10**9,) * 4)
    assert [len(p) for p in parts] == [500, 100, 200, 200]


def test_ebm_energy_separates_a_gaussian_from_far_points() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 3))
    ebm = M.train_ebm(x, seed=0, hidden=64, steps=1500)
    inside = ebm.score(rng.normal(size=(500, 3)))
    far = ebm.score(rng.normal(size=(500, 3)) * 3.0 + 4.0)
    assert auroc(inside, far) > 0.9  # higher energy = more unusual


def test_vae_and_ddpm_scores_have_one_value_per_row_and_are_deterministic() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, 4))
    vae = M.train_vae(x, seed=0, latent=2, steps=50)
    ddpm = M.train_ddpm(x, seed=0, hidden=32, steps=50)
    for fn in (vae.nll, vae.recon_error, ddpm.dsm):
        a = fn(x[:100])
        assert a.shape == (100,) and np.isfinite(a).all()
        assert np.allclose(a, fn(x[:100]))


def test_ebm_nce_loss_is_two_log_two_when_the_model_equals_the_noise_density() -> None:
    # if the unnormalised model density is exactly the noise density, the classifier logit is 0 everywhere
    rng = np.random.default_rng(2)
    x = rng.normal(size=(5000, 2))
    ebm = M.EBM(x, hidden=8, noise_scale=1.0).to(M.DEVICE)

    class MatchNoise(torch.nn.Module):
        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return -ebm.log_noise(z).unsqueeze(-1)

    ebm.energy = MatchNoise()  # type: ignore[assignment]
    assert abs(ebm.val_loss(x) - 2 * np.log(2)) < 1e-4
