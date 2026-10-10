"""Tests for the calibration benchmark: exact density, split conformal, metrics, generators."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.calibration import data
from src.calibration.conformal import (
    conformal_threshold,
    coverage,
    coverage_ece,
    repeated_coverage,
)
from src.calibration.generators import DDPM, VAE, train_ddpm, train_vae
from src.calibration.metrics import Mahalanobis, auroc

Array = npt.NDArray[np.float64]

# ---------------------------------------------------------------------------------- exact data


def test_density_integrates_to_one() -> None:
    xs = np.linspace(-1.8, 2.8, 230)
    ys = np.linspace(-1.5, 2.0, 175)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    total = np.exp(data.true_logpdf(pts)).sum() * (xs[1] - xs[0]) * (ys[1] - ys[0])
    assert total == pytest.approx(1.0, abs=2e-3)


def test_samples_are_far_likelier_than_uniform_box() -> None:
    rng = np.random.default_rng(0)
    on = -data.true_logpdf(data.sample(2000, rng)).mean()
    off = -data.true_logpdf(data.uniform_box(2000, rng)).mean()
    assert on < off - 3.0


def test_sampler_matches_density_via_importance_check() -> None:
    """E_p[f(x)] from samples equals the integral of f * density, for a simple f."""
    rng = np.random.default_rng(1)

    def f(x: Array) -> Array:
        return np.asarray(x[:, 0] ** 2 + x[:, 1], dtype=np.float64)

    mc = f(data.sample(200_000, rng)).mean()
    xs = np.linspace(-1.8, 2.8, 230)
    ys = np.linspace(-1.5, 2.0, 175)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    quad = (f(pts) * np.exp(data.true_logpdf(pts))).sum() * (xs[1] - xs[0]) * (ys[1] - ys[0])
    assert mc == pytest.approx(quad, abs=0.01)


def test_translate_moves_every_point() -> None:
    x = data.sample(10, np.random.default_rng(2))
    assert np.allclose(data.translate(x, 0.5)[:, 0] - x[:, 0], 0.5)


# ---------------------------------------------------------------------------- split conformal


def test_threshold_known_values() -> None:
    scores = np.arange(1.0, 10.0, dtype=np.float64)  # n = 9
    assert conformal_threshold(scores, 0.1) == 9.0  # ceil(10 * 0.9) = 9
    assert conformal_threshold(scores, 0.5) == 5.0  # ceil(10 * 0.5) = 5
    assert math.isinf(conformal_threshold(scores, 0.05))  # ceil(10 * 0.95) = 10 > n


def test_threshold_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        conformal_threshold(np.array([1.0]), 0.0)
    with pytest.raises(ValueError):
        conformal_threshold(np.array([1.0]), 1.0)
    with pytest.raises(ValueError):
        conformal_threshold(np.array([]), 0.1)


@settings(max_examples=20, deadline=None)
@given(st.integers(20, 200), st.floats(0.05, 0.5), st.integers(0, 10_000))
def test_marginal_coverage_matches_exact_expectation(n: int, alpha: float, seed: int) -> None:
    """For exchangeable scores, coverage averaged over calibration draws equals k / (n + 1)
    with k = ceil((n + 1)(1 - alpha)): at least 1 - alpha, and at most 1 / (n + 1) above it.
    The tolerance is 4.5 Monte-Carlo standard errors, derived from the number of trials."""
    rng = np.random.default_rng(seed)
    trials = 3000
    cal = rng.normal(size=(trials, n))
    new = rng.normal(size=trials)
    cov = float(np.mean([new[i] <= conformal_threshold(cal[i], alpha) for i in range(trials)]))
    expected = math.ceil((n + 1) * (1 - alpha)) / (n + 1)
    assert expected >= 1 - alpha and expected <= 1 - alpha + 1.0 / (n + 1)
    tol = 4.5 * math.sqrt(expected * (1 - expected) / trials)
    assert abs(cov - expected) <= tol


def test_coverage_breaks_when_exchangeability_breaks() -> None:
    rng = np.random.default_rng(3)
    cal = rng.normal(size=2000)
    q = conformal_threshold(cal, 0.1)
    assert coverage(rng.normal(size=5000), q) == pytest.approx(0.9, abs=0.02)
    assert coverage(rng.normal(loc=2.0, size=5000), q) < 0.5  # shifted scores: guarantee void


def test_repeated_coverage_mean_near_nominal_and_spread_matches_beta() -> None:
    rng = np.random.default_rng(4)
    pool, test = rng.normal(size=20_000), rng.normal(size=20_000)
    cov = repeated_coverage(pool, test, 0.1, 500, 400, rng)
    assert cov.mean() == pytest.approx(0.9, abs=0.01)
    # Beta(451, 50) has sd ~ 0.0134; allow a loose band
    assert 0.007 < cov.std() < 0.025


def test_coverage_ece_zero_when_perfect() -> None:
    nom = np.array([0.5, 0.9])
    assert coverage_ece(nom, nom) == 0.0
    assert coverage_ece(nom, nom - 0.1) == pytest.approx(0.1)


# --------------------------------------------------------------------------------------- metrics


def test_auroc_known_cases() -> None:
    assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 0.0
    assert auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5  # ties count half


@given(
    st.lists(st.floats(-10, 10), min_size=1, max_size=40),
    st.lists(st.floats(-10, 10), min_size=1, max_size=40),
)
def test_auroc_antisymmetric_and_bounded(a: list[float], b: list[float]) -> None:
    x, y = np.array(a), np.array(b)
    assert 0.0 <= auroc(x, y) <= 1.0
    assert auroc(x, y) == pytest.approx(1.0 - auroc(y, x))


def test_mahalanobis_basics() -> None:
    rng = np.random.default_rng(5)
    feats = rng.normal(size=(5000, 2))
    m = Mahalanobis(feats)
    assert m(feats.mean(axis=0, keepdims=True))[0] == pytest.approx(0.0, abs=1e-9)
    assert m(feats).mean() == pytest.approx(2.0, abs=0.1)  # E[d^2] = dimension for a Gaussian


# ----------------------------------------------------------------------------------- generators


@pytest.fixture(scope="module")
def tiny_data() -> tuple[Array, Array]:
    rng = np.random.default_rng(6)
    return data.sample(1500, rng), data.sample(200, rng)


def test_vae_shapes_finite_and_deterministic(tiny_data: tuple[Array, Array]) -> None:
    train, test = tiny_data
    v = train_vae(train, seed=0, steps=200)
    for out in (v.nll(test, k=16), v.recon_error(test)):
        assert out.shape == (200,) and np.isfinite(out).all()
    assert np.array_equal(v.nll(test, k=16, seed=1), v.nll(test, k=16, seed=1))
    assert v.latent_mean(test).shape == (200, 2)


def test_vae_nll_is_an_upper_bound_on_true_nll(tiny_data: tuple[Array, Array]) -> None:
    train, test = tiny_data
    v = train_vae(train, seed=0, steps=300)
    assert v.nll(test, k=32).mean() > (-data.true_logpdf(test)).mean()


def test_ddpm_shapes_finite_and_deterministic(tiny_data: tuple[Array, Array]) -> None:
    train, test = tiny_data
    d = train_ddpm(train, seed=0, steps=200)
    for out in (d.nll(test), d.dsm(test)):
        assert out.shape == (200,) and np.isfinite(out).all()
    assert np.array_equal(d.dsm(test, seed=1), d.dsm(test, seed=1))


def _gaussian_oracle_ddpm(steps: int, beta_max: float, s: float) -> DDPM:
    """A DDPM whose noise predictor is the exact optimum for data ~ N(0, s^2 I)."""
    d = DDPM(steps=steps, beta_max=beta_max)

    def eps_oracle(xt: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        a = d.abar[t_idx][..., None]
        return (1 - a).sqrt() * xt / (a * s**2 + 1 - a)

    d.eps = eps_oracle  # type: ignore[method-assign]
    return d


def test_ddpm_vlb_validated_against_analytic_gaussian_oracle() -> None:
    """With the exact noise predictor the model is essentially perfect, so the variational bound
    must sit above the true entropy, and its looseness must shrink as the step count grows."""
    s = 0.5
    x = np.random.default_rng(7).normal(0, s, (6000, 2))
    true = 0.5 * 2 * (math.log(2 * math.pi * s * s) + 1)
    gap_200 = _gaussian_oracle_ddpm(200, 0.05, s).nll(x).mean() - true
    gap_1000 = _gaussian_oracle_ddpm(1000, 0.02, s).nll(x).mean() - true
    assert gap_200 > 0 and gap_1000 > 0  # valid upper bounds
    assert gap_1000 < gap_200 / 2  # fixed-variance VLB looseness shrinks with T


def test_vae_class_exposes_expected_api() -> None:
    assert isinstance(VAE(), torch.nn.Module) and VAE().latent == 2
