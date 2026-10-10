"""Two interleaving half-moons with heteroscedastic Gaussian noise, and its exact density.

Because the generating process is known, the exact negative log-likelihood of any point can be
computed by numerical integration over the arc angle. That gives every learned model an
absolute reference: the entropy of the data itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class MoonsSpec:
    sigma0: float = 0.05  # noise std at the ends of each arc
    kappa: float = 1.0  # sigma(t) = sigma0 * (1 + kappa * sin t): noisiest at the top of the arc
    grid: int = 512  # quadrature points for the exact density


def _arc(moon: int, t: Array) -> Array:
    """Centre line of each moon, t in [0, pi]."""
    if moon == 0:
        return np.stack([np.cos(t), np.sin(t)], axis=-1)
    return np.stack([1.0 - np.cos(t), 0.5 - np.sin(t)], axis=-1)


def sigma_of_t(spec: MoonsSpec, t: Array) -> Array:
    return np.asarray(spec.sigma0 * (1.0 + spec.kappa * np.sin(t)), dtype=np.float64)


def sample(n: int, rng: np.random.Generator, spec: MoonsSpec | None = None) -> Array:
    spec = spec or MoonsSpec()
    moon = rng.integers(0, 2, n)
    t = rng.uniform(0.0, math.pi, n)
    centre = np.where(moon[:, None] == 0, _arc(0, t), _arc(1, t))
    noise = rng.normal(0.0, 1.0, (n, 2)) * sigma_of_t(spec, t)[:, None]
    return np.asarray(centre + noise, dtype=np.float64)


def true_logpdf(x: Array, spec: MoonsSpec | None = None) -> Array:
    """Exact log density of `sample`, by midpoint quadrature over t (error ~1e-4 nats)."""
    spec = spec or MoonsSpec()
    t = (np.arange(spec.grid) + 0.5) * math.pi / spec.grid
    sig = sigma_of_t(spec, t)  # (G,)
    dens = np.zeros(len(x))
    for moon in (0, 1):
        mu = _arc(moon, t)  # (G, 2)
        d2 = ((x[:, None, :] - mu[None, :, :]) ** 2).sum(-1)  # (N, G)
        comp = np.exp(-0.5 * d2 / sig**2) / (2.0 * math.pi * sig**2)
        dens += 0.5 * comp.mean(axis=1)
    return np.asarray(np.log(np.maximum(dens, 1e-300)), dtype=np.float64)


# ---- shifted / out-of-distribution sets ---------------------------------------------------


def translate(x: Array, shift: float) -> Array:
    """Covariate shift: move every point along +x by `shift`."""
    return np.asarray(x + np.array([shift, 0.0]), dtype=np.float64)


def uniform_box(n: int, rng: np.random.Generator) -> Array:
    return np.stack([rng.uniform(-1.5, 2.5, n), rng.uniform(-1.0, 1.5, n)], axis=-1)


def wide_gaussian(n: int, rng: np.random.Generator) -> Array:
    return np.asarray(rng.normal([0.5, 0.25], 1.0, (n, 2)), dtype=np.float64)


def ood_sets(n: int, rng: np.random.Generator, spec: MoonsSpec | None = None) -> dict[str, Array]:
    base = sample(n, rng, spec)
    return {
        "shifted_by_1.0": translate(base, 1.0),
        "uniform_box": uniform_box(n, rng),
        "wide_gaussian": wide_gaussian(n, rng),
    }
