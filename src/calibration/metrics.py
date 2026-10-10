"""AUROC and Mahalanobis distance, written out so they can be tested against known cases."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

Array = npt.NDArray[np.float64]


def auroc(scores_in: Array, scores_out: Array) -> float:
    """P(score of an OOD point > score of an in-distribution point); ties count half.

    Computed from ranks (Mann-Whitney U), so it needs no threshold.
    """
    n_in, n_out = len(scores_in), len(scores_out)
    if n_in == 0 or n_out == 0:
        raise ValueError("need scores from both groups")
    ranks = rankdata(np.concatenate([scores_in, scores_out]))
    u = ranks[n_in:].sum() - n_out * (n_out + 1) / 2.0
    return float(u / (n_in * n_out))


class Mahalanobis:
    """Squared Mahalanobis distance to a Gaussian fitted on `features`."""

    def __init__(self, features: Array, ridge: float = 1e-6) -> None:
        self.mean = features.mean(axis=0)
        cov = np.cov(features, rowvar=False) + ridge * np.eye(features.shape[1])
        self.precision = np.linalg.inv(cov)

    def __call__(self, x: Array) -> Array:
        d = x - self.mean
        return np.asarray(np.einsum("ni,ij,nj->n", d, self.precision, d), dtype=np.float64)
