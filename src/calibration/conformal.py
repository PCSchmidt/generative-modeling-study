"""Split conformal inference on a scalar nonconformity score (higher = more unusual)."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def conformal_threshold(cal_scores: Array, alpha: float) -> float:
    """Smallest q so that P(score_new <= q) >= 1 - alpha for exchangeable data.

    q is the ceil((n + 1)(1 - alpha))-th smallest calibration score; if that rank exceeds n the
    guarantee can only be met by the trivial set (q = +inf).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    n = len(cal_scores)
    if n == 0:
        raise ValueError("need at least one calibration score")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    return float(np.sort(cal_scores)[k - 1])


def coverage(test_scores: Array, q: float) -> float:
    """Fraction of test points whose score is within the conformal threshold."""
    return float(np.mean(test_scores <= q))


def coverage_curve(
    cal_scores: Array, test_scores: Array, alphas: Array
) -> tuple[Array, Array]:
    """(nominal, empirical) coverage at each alpha, with one calibration set."""
    nominal = 1.0 - alphas
    empirical = np.array([coverage(test_scores, conformal_threshold(cal_scores, a)) for a in alphas])
    return nominal, empirical


def coverage_ece(nominal: Array, empirical: Array) -> float:
    """Mean absolute gap between nominal and empirical coverage across levels."""
    return float(np.mean(np.abs(nominal - empirical)))


def repeated_coverage(
    pool_scores: Array,
    test_scores: Array,
    alpha: float,
    n_cal: int,
    repeats: int,
    rng: np.random.Generator,
) -> Array:
    """Coverage over many random calibration subsets drawn from `pool_scores`.

    Under exchangeability each draw's coverage is Beta(k, n_cal + 1 - k) distributed, so this
    shows the spread a single calibration set can produce, not just its mean.
    """
    out = np.empty(repeats)
    for i in range(repeats):
        idx = rng.choice(len(pool_scores), n_cal, replace=False)
        out[i] = coverage(test_scores, conformal_threshold(pool_scores[idx], alpha))
    return out
