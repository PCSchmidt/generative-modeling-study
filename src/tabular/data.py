"""Real tabular datasets, pools and seeded splits (docs/PROTOCOL_TABULAR.md)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
CACHE = Path(__file__).resolve().parents[2] / "data" / "sklearn"
CA_LOG_COLUMNS = (2, 3, 4, 5)  # AveRooms, AveBedrms, Population, AveOccup
CA_LAT = 6
DEV_BYTE_LIMIT = 77  # first byte of sha256(row index) below this -> development pool (30%)
CLIP = 10.0


def _load_ca() -> Array:
    from sklearn.datasets import fetch_california_housing

    CACHE.mkdir(parents=True, exist_ok=True)
    x = np.asarray(fetch_california_housing(data_home=str(CACHE)).data, dtype=np.float64)
    x[:, list(CA_LOG_COLUMNS)] = np.log(x[:, list(CA_LOG_COLUMNS)])
    return x


def _load_cov() -> tuple[Array, npt.NDArray[np.int64]]:
    from sklearn.datasets import fetch_covtype

    CACHE.mkdir(parents=True, exist_ok=True)
    b = fetch_covtype(data_home=str(CACHE))
    return np.asarray(b.data[:, :10], dtype=np.float64), np.asarray(b.target, dtype=np.int64)


def pool_of(n: int) -> npt.NDArray[np.bool_]:
    """True where a row belongs to the development pool."""
    return np.array([hashlib.sha256(str(i).encode()).digest()[0] < DEV_BYTE_LIMIT for i in range(n)])


@dataclass
class Task:
    """Standardised arrays for one seed: train/val/cal/test in-distribution rows and named shifted or OOD sets."""

    name: str
    train: Array
    val: Array
    cal: Array
    test: Array
    shifted: dict[str, Array]
    info: dict[str, object]


def _standardise(train: Array, *others: Array) -> tuple[Array, ...]:
    mu, sd = train.mean(0), train.std(0) + 1e-8
    return tuple(np.clip((a - mu) / sd, -CLIP, CLIP) for a in (train, *others))


def _split(rows: npt.NDArray[np.int64], rng: np.random.Generator, fracs: tuple[float, ...], caps: tuple[int, ...]) -> list[npt.NDArray[np.int64]]:
    rows = rng.permutation(rows)
    n = len(rows)
    sizes = [min(int(round(f * n)), c) for f, c in zip(fracs, caps, strict=True)]
    out, start = [], 0
    for s in sizes:
        out.append(rows[start : start + s])
        start += int(round(fracs[len(out) - 1] * n))
    return out


_UNCAPPED = (10**9,) * 4


def california(seed: int, dev: bool) -> Task:
    x = _load_ca()
    pool = pool_of(len(x)) == dev
    lat = x[:, CA_LAT]
    q50, q75 = np.quantile(lat, [0.5, 0.75])  # global quantiles, fixed by the protocol
    ind = np.where(pool & (lat < q50))[0]
    b1 = np.where(pool & (lat >= q50) & (lat < q75))[0]
    b2 = np.where(pool & (lat >= q75))[0]
    rng = np.random.default_rng(seed)
    tr, va, ca, te = _split(ind, rng, (0.5, 0.1, 0.2, 0.2), _UNCAPPED)
    a_tr, a_va, a_ca, a_te, a_b1, a_b2 = _standardise(x[tr], x[va], x[ca], x[te], x[b1], x[b2])
    return Task("california", a_tr, a_va, a_ca, a_te, {"band1": a_b1, "band2": a_b2}, {"n_in": len(ind), "n_b1": len(b1), "n_b2": len(b2)})


def covertype(seed: int, dev: bool) -> Task:
    x, y = _load_cov()
    pool = pool_of(len(x)) == dev
    rng = np.random.default_rng(seed)
    ood_classes = sorted(int(c) for c in rng.choice(np.arange(1, 8), 3, replace=False))
    in_mask = pool & ~np.isin(y, ood_classes)
    ind = np.where(in_mask)[0]
    tr, va, ca, te = _split(ind, rng, (0.6, 0.1, 0.1, 0.2), (30_000, 5_000, 5_000, 10_000))
    ood_rows = np.where(pool & np.isin(y, ood_classes))[0]
    ood = rng.permutation(ood_rows)[:10_000]
    a_tr, a_va, a_ca, a_te, a_ood = _standardise(x[tr], x[va], x[ca], x[te], x[ood])
    return Task("covertype", a_tr, a_va, a_ca, a_te, {"ood": a_ood}, {"ood_classes": ood_classes, "n_train": len(tr), "n_ood": len(ood)})


TASKS = {"california": california, "covertype": covertype}
