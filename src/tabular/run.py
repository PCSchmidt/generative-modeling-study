"""Driver for the real-data calibration study (docs/PROTOCOL_TABULAR.md).

    python -m src.tabular.run dev        # choose latent size / widths / GMM k on development data, writes dev_choice.json
    python -m src.tabular.run test       # ONE-SHOT: seeds 100-109 with the frozen choices; refuses to overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.mixture import GaussianMixture

from src.calibration.conformal import conformal_threshold
from src.calibration.metrics import Mahalanobis, auroc

from . import models as M
from .data import TASKS, Task

Array = npt.NDArray[np.float64]
OUT = Path("results/tabular")
DEV_SEEDS = (0, 1, 2)
TEST_SEEDS = tuple(range(100, 110))
ALPHA = 0.1
N_CAL = 1000
REPEATS = 60
GRID = {"latent": [2, 4, 8], "ddpm_hidden": [128, 256], "ebm_hidden": [128, 256], "gmm_k": [1, 2, 4, 8, 16]}
SCORES = ["vae_recon_error", "vae_nll", "ddpm_denoising_error", "ebm_energy", "gaussian", "gmm"]


def fit_gmm(train: Array, k: int, seed: int) -> GaussianMixture:
    return GaussianMixture(k, covariance_type="full", reg_covar=1e-4, random_state=seed, max_iter=200).fit(train)


def tune(dataset: str) -> dict[str, Any]:
    """Pick each hyper-parameter by mean validation loss over the development seeds (no OOD data used)."""
    val: dict[str, dict[Any, list[float]]] = {k: {v: [] for v in vs} for k, vs in GRID.items()}
    for seed in DEV_SEEDS:
        task = TASKS[dataset](seed, dev=True)
        for lat in GRID["latent"]:
            vae = M.train_vae(task.train, seed, lat)
            val["latent"][lat].append(float(vae.nll(task.val).mean()))
        for h in GRID["ddpm_hidden"]:
            ddpm = M.train_ddpm(task.train, seed, h)
            val["ddpm_hidden"][h].append(M.ddpm_val_loss(ddpm, task.val))
        for h in GRID["ebm_hidden"]:
            ebm = M.train_ebm(task.train, seed, h)
            val["ebm_hidden"][h].append(ebm.val_loss(task.val))
        for k in GRID["gmm_k"]:
            val["gmm_k"][k].append(float(-fit_gmm(task.train, k, seed).score(task.val)))
        print(dataset, "dev seed", seed, "done", flush=True)
    chosen = {k: min(vs, key=lambda v: float(np.mean(vs[v]))) for k, vs in val.items()}
    return {"chosen": chosen, "validation": {k: {str(v): float(np.mean(x)) for v, x in vs.items()} for k, vs in val.items()}}


def score_all(task: Task, seed: int, choice: dict[str, Any]) -> tuple[dict[str, dict[str, Array]], dict[str, Any]]:
    """Train every model on task.train; return scores[name][split] for cal/test/shifted sets, and extra numbers."""
    t0 = time.perf_counter()
    vae = M.train_vae(task.train, seed, choice["latent"])
    ddpm = M.train_ddpm(task.train, seed, choice["ddpm_hidden"])
    ebm = M.train_ebm(task.train, seed, choice["ebm_hidden"])
    gmm = fit_gmm(task.train, choice["gmm_k"], seed)
    maha = Mahalanobis(task.train)
    train_s = time.perf_counter() - t0
    fns = {
        "vae_recon_error": vae.recon_error,
        "vae_nll": vae.nll,
        "ddpm_denoising_error": ddpm.dsm,
        "ebm_energy": ebm.score,
        "gaussian": maha,
        "gmm": lambda x: -gmm.score_samples(x),
    }
    sets = {"cal": task.cal, "test": task.test, **task.shifted}
    scores = {name: {s: np.asarray(fn(arr), dtype=np.float64) for s, arr in sets.items()} for name, fn in fns.items()}
    extra = {
        "train_seconds": train_s,
        "test_nll": {
            "vae_iwae": float(scores["vae_nll"]["test"].mean()),
            "gmm": float(scores["gmm"]["test"].mean()),
            "gaussian_nll": float(
                0.5 * (scores["gaussian"]["test"] + task.train.shape[1] * np.log(2 * np.pi) + np.linalg.slogdet(np.cov(task.train, rowvar=False))[1]).mean()
            ),
        },
    }
    return scores, extra


def evaluate(scores: dict[str, dict[str, Array]], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for name, by_set in scores.items():
        cal = by_set["cal"]
        n_cal = min(N_CAL, len(cal))
        cov: dict[str, list[float]] = {s: [] for s in by_set if s != "cal"}
        for _ in range(REPEATS):
            q = conformal_threshold(cal[rng.choice(len(cal), n_cal, replace=False)], ALPHA)
            for s in cov:
                cov[s].append(float(np.mean(by_set[s] <= q)))
        out[name] = {
            "coverage": {s: float(np.mean(v)) for s, v in cov.items()},
            "auroc": {s: auroc(by_set["test"], by_set[s]) for s in by_set if s not in ("cal", "test")},
        }
    return out


def run_test() -> None:
    dest = OUT / "test.json"
    if dest.exists():
        sys.exit("results/tabular/test.json exists: the test seeds are one-shot (docs/PROTOCOL_TABULAR.md)")
    choices = json.loads((OUT / "dev_choice.json").read_text(encoding="utf-8"))
    res: dict[str, Any] = {"choices": choices, "per_seed": {}}
    for dataset in TASKS:
        res["per_seed"][dataset] = {}
        for seed in TEST_SEEDS:
            task = TASKS[dataset](seed, dev=False)
            scores, extra = score_all(task, seed, choices[dataset]["chosen"])
            res["per_seed"][dataset][str(seed)] = {"metrics": evaluate(scores, seed), "extra": extra, "info": task.info}
            print(dataset, seed, "done", f"{extra['train_seconds']:.0f}s", flush=True)
    dest.write_text(json.dumps(res, indent=1), encoding="utf-8")


def run_dev() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    choices = {ds: tune(ds) for ds in TASKS}
    (OUT / "dev_choice.json").write_text(json.dumps(choices, indent=1), encoding="utf-8")
    for ds, c in choices.items():
        print(ds, c["chosen"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "test"])
    a = ap.parse_args()
    {"dev": run_dev, "test": run_test}[a.stage]()


if __name__ == "__main__":
    main()
