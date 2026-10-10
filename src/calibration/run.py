"""Benchmark driver: train, score, calibrate, stress-test, and write results/summary.json + figures.

    python -m src.calibration.run --seeds 3 --out results/calibration

Everything is seeded. For seed s the training data, calibration pool and test sets come from
generators seeded off s, so seeds differ in both data and initialisation.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from . import data
from .conformal import coverage_ece, repeated_coverage
from .generators import DDPM, VAE, train_ddpm, train_vae
from .metrics import Mahalanobis, auroc

Array = npt.NDArray[np.float64]
Score = Callable[[Array], Array]

ALPHA = 0.1
N_CAL = 500
REPEATS = 300
SHIFTS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
ALPHAS = np.round(np.arange(0.05, 0.55, 0.05), 2)


def build_scores(vae: VAE, ddpm: DDPM, train: Array) -> dict[str, Score]:
    lat = Mahalanobis(vae.latent_mean(train))
    raw = Mahalanobis(train)
    return {
        "true_nll (oracle)": lambda x: -data.true_logpdf(x),
        "vae_recon_error": vae.recon_error,
        "vae_nll": vae.nll,
        "vae_latent_mahalanobis": lambda x: lat(vae.latent_mean(x)),
        "ddpm_denoising_error": ddpm.dsm,
        "ddpm_nll": ddpm.nll,
        "raw_mahalanobis": raw,
    }


def run_seed(seed: int, vae_steps: int, ddpm_steps: int) -> dict[str, Any]:
    rng = np.random.default_rng(10_000 + seed)
    train = data.sample(8000, rng)
    pool = data.sample(20000, rng)
    test = data.sample(5000, rng)
    ood = data.ood_sets(5000, rng)

    t0 = time.perf_counter()
    vae = train_vae(train, seed, steps=vae_steps)
    ddpm = train_ddpm(train, seed, steps=ddpm_steps)
    train_s = time.perf_counter() - t0
    scores = build_scores(vae, ddpm, train)

    out: dict[str, Any] = {
        "seed": seed,
        "train_seconds": train_s,
        "nll_test": {
            "true (entropy)": float(-data.true_logpdf(test).mean()),
            "vae_iwae_k128": float(vae.nll(test).mean()),
            "ddpm_vlb": float(ddpm.nll(test).mean()),
        },
        "scores": {},
    }
    cal_rng = np.random.default_rng(777 + seed)
    for name, fn in scores.items():
        pool_s, test_s = fn(pool), fn(test)
        cov = repeated_coverage(pool_s, test_s, ALPHA, N_CAL, REPEATS, cal_rng)
        mean_cov = [
            float(repeated_coverage(pool_s, test_s, float(a), N_CAL, 60, cal_rng).mean())
            for a in ALPHAS
        ]
        shift = {}
        for s in SHIFTS:
            shifted_s = fn(data.translate(test, s)) if s else test_s
            shift[str(s)] = float(repeated_coverage(pool_s, shifted_s, ALPHA, N_CAL, 60, cal_rng).mean())
        out["scores"][name] = {
            "coverage_alpha0.1_mean": float(cov.mean()),
            "coverage_alpha0.1_p5": float(np.percentile(cov, 5)),
            "coverage_alpha0.1_p95": float(np.percentile(cov, 95)),
            "frac_repeats_at_or_above_nominal": float((cov >= 1 - ALPHA).mean()),
            "coverage_ece": coverage_ece(1.0 - ALPHAS, np.array(mean_cov)),
            "coverage_by_nominal": dict(zip((f"{1 - a:.2f}" for a in ALPHAS), mean_cov, strict=True)),
            "shift_coverage": shift,
            "auroc": {k: auroc(test_s, fn(v)) for k, v in ood.items()},
        }
    return out


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def ms(vals: list[float]) -> dict[str, float]:
        return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}

    agg: dict[str, Any] = {"n_seeds": len(runs), "nll_test": {}, "scores": {}}
    for k in runs[0]["nll_test"]:
        agg["nll_test"][k] = ms([r["nll_test"][k] for r in runs])
    for name in runs[0]["scores"]:
        s0 = runs[0]["scores"][name]
        agg["scores"][name] = {
            "coverage_alpha0.1": ms([r["scores"][name]["coverage_alpha0.1_mean"] for r in runs]),
            "coverage_alpha0.1_p5": ms([r["scores"][name]["coverage_alpha0.1_p5"] for r in runs]),
            "coverage_alpha0.1_p95": ms([r["scores"][name]["coverage_alpha0.1_p95"] for r in runs]),
            "coverage_ece": ms([r["scores"][name]["coverage_ece"] for r in runs]),
            "shift_coverage": {
                s: ms([r["scores"][name]["shift_coverage"][s] for r in runs]) for s in s0["shift_coverage"]
            },
            "auroc": {k: ms([r["scores"][name]["auroc"][k] for r in runs]) for k in s0["auroc"]},
            "coverage_by_nominal": {
                k: ms([r["scores"][name]["coverage_by_nominal"][k] for r in runs])
                for k in s0["coverage_by_nominal"]
            },
        }
    return agg


def figures(agg: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(agg["scores"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0.5, 0.95], [0.5, 0.95], "k--", lw=1, label="ideal")
    for n in names:
        c = agg["scores"][n]["coverage_by_nominal"]
        ax.plot([float(k) for k in c], [v["mean"] for v in c.values()], marker="o", ms=3, lw=1, label=n)
    ax.set(xlabel="nominal coverage", ylabel="empirical coverage (in-distribution test)", title="Split-conformal coverage")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "coverage_vs_nominal.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for n in names:
        sc = agg["scores"][n]["shift_coverage"]
        ax.plot([float(s) for s in sc], [v["mean"] for v in sc.values()], marker="o", ms=3, lw=1, label=n)
    ax.axhline(1 - ALPHA, color="k", ls="--", lw=1)
    ax.set(xlabel="covariate shift (translation along x)", ylabel="empirical coverage", title="Coverage under covariate shift (nominal 0.90)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "shift_breakdown.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--vae-steps", type=int, default=6000)
    ap.add_argument("--ddpm-steps", type=int, default=20000)
    ap.add_argument("--out", type=Path, default=Path("results/calibration"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = []
    for s in range(args.seeds):
        print(f"seed {s} ...", flush=True)
        runs.append(run_seed(s, args.vae_steps, args.ddpm_steps))
    agg = aggregate(runs)
    (args.out / "per_seed.json").write_text(json.dumps(runs, indent=1))
    (args.out / "summary.json").write_text(json.dumps(agg, indent=1))
    figures(agg, args.out)
    print(json.dumps(agg["nll_test"], indent=1))


if __name__ == "__main__":
    main()
