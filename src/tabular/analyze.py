"""Summarise results/tabular/test.json against the hypotheses of docs/PROTOCOL_TABULAR.md.

python -m src.tabular.analyze
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

OUT = Path("results/tabular")
SCORES = [
    "vae_recon_error",
    "vae_nll",
    "ddpm_denoising_error",
    "ebm_energy",
    "gaussian",
    "gmm",
]
DEEP = ["vae_recon_error", "vae_nll", "ddpm_denoising_error", "ebm_energy"]
DENSITY = ["vae_nll", "ddpm_denoising_error", "ebm_energy", "gaussian", "gmm"]


def boot_ci(d: np.ndarray, n: int = 2000) -> list[float]:
    rng = np.random.default_rng(0)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main() -> None:
    res = json.loads((OUT / "test.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for ds, seeds in res["per_seed"].items():
        ks = sorted(seeds)
        sets = [s for s in seeds[ks[0]]["metrics"][SCORES[0]]["auroc"]]
        table: dict[str, Any] = {}
        for sc in SCORES:
            cov_in = np.array([seeds[k]["metrics"][sc]["coverage"]["test"] for k in ks])
            row: dict[str, Any] = {
                "coverage_in": [float(cov_in.mean()), float(cov_in.std(ddof=1))]
            }
            for s in sets:
                a = np.array([seeds[k]["metrics"][sc]["auroc"][s] for k in ks])
                c = np.array([seeds[k]["metrics"][sc]["coverage"][s] for k in ks])
                row[f"auroc_{s}"] = [float(a.mean()), float(a.std(ddof=1))]
                row[f"coverage_{s}"] = [float(c.mean()), float(c.std(ddof=1))]
            table[sc] = row
        out[ds] = {"table": table, "sets": sets}
        # paired differences against the GMM, by seed
        diffs = {}
        for sc in DEEP + ["gaussian"]:
            for s in sets:
                d = np.array(
                    [
                        seeds[k]["metrics"][sc]["auroc"][s]
                        - seeds[k]["metrics"]["gmm"]["auroc"][s]
                        for k in ks
                    ]
                )
                diffs[f"{sc}_minus_gmm_{s}"] = {
                    "mean": float(d.mean()),
                    "ci95": boot_ci(d),
                    "seeds_better": int((d > 0).sum()),
                }
        out[ds]["paired_vs_gmm"] = diffs
        # T5 monotone coverage for California
        if ds == "california":
            mono = {}
            for sc in DENSITY:
                ok = 0
                for k in ks:
                    m = seeds[k]["metrics"][sc]["coverage"]
                    ok += int(m["test"] > m["band1"] > m["band2"])
                mono[sc] = ok
            out[ds]["monotone_seeds"] = mono
        out[ds]["nll"] = {
            n: [
                float(np.mean([seeds[k]["extra"]["test_nll"][n] for k in ks])),
                float(np.std([seeds[k]["extra"]["test_nll"][n] for k in ks], ddof=1)),
            ]
            for n in ("vae_iwae", "gmm", "gaussian_nll")
        }
        out[ds]["ebm_min_auroc"] = {
            s: float(min(seeds[k]["metrics"]["ebm_energy"]["auroc"][s] for k in ks))
            for s in sets
        }
        out[ds]["train_seconds_mean"] = float(
            np.mean([seeds[k]["extra"]["train_seconds"] for k in ks])
        )
        if ds == "covertype":
            out[ds]["ood_classes"] = {k: seeds[k]["info"]["ood_classes"] for k in ks}
    (OUT / "test_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for ds, o in out.items():
        print("==", ds, "choices", res["choices"][ds]["chosen"])
        for sc, r in o["table"].items():
            parts = [f"cov_in {r['coverage_in'][0]:.3f}±{r['coverage_in'][1]:.3f}"]
            for s in o["sets"]:
                parts.append(
                    f"{s}: AUROC {r[f'auroc_{s}'][0]:.3f}±{r[f'auroc_{s}'][1]:.3f} cov {r[f'coverage_{s}'][0]:.3f}"
                )
            print(f"  {sc:22s} " + " | ".join(parts))
        print("  NLL", {k: [round(x, 3) for x in v] for k, v in o["nll"].items()})
        print("  EBM min AUROC over seeds", o["ebm_min_auroc"])
        if "monotone_seeds" in o:
            print("  monotone coverage seeds (of 10)", o["monotone_seeds"])
        for k, v in o["paired_vs_gmm"].items():
            print(
                f"  {k:42s} {v['mean']:+.3f} {[round(x, 3) for x in v['ci95']]} better in {v['seeds_better']}/10"
            )


if __name__ == "__main__":
    main()
