# Results: calibration and novelty detection on real tabular data

Plan: [PROTOCOL_TABULAR.md](PROTOCOL_TABULAR.md), pushed before any model was trained on the real data, with one clarification logged before any run (split sizes). Choices were made on development seeds 0 to 2 from a separate 30% pool of rows; the ten test seeds (100 to 109) were run **once** on the other 70% (`results/tabular/test.json`, summarised in `test_summary.json`). Datasets: California Housing (8 features; in-distribution = southern half by latitude, shifted = two northern bands) and Covertype (10 quantitative features; per seed, 3 of 7 forest-cover classes drawn at random as novel).

> **Read this first.** Two datasets, ten seeds that share rows, small MLP models, one author who wrote the models and the protocol. The Covertype numbers vary a lot with *which* classes are novel (standard deviation 0.09 to 0.14 AUROC across seeds), so differences between scores are judged on paired seeds, not on the means alone. The tuned mixture hit the top of its grid (16 components), so "deep versus classical" below compares against a classical model that may be under-fitted.

## Frozen choices (development data only, no OOD labels used)

| Dataset | VAE latent size | DDPM width | EBM width | GMM components |
| --- | --- | --- | --- | --- |
| California Housing | 8 | 128 | 128 | 16 (grid edge) |
| Covertype | 4 | 256 | 256 | 16 (grid edge) |

## California Housing (southern half in-distribution; northern bands shifted)

| Score | Coverage in-dist (nominal 0.90) | AUROC band 1 | Coverage on band 1 | AUROC band 2 | Coverage on band 2 |
| --- | --- | --- | --- | --- | --- |
| VAE reconstruction error | 0.902 ± 0.011 | 0.972 ± 0.005 | 0.064 | 1.000 | 0.000 |
| VAE NLL (IWAE) | 0.903 ± 0.011 | 0.991 ± 0.001 | 0.029 | 1.000 | 0.000 |
| DDPM denoising error | 0.905 ± 0.009 | 0.981 ± 0.001 | 0.049 | 1.000 | 0.000 |
| EBM energy | 0.907 ± 0.008 | 0.986 ± 0.001 | 0.042 | 1.000 | 0.000 |
| One Gaussian | 0.900 ± 0.009 | 0.953 ± 0.001 | 0.098 | 0.999 | 0.000 |
| GMM (16 components) | 0.905 ± 0.012 | 0.982 ± 0.002 | 0.050 | 1.000 | 0.000 |

Test negative log-likelihood (nats): VAE 7.21 ± 0.20, GMM 7.42 ± 0.10, single Gaussian 10.15 ± 0.11. The shift is easy to see because latitude and longitude are features, so band 2 is separated perfectly by every score; band 1 separates the scores a little (a single Gaussian is the weakest, 0.953).

## Covertype (3 of 7 classes novel, drawn per seed)

| Score | Coverage in-dist | AUROC, in-dist vs novel classes | Coverage on novel rows |
| --- | --- | --- | --- |
| VAE reconstruction error | 0.900 ± 0.003 | 0.696 ± 0.133 | 0.702 |
| VAE NLL (IWAE) | 0.897 ± 0.007 | 0.644 ± 0.093 | 0.717 |
| DDPM denoising error | 0.898 ± 0.003 | **0.699 ± 0.111** | 0.638 |
| EBM energy | 0.898 ± 0.005 | 0.679 ± 0.100 | 0.651 |
| One Gaussian | 0.897 ± 0.004 | 0.631 ± 0.138 | 0.749 |
| GMM (16 components) | 0.899 ± 0.005 | 0.646 ± 0.118 | 0.741 |

Test NLL: VAE 0.69 ± 0.56, GMM 3.78 ± 0.37, single Gaussian 10.51 ± 0.39. Paired by seed against the GMM: DDPM +0.052 AUROC [+0.034, +0.074], better in 10 of 10 seeds; VAE reconstruction error +0.049 [+0.032, +0.063], 9 of 10; EBM +0.032 [+0.005, +0.060], 7 of 10; VAE NLL -0.003 [-0.024, +0.018], 5 of 10; single Gaussian -0.015 [-0.040, +0.006], 3 of 10.

## Hypotheses against their pre-stated criteria

| Hypothesis | Verdict |
| --- | --- |
| **T1.** Every score's in-distribution coverage is within [0.88, 0.92] on both datasets | **Held.** 0.897 to 0.907 |
| **T2.** No deep score beats the tuned GMM on Covertype by 0.02 AUROC or more | **Falsified.** DDPM (+0.052), VAE reconstruction error (+0.049) and EBM (+0.032) all do, with paired intervals above zero. Caveat: the GMM was capped at 16 components |
| **T3.** The EBM's Covertype AUROC is below the DDPM's | **Held** (0.679 against 0.699), a small difference |
| **T4.** VAE reconstruction error is lowest or second-lowest of the four deep scores on both datasets | **Falsified.** Lowest on California band 1, but second *highest* of the four on Covertype (0.696) |
| **T5.** On California Housing, coverage falls monotonically (in-distribution > band 1 > band 2) in at least 9 of 10 seeds for every density-based score | **Held.** 10 of 10 for all five |

## Pre-stated numerical expectations

| Expectation | Outcome |
| --- | --- |
| In-distribution coverage 0.89 to 0.91 | Yes |
| Covertype: the best score's AUROC between 0.70 and 0.90 | **Just missed:** 0.699 |
| Covertype: the GMM at least as good as the single Gaussian | Yes, +0.015, but the interval includes zero |
| California band 2 AUROC above 0.85 for density scores; reconstruction error much lower | Density scores yes (1.000); **reconstruction error is not lower** (1.000) |
| The EBM falls below 0.5 AUROC in at least one test seed on one dataset | **No.** Its worst seeds were 0.984 (California band 1) and 0.552 (Covertype) |

## What this shows and does not show

- **Shown:** split-conformal coverage holds on real tabular data for every score, including the weak ones, as theory says. Natural geographic shift destroys coverage in the order of the shift's size. On Covertype, novelty detection by any of these scores is poor: the best AUROC is 0.70, and 64 to 75% of the novel rows still fall inside the 90% region.
- **A better likelihood is not a better novelty detector.** The VAE's NLL is far better than the GMM's (0.69 against 3.78 nats) yet its AUROC is the same as the GMM's (0.644 against 0.646), and the VAE's reconstruction error and the DDPM's denoising error detect novel classes better than either likelihood. In the 2-D study, VAE reconstruction error was among the weakest scores; here it is competitive. The ranking of scores is not stable across data.
- **The energy-based model is not special:** trained by noise-contrastive estimation, it is competitive with the DDPM and better than the GMM on Covertype, and no seed failed to train (its worst Covertype AUROC was 0.552, which depends on which classes were novel).
- **Not shown:** that these rankings hold for other datasets, larger models, or other training choices; that the GMM would not close the gap with more components; any conformal variant beyond plain split conformal; tighter likelihoods for the DDPM and EBM.

## Process notes and limits

1. Protocol clarification (split sizes), logged before any training.
2. `scikit-learn` was added as a development dependency for the dataset loaders and the GMM.
3. The "Sensorless Drive" dataset named in the original roadmap could not be downloaded (UCI and OpenML were unreachable from this machine), so Covertype stands in as the real-sensor-like tabular set.
4. Rows of Covertype that are neighbours in the original data are not independent, so random splits favour the density models.
5. Seeds share the same underlying rows, so the standard deviations understate dataset-level uncertainty.
6. After the test run, typing and lint edits were made to `src/tabular` (type annotations, an unused-cast removal, a loop rewritten to use `.items()`); no computation changed, the unit tests pass, and the results were not regenerated.

## Reproduce

```bash
pip install -e ".[dev,study]"     # PyTorch with CUDA is recommended; the datasets download from their original hosts
python -m src.tabular.run dev     # development seeds 0-2: choose sizes and GMM k
python -m src.tabular.run test    # one-shot: test seeds 100-109; refuses to overwrite results/tabular/test.json
python -m src.tabular.analyze     # writes results/tabular/test_summary.json and prints the tables
```
