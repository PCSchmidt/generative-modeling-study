# Protocol: do the two-dimensional calibration findings hold on real tabular data? (written before any model was trained on the real data)

## Why

[RESULTS.md](RESULTS.md) is a synthetic, two-dimensional benchmark. Its own "Not done" list names three gaps: real data, an energy-based model, and conformal variants. Its development protocol was informal (training length chosen on separate seeds, no protocol file pushed first). This study repeats the question on two real tabular datasets under a fixed protocol: protocol first, choices on development data only, test seeds run once, and every hypothesis judged against a criterion stated now.

## What I knew before writing this

From the 2-D study: split-conformal coverage held for every score; diffusion scores detected OOD points best and VAE reconstruction error was among the worst; a single Gaussian was a competitive baseline on some OOD sets. I have not looked at either real dataset beyond its shape and column names.

## Data (both public, both downloaded by scikit-learn from their original hosts)

- **California Housing** (20,640 rows, 8 features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude; the target is not used). *Natural covariate shift:* **in-distribution** = rows with Latitude below the median; two **shifted** sets from the northern half: **band 1** = latitude between the median and the 75th percentile, **band 2** = latitude at or above the 75th percentile. Preprocessing: `log` of AveRooms, AveBedrms, Population, AveOccup (heavy tails), then standardise with training-set statistics and clip to [-10, 10].
- **Covertype** (581,012 rows; the 10 quantitative columns: elevation, aspect, slope, four distances, three hillshade indices; 7 forest-cover classes). *Natural novelty:* for each seed, **3 of the 7 classes are drawn at random as OOD**; the other 4 are in-distribution. Preprocessing: standardise with training-set statistics, clip to [-10, 10].

**Development and test pools.** Rows of each dataset are assigned once, by the SHA-256 of the row index, to a development pool (30%) or a test pool (70%): a first byte below 77 means development. Development seeds (0, 1, 2) draw only from the development pool; test seeds (100 to 109) draw only from the test pool. Within a pool and a seed, in-distribution rows are split at random into train (60%), validation (20%), calibration (20%); for Covertype in-distribution rows are first subsampled to at most 30,000 train, 10,000 validation and 10,000 calibration/test rows, and 10,000 OOD rows are drawn from the three OOD classes. California Housing keeps all rows of its pool. Rows of Covertype that are neighbours in the original data are not independent, so random splits are optimistic for density models; this is a limit, stated now.

## Models and scores (all small, seeded, GPU where available)

| Model | Detail | Scores (higher = more unusual) |
| --- | --- | --- |
| VAE | MLP encoder/decoder (hidden 256, 2 layers), Gaussian decoder with learned diagonal variance, 8,000 steps, batch 512 | reconstruction error; IWAE negative log-likelihood (64 samples) |
| DDPM | T = 200, linear schedule, MLP noise predictor (3 layers), 20,000 steps, batch 512 | mean denoising error over all 200 steps (one fixed noise draw per step) |
| **EBM** (new) | MLP energy `E(x)` (3 layers, SiLU) trained by **noise-contrastive estimation** against a Gaussian with the training mean and **twice** the training covariance, with a learnable normaliser, 10,000 steps, batch 512 | the energy `E(x)` (no likelihood: the energy is unnormalised) |
| Gaussian | one full-covariance Gaussian | Mahalanobis distance (equivalently the negative log-density) |
| GMM | full-covariance Gaussian mixture, `k` chosen on development | negative log-density |

**Choices made on development data only, with no OOD labels used for any choice**: VAE latent dimension in {2, 4, 8} by validation IWAE NLL; DDPM hidden width in {128, 256} by validation denoising loss; EBM hidden width in {128, 256} by validation NCE loss; GMM `k` in {1, 2, 4, 8, 16} by validation NLL. Each is chosen per dataset using development seeds 0 to 2 (the setting with the best mean validation value across the three seeds), then frozen. Optimiser settings (Adam, cosine decay, learning rate 2e-3) are fixed and not tuned.

## Metrics

1. **Conformal coverage** (`alpha` = 0.1) on the in-distribution test rows with a calibration set of 1,000 rows drawn from the calibration split, averaged over 60 draws per seed; plus the same thresholds applied to each shifted or OOD set (the miss rate is `1 - coverage` there).
2. **AUROC** of each score separating the in-distribution test rows from each shifted or OOD set.
3. **Negative log-likelihood** on in-distribution test rows where a model has one (VAE IWAE, Gaussian, GMM).
4. Per-seed values for every quantity, and the mean and standard deviation over the 10 test seeds. Paired differences between scores use the same seeds; intervals are a bootstrap over seeds (2,000 resamples) and are reported as rough with n = 10.

## Hypotheses, each with a criterion stated now

- **T1 (the guarantee on real data).** For every score and dataset, mean in-distribution coverage lies in [0.88, 0.92]. Falsified if any is outside.
- **T2 (deep versus classical).** On Covertype, **no deep score** (VAE reconstruction, VAE NLL, DDPM denoising, EBM energy) exceeds the tuned **GMM** in mean AUROC by 0.02 or more. Falsified if one does, with a paired bootstrap interval above zero.
- **T3 (the EBM).** On Covertype the EBM energy has lower mean AUROC than the DDPM denoising score. Falsified otherwise.
- **T4 (reconstruction error).** Among the four deep scores (VAE reconstruction error, VAE NLL, DDPM denoising error, EBM energy), VAE reconstruction error has the lowest or second-lowest mean AUROC on **both** datasets. Falsified if it ranks third or fourth on either.
- **T5 (graded shift).** On California Housing, for every density-based score (VAE NLL, DDPM, EBM, Gaussian, GMM), coverage falls monotonically: in-distribution > band 1 > band 2, in at least 9 of 10 seeds.

## Pre-stated numerical expectations

- In-distribution coverage 0.89 to 0.91 for all scores.
- Covertype AUROC: the best score between 0.70 and 0.90; the GMM at least as good as the single Gaussian.
- California Housing AUROC against band 2: above 0.85 for density scores (latitude and longitude are features, so the shift is easy to see); VAE reconstruction error much lower.
- EBM training is the least stable; I expect at least one test seed where its AUROC is below 0.5 on one dataset.

## Limits, stated in advance

Two datasets, one family of shifts each; 10 seeds that share the same underlying rows, so seed variation understates dataset-level uncertainty. Random row splits leak neighbours. Small MLP models and a short training schedule; no conformal variant beyond plain split conformal (adaptive, localised, MAPIE and conformalised quantile regression are not covered); the DDPM and EBM have no usable likelihood here, so only the VAE, Gaussian and GMM report one. NCE with a covariance-matched Gaussian noise is one of several ways to train an EBM, chosen for stability. One author, who wrote the models and the protocol.

## Deviations

(none yet)
