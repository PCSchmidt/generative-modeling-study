# generative-modeling-study: Evidence Notes

Companion to generative-modeling-study.md. Labels: Confirmed, Ran it, Per the docs, My reading, Ask the author, Author says.

## E1 Fact sheet
- Repo: PCSchmidt/generative-modeling-study, public, MIT, 0 stars, one release v0.2.0 (2026-10-13). Python, PyTorch. ~2,100 lines of Python in src/ and tests/; 40 tracked files. Confirmed.
- Kind: study (two studies plus an older digit study). Status: v0.2 portfolio project. Confirmed.
- Commits: 14. First 2026-04-18 (digit study, 2 commits), then 2026-10-10 (calibration benchmark, 4 commits plus merge) and 2026-10-13/14 (tabular study, 7 commits). Confirmed (git log).
- Tests: 43 (11 + 8 + 18 + 6). Ran it: all 43 pass in a scratch copy, 29 s, torch 2.14, numpy 2.5, Python 3.14, no git history needed. Type check on src/calibration, src/tabular and their tests: no issues (13 files). Ran it.
- Needs: torch; real-data study needs CUDA GPU (recommended) and dataset downloads (scikit-learn). No paid services, no keys. Per the docs.
- Deployed: no. Users: none shown. Confirmed (README, no deployment files beyond Dockerfile).
- GitHub description still says "19 tests" (E14). Confirmed.

## E2 Reading and running log
Read: README, DEMO_GUIDE, all docs, pyproject, Dockerfile, CI workflow, .gitignore, all src files except visualization.py (skimmed by name), tests/test_calibration.py and test_tabular.py in full, the start of test_models/test_training, the notebook cells and outputs, results JSON (summary, per_seed, dev_choice, test, test_summary), git log with credits, GitHub run list and release list.
Not read: notebook PDF; notebook figures; no .env read (none exists).
Ran (all in a scratch copy outside the clone, in a fresh venv): pytest (43 pass); mypy; synthetic benchmark with `--seeds 1` (CPU, about 100 s wall, 77 s training), compared with the committed per_seed.json seed 0: identical to the printed precision for NLLs, coverage and all AUROCs. Ran it. The run used torch 2.14, newer than any version the project could have used, so bit-for-bit repeat on CPU across versions is a finding. Real-data study not run (needs download and GPU). Docker not run. No paid calls. Network: pip installs only.
git status in clone: clean (nothing written there; the clone has a .git folder and was copied without it).

## E3 Claims check
Level L1 = doc matches committed result file; L3 = recomputed.
- CL-01 coverage 0.895-0.902 synthetic, 3 seeds: L1 against summary.json; seed 0 re-run L2. Supported.
- CL-02 coverage 0.897-0.907 on real tables: L1 against test_summary.json (California 0.900-0.907, Covertype 0.897-0.900). Supported.
- CL-03 Covertype best AUROC 0.70 (0.699): L3 from test.json (per-seed mean). Supported.
- CL-04 novel rows still inside 64-75%: L3 (0.638-0.749). Supported.
- CL-05 T2 paired: DDPM +0.052 (10/10), VAE recon +0.049 (9/10), EBM +0.032 (7/10): L3 for means and counts; DDPM bootstrap range recomputed as 0.034 to 0.074 (matches the docs' [+0.034, +0.074]). Supported.
- CL-06 DDPM bound looseness 3.2 / 0.9 nats: L2 (ran the oracle test code: 3.17, 0.91). Supported.
- CL-07 true NLL 0.77, VAE 1.58, DDPM 3.78: L1 and L2 (seed 0 rerun matches). Supported.
- CL-08 California band 2 AUROC 1.000, monotone 10/10: L1. Supported.
- CL-09 "two of five hypotheses falsified": L1 against RESULTS_TABULAR and protocol criteria; T4 verdict checked against the table (VAE recon is third of four on Covertype). Supported.
- CL-10 README "Tests: 43": Ran it. Supported. GitHub description "19 tests": contradicted (stale).
- CL-11 test-order claim: protocol commit 12:11 on 13 Oct, CI run created 16:11:30Z (= 12:11 local), dev frozen 12:43 (16:43:08Z), results 13:24 (17:24:38Z). L1 on times. Supported, with the limit that dates are author-controlled but CI start times match.
- CL-12 digit study "good disentanglement-quality balance" at beta 4: partly (loss numbers support the trade-off; no disentanglement metric exists in the code). Partly.
- CL-13 DEMO_GUIDE "production-grade project structure"; "evidence/ contains PNGs": the folder is git-ignored and absent. Partly/unsupported.
Count of claims checked: 13; one contradicted (stale), two partly.

## E4 Rule attacks (ran it, scratch copy)
- conformal_threshold: alpha 0 or 1 raises ValueError; empty raises ValueError; n=9, alpha 0.05 returns inf (tested); n=20 clean scores 1..20 returns 19.0; one NaN in the 20 returns 20.0 (NaN sorts last, so the threshold moves up silently); 19 NaNs of 20 returns NaN; one inf returns 20.0; a 2-D array raises TypeError.
- coverage with NaN test scores: NaN <= q is False, so they count as outside; silent.
- auroc: empty group raises; NaN input returns NaN.
- Re-running the test stage: refuses (code path read, not triggered because it would need results).
- Exchangeable random normals, n=500, alpha 0.1, 300 draws: mean 0.9006, 5th-95th percentile 0.875-0.926. Ran it.

## E5 Code map
src/models.py (conv Encoder, Decoder, BetaVAE), src/training.py (vae_loss, train_model, evaluate), src/visualization.py. src/calibration/: data.py (moons, exact density by quadrature, shift and OOD sets), generators.py (VAE, DDPM with IWAE and variational bound), conformal.py, metrics.py (AUROC from ranks, Mahalanobis), run.py. src/tabular/: data.py (hash-based pools, splits, tasks), models.py (VAE, DDPM, EBM with NCE), run.py (dev and one-shot test), analyze.py. Entry points: `python -m src.calibration.run`, `python -m src.tabular.run dev|test`, `python -m src.tabular.analyze`.

## E6 Model call sites
No external model calls. All models trained in-repo from random initialisation (src/models.py, src/calibration/generators.py, src/tabular/models.py). No pretrained weights, no prompts, no paid services.

## E7 Tests and CI
43 tests as above. Spot checks: (1) test_marginal_coverage_matches_exact_expectation asserts mean coverage equals ceil((n+1)(1-alpha))/(n+1) within 4.5 Monte-Carlo standard errors for random n, alpha: real behaviour. (2) test_ddpm_vlb_validated_against_analytic_gaussian_oracle asserts the bound is above the true entropy and that looseness shrinks with T: real. (3) test_ebm_nce_loss_is_two_log_two_when_the_model_equals_the_noise_density asserts the loss equals 2 ln 2 for a known case: real. Weak: test_vae_class_exposes_expected_api checks only the class type and latent size; digit-study tests use fake random data and check shapes and basic loss properties.
CI: matrix Python 3.10 and 3.12; installs CPU torch; mypy on calibration and tabular packages and their tests; pytest on all tests. Does not run benchmarks, the real-data study, ruff or the notebook. The digit code is not type-checked. One CI run failed on 520a4de (results commit) and later runs passed (reason not read from logs; log fetch returned nothing; My reading: typing, given the next commit is "Typing and lint fixes"). Coverage percentage not measured.

## E8 Plan-versus-results and history
See CL-11. Protocol states what the author knew from the 2-D study and that no real data had been inspected beyond shape and column names. One deviation logged before any run (split sizes). After the test run, typing and lint edits to src/tabular (disclosed in RESULTS_TABULAR process note 6); results not regenerated. Commit credits: an AI model is credited on 11 of 14 commits (all October non-merge commits); April commits and the merge carry none. Tool names deliberately not recorded in the guide per the author's instruction. No agent instruction files in the repo. October work is a handful of large commits.

## E9 Gates and approvals
None. The one-shot rule is a code check (refuses to overwrite test.json); a file deletion bypasses it. My reading.

## E10 Environment
pyproject gives lower bounds only (torch>=2.0, numpy>=1.24, ...). No lockfile. Results JSON do not record library versions. A fresh install (Python 3.14, torch 2.14) ran 43 tests and reproduced seed-0 synthetic numbers exactly. Undeclared: jupyter is needed for the notebook but not listed. scikit-learn is in dev and study extras, not the base dependencies.

## E11 Security and privacy
No .env files, no keys. Data public (MNIST, California Housing, Covertype); downloads from original hosts via scikit-learn; licences not checked. Secret scan not run with a tool; grep-level only (no secrets seen in files read). Ask the author for any history scan.

## E12 Shortcomings register
1. Mixture capped at grid edge (k=16 both tables). Impact: T2 verdict depends on it. Severity Medium. Disclosed. Options: extend grid (small effort), leave and document. Recommend extending the grid and re-running only the mixture (it is cheap; but the one-shot rule means a documented deviation).
2. Silent NaN handling in conformal threshold and coverage. Severity Low-Medium. Not disclosed. Option A: raise on non-finite input; option B: document. Recommend A.
3. Stale docs: GitHub description (19 tests), DEMO_GUIDE (19 tests, evidence/ PNGs absent, "production-grade"), pyproject description, RESULTS.md Not-done list. Severity Low. Partly disclosed. Fix by editing.
4. Digit study measures no disentanglement and makes visual claims. Severity Low-Medium. Not disclosed. Option: add a disentanglement metric or soften wording.
5. Seeds share rows; California calibration draws overlap (about 1,400 rows, draws of 1,000). Disclosed. Low.
6. Dev validation NLL for the California VAE (16.5 at latent 8) is much larger than the reported test NLL (7.21 plus or minus 0.20). Unexplained here; the dev and test pools are from the same data. Severity Unknown. Ask the author.
7. Covertype neighbouring rows leak. Disclosed. Low.
8. No monitoring, deployment or users. Deliberate scope.

## E13 Math detail
Conformal threshold: q = sorted(cal)[k-1], k = ceil((n+1)(1-alpha)); P(new <= q) = k/(n+1) >= 1-alpha under exchangeability with continuous ties-free scores; for n=500, alpha=0.1, k=451 and coverage across calibration draws is Beta(451, 50), sd about 0.0134. AUROC = Mann-Whitney U / (n_in * n_out), ties half. IWAE NLL = -log mean_k exp(log p(x|z) + log p(z) - log q(z|x)), k=128 (2-D) or 64 (tabular). DDPM bound: Ho et al. 2020 eq. 5. EBM: NCE logit = -E(x) - log c - log p_noise(x), noise N(mean, 2 cov). Paired differences use a bootstrap over 10 seeds, 2,000 resamples. Beta-VAE loss: BCE + beta KL.

## E14 Discrepancies
Docs vs code or state: GitHub description and DEMO_GUIDE "19 tests" vs 43; DEMO_GUIDE evidence/ files vs git-ignored absent folder; RESULTS.md says everything synthetic and lists the EBM and real data as not done (superseded by the tabular study; README is current); pyproject description names only the digit study; notebook test losses differ between two evaluations of the same models (101.56 vs 101.60, 91.11 vs 91.19, 143.73 vs 143.60, 181.58 vs 181.88) because evaluation samples the code and training is not seeded; the April pyproject named a back end that does not exist (setuptools.backends._legacy:_Backend), so install and CI failed until 2026-10-10 (per RESULTS.md, confirmed in the April commits).

## E15 Production readiness
Clean start: partly (README quickstart worked; versions unpinned). Checks block bad changes: partly (tests and type check only). Quality vs baseline: yes, in the study. Safe failure: n/a (study). Human review: no. Versioning: git only. Logs: result files only. Secrets: none. Placement: works locally. Gaps: hosting, drift monitoring, rolling calibration refresh, pinned versions, quality gate in CI, runbook.

## E16 Review dispositions
Author's tool-naming instruction applied: builder named only as "AI coding agents".

## E17 Completeness and backlog
Built and tested: digit study (notebook, 19 tests), synthetic benchmark, tabular study (code, protocol, results). Planned only (per README): Swiss Roll and warped manifolds, UCI Sensorless Drive, MAPIE and adaptive conformal, latent diffusion, experiment tracking, classification-style ECE.
Backlog. Fix: doc drift (small, low payoff, high honesty); NaN checks (small). Improve: uncapped mixture baseline (small-medium, medium payoff); pin versions and log them in results (small). Add: a CI gate that re-checks a small synthetic coverage number (small-medium); a rolling-calibration drift demo (medium). Order: fix docs and NaN checks, then the mixture grid, then the CI gate. Next milestone exit test: README, description and demo guide agree with the test count; non-finite scores raise.

## Diagram source
Images generated with matplotlib from results/calibration/summary.json and results/tabular/test_summary.json: learned-vs-rules, parts, coverage-shift, novelty.

## Revision log (reader-check pass, 2026-10-28)
Archived the prior draft as generative-modeling-study.v1draft.md. All ten reader-check fixes applied.
1. Plain conclusion added directly under the status line.
2. VAE, diffusion model, likelihood, conformal, bound, drift glossed at first use; nats, KL, IWAE, DDPM, NCE, entropy, oracle, latent, Mahalanobis, MLP dropped or replaced with plain words.
3. Methods and models chapters cut by about half (now one merged chapter, five bullets); engineering numbers (82.61, 1.65 million, 3.2 nats, step counts) removed.
4. Added the plain explanation of why a better drift detector shows a faster coverage drop (smoke-detector analogy).
5. Caveats stated once in the opening; repeats removed from results, shortcomings and recruiter text. Number-heavy passages cut to headline figures.
6. Predictions turned into a numbered list. True count checked against docs/PROTOCOL_TABULAR.md and docs/RESULTS_TABULAR.md: five hypotheses T1 to T5 (T1, T3, T5 held; T2, T4 failed). The two extra statements in v1 were loose numerical expectations (best AUROC 0.70, energy model below 0.5), now described as separate near-miss expectations, not predictions.
7. Digit study shrunk to one short paragraph in the results and one mention up top.
8. "Hard to fake" commit-timestamp wording softened to "some evidence of order, but one author controls the history". Evaluation-discipline strength tied to self-attested protocol.
9. Docker, JSON, lint, back end, agent instruction files, CI jargon dropped or glossed; "AI coding agents" only.
10. Figures regenerated from results/calibration/summary.json and results/tabular/test_summary.json and viewed: learned-vs-rules (text now inside boxes), parts (box 4 no longer says "k-th smallest"), coverage-shift (plain x-axis label, legend moved off the lines; oracle renamed exact-density reference), novelty ("coin flip" and "useless score" moved to legends, no collision with dashed lines).
Word count by wc -w: 4,608 (v1: 5,549).
